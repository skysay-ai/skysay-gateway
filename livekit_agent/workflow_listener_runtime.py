# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Construct a declared source only from its fresh permission and leased secret."""
from __future__ import annotations

import copy
import re
from types import MappingProxyType

try:
    from .capabilities.workflow_sources import SourceAdapterDeclaration, qualified_source_adapter
    from .workflow_source_transport import FreshSourcePermission, SourceCredential, SourceOwner
    from .workflow_speech import SpeechRefused
except ImportError:
    from capabilities.workflow_sources import SourceAdapterDeclaration, qualified_source_adapter
    from workflow_source_transport import FreshSourcePermission, SourceCredential, SourceOwner
    from workflow_speech import SpeechRefused


# A region is a lower-case host label (the released Azure region grammar); the
# bootstrap proves it with the provider's own validator before it gets here.
_REGION = re.compile(r"[a-z][a-z0-9]{1,31}")


class WorkflowSourceCredentialRefused(SpeechRefused):
    reason = "workflow_source_credential_unusable"

    def __init__(self):
        super().__init__(self.reason)


def _component_pair(pair):
    return (type(pair) is tuple and len(pair) == 2
            and type(pair[0]) is str and pair[0] in ("stt", "tts")
            and type(pair[1]) is str and bool(pair[1].strip()) and len(pair[1]) <= 256)


class ManagedSourceKeys:
    """A copied, frozen set of explicitly injected account keys.

    ``regions`` is platform configuration beside a key (for example an Azure
    speech region): a proven region name per ``(component, provider)`` that
    also has a key here. It is not a secret and never part of a constructor.
    """
    __slots__ = ("_keys", "_regions")

    def __init__(self, values, *, regions=None):
        if type(values) is not dict or (regions is not None and type(regions) is not dict):
            raise WorkflowSourceCredentialRefused() from None
        copied = dict(values)
        regional = dict(regions or {})
        if not copied:
            raise WorkflowSourceCredentialRefused() from None
        for pair, secret in copied.items():
            if (not _component_pair(pair)
                    or type(secret) is not str or not secret.strip() or len(secret) > 16384):
                raise WorkflowSourceCredentialRefused() from None
        for pair, region in regional.items():
            if (not _component_pair(pair) or pair not in copied or type(region) is not str
                    or _REGION.fullmatch(region) is None):
                raise WorkflowSourceCredentialRefused() from None
        object.__setattr__(self, "_keys", MappingProxyType(copied))
        object.__setattr__(self, "_regions", MappingProxyType(regional))

    def __setattr__(self, name, value):
        raise AttributeError("managed source keys are frozen") from None

    def __delattr__(self, name):
        raise AttributeError("managed source keys are frozen") from None

    def __init_subclass__(cls, **kwargs):
        raise TypeError("managed source keys are final") from None

    def __repr__(self):
        return "<ManagedSourceKeys redacted>"

    def __getstate__(self):
        raise TypeError("managed source keys cannot be serialized") from None

    def _lookup(self, component, provider):
        return self._keys.get((component, provider))

    def _region(self, component, provider):
        return self._regions.get((component, provider))


def _resolve_source_credential(permission, credential, constructor, declaration, managed):
    if not isinstance(permission, FreshSourcePermission):
        raise WorkflowSourceCredentialRefused()
    owner = permission.owner
    if (not isinstance(owner, SourceOwner) or not isinstance(credential, SourceCredential)
            or not isinstance(declaration, SourceAdapterDeclaration) or type(constructor) is not dict
            or credential.source_attempt_id != owner.source_attempt_id
            or credential.binding_digest != owner.binding_digest
            or (credential.component, credential.provider, credential.model) !=
               (declaration.component, declaration.provider, declaration.model)
            or (constructor.get("provider"), constructor.get("model")) !=
               (declaration.provider, declaration.model)):
        raise WorkflowSourceCredentialRefused()
    region = None
    if credential.credential_source == "customer_byok":
        # A customer's key belongs to the customer's own resource; the
        # platform's region is not its endpoint. A regional source is
        # platform-managed only.
        if declaration.requires_region is not False:
            raise WorkflowSourceCredentialRefused() from None
        # A platform-only source never receives a customer key, and a
        # non-boolean policy is not an absent one.
        if declaration.platform_managed_only is not False:
            raise WorkflowSourceCredentialRefused() from None
        secret = credential.secret
    elif credential.credential_source == "platform_managed":
        if credential.secret is not None or type(managed) is not ManagedSourceKeys:
            raise WorkflowSourceCredentialRefused() from None
        secret = managed._lookup(credential.component, credential.provider)
        if declaration.requires_region is not False:
            region = managed._region(credential.component, credential.provider)
            if declaration.requires_region is not True or type(region) is not str:
                raise WorkflowSourceCredentialRefused() from None
    else:
        raise WorkflowSourceCredentialRefused() from None
    if type(secret) is not str or not secret.strip():
        raise WorkflowSourceCredentialRefused() from None
    if permission.owner is not owner:
        raise WorkflowSourceCredentialRefused() from None
    credential.require_live()
    return owner, secret, region


def require_source_credential(permission, credential, constructor, declaration, managed=None):
    """Run immediately after leasing, before consuming provider permission."""
    owner, _, _ = _resolve_source_credential(permission, credential, constructor, declaration, managed)
    return owner


def build_workflow_source(permission, credential, constructor, declaration, managed=None):
    """Single provider boundary; factories receive exactly one opaque secret."""
    validated_owner, secret, region = _resolve_source_credential(permission, credential, constructor, declaration, managed)
    selected = qualified_source_adapter(declaration.role, declaration.provider, declaration.model)
    if selected is not declaration:
        raise SpeechRefused("workflow_source_adapter_unqualified")
    consumed_owner = permission.consume()
    if consumed_owner is not validated_owner:
        raise WorkflowSourceCredentialRefused()
    regional = {"region": region} if declaration.requires_region is True else {}
    return declaration.factory(secret=secret, constructor=copy.deepcopy(constructor),
        client_reference_id=consumed_owner.source_attempt_id, **regional)


def enter_session_source(permission, credential, constructor, declaration, managed=None):
    """The live role's provider boundary: the call's open session is the provider.

    The same credential, qualification and one-time permission checks as
    ``build_workflow_source``, at the moment the session is asked to speak.
    Nothing is constructed: the declaration's factory built the call's one
    session at start, and a second would be a second provider connection.
    """
    if not isinstance(declaration, SourceAdapterDeclaration) or declaration.role != "live":
        raise WorkflowSourceCredentialRefused()
    validated_owner, secret, _region = _resolve_source_credential(permission, credential, constructor,
                                                                  declaration, managed)
    del secret
    selected = qualified_source_adapter(declaration.role, declaration.provider, declaration.model)
    if selected is not declaration:
        raise SpeechRefused("workflow_source_adapter_unqualified")
    consumed_owner = permission.consume()
    if consumed_owner is not validated_owner:
        raise WorkflowSourceCredentialRefused()
    return consumed_owner
