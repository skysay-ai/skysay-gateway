# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Copy explicitly selected platform source keys, never customer overlays."""
from collections.abc import Mapping

try:
    from .component_credentials import (_ENV_BY_COMPONENT, _MANAGED_SOURCE_REGION_ENV,
        _SCOPED_COMPONENT_VARIABLES, managed_source_environment_name)
    from .workflow_listener_runtime import ManagedSourceKeys, WorkflowSourceCredentialRefused
except ImportError:
    from component_credentials import (_ENV_BY_COMPONENT, _MANAGED_SOURCE_REGION_ENV,
        _SCOPED_COMPONENT_VARIABLES, managed_source_environment_name)
    from workflow_listener_runtime import ManagedSourceKeys, WorkflowSourceCredentialRefused


def managed_keys_for_sources(selected, *, environment, leased_components):
    if not isinstance(environment, Mapping) or not isinstance(leased_components, Mapping):
        raise WorkflowSourceCredentialRefused()
    leases = leased_components
    keys, regions = {}, {}
    for declaration, item in selected:
        owner = item.get("credential_source")
        if owner == "customer_byok":
            continue
        if declaration.component in leases:
            raise WorkflowSourceCredentialRefused()
        pair = (declaration.component, declaration.provider)
        if pair in keys:
            continue
        names = _ENV_BY_COMPONENT.get(pair)
        platform_name = managed_source_environment_name(declaration.role, declaration.provider, leased_components=leases)
        if platform_name is not None:
            names = (platform_name, (), None)
        if owner != "platform_managed" or names is None:
            raise WorkflowSourceCredentialRefused()
        # Scoped lease variables are not platform accounts. Missing platform
        # names stay refused until the shared provider declaration supplies one.
        candidates = [name for name in (names[0], *names[1]) if name not in _SCOPED_COMPONENT_VARIABLES]
        secret = None
        for name in candidates:
            value = environment.get(name)
            if isinstance(value, str) and value.strip():
                secret = value
                break
        if secret is None:
            raise WorkflowSourceCredentialRefused()
        keys[pair] = secret
        if getattr(declaration, "requires_region", False) is not False:
            regions[pair] = _managed_region(pair, environment, _MANAGED_SOURCE_REGION_ENV, declaration)
    return ManagedSourceKeys(keys, regions=regions) if keys else None


def _managed_region(pair, environment, names, declaration):
    """The proven platform region beside a regional source's managed key, or a refusal."""
    try:
        from .workflow_listener_runtime import WorkflowSourceCredentialRefused
        from .azure_speech import azure_region
    except ImportError:
        from workflow_listener_runtime import WorkflowSourceCredentialRefused
        from azure_speech import azure_region
    name = names.get(pair)
    value = environment.get(name) if name is not None else None
    # Azure's region grammar is the only one declared; another provider that
    # needs a region needs its own validator here first.
    if declaration.requires_region is not True or pair[1] != "azure" or not isinstance(value, str):
        raise WorkflowSourceCredentialRefused()
    try:
        return azure_region(value)
    except Exception:
        raise WorkflowSourceCredentialRefused() from None
