# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Explicit source-adapter inputs, separate from permission and from testing.

Two tiers (founder ruling 2026-10-21). A source is ADMITTED for a language
its capability-catalogue row declares (``capabilities.json``: a listener's
STT row, an authored voice's TTS row, a live source's ``live_sources`` row,
which may declare every language); no Skysay probe, record or release turns
a declared language on. A functional record (``QUALIFICATION_RECORDS``:
recorded evidence plus a lead ruling) says Skysay has TESTED that source in
that language: the dashboard's "Tested by Skysay" badge, our own guarantees
and the release smoke read it, and no admission seam does.

Registration is every declared adapter except a named operational withhold;
neither a configuration document nor a worker claim can populate it. Names
such as ``qualified_source_adapter`` keep their historical spelling and mean
"admitted". This file must import nothing outside the standard library at
module scope: the packaging contract loads it in isolation, and there the
catalogue is unreachable, so every language question refuses.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field


MAX_CALL_ATTEMPTS = 4096

# One frozen authored constructor for every authored provider. ``locale`` is
# the language profile's own key (the canonical primary tag a profile is
# stored under, for example ``si``), never read off the voice name; each
# adapter maps it to its provider's parameter or validates and ignores it.
AUTHORED_CONSTRUCTOR_FIELDS = frozenset({"provider", "model", "voice_id", "locale"})
_AUTHORED_LOCALE = re.compile(r"[a-z]{2,3}")


def is_authored_constructor(value) -> bool:
    """The one shape check every authored factory and the root plan share."""
    return (type(value) is dict and len(value) == len(AUTHORED_CONSTRUCTOR_FIELDS)
            and all(type(key) is str for key in value) and set(value) == AUTHORED_CONSTRUCTOR_FIELDS
            and all(_text(value[key]) for key in ("provider", "model", "voice_id"))
            and type(value["locale"]) is str and _AUTHORED_LOCALE.fullmatch(value["locale"]) is not None)


def authored_constructor(voice, locale) -> dict | None:
    """Project a frozen voice selection and its profile's locale, or ``None``.

    A voice that names a language must name this one: a voice resolved for
    another profile is not this profile's constructor.
    """
    if type(voice) is not dict or ("language" in voice and voice["language"] != locale):
        return None
    constructor = {"provider": voice.get("provider"), "model": voice.get("model"),
                   "voice_id": voice.get("voice_id"), "locale": locale}
    return constructor if is_authored_constructor(constructor) else None


@dataclass(frozen=True, slots=True)
class SourceAdapterDeclaration:
    role: str
    component: str
    provider: str
    model: str
    unit: str
    max_native_units: tuple[tuple[str, int | float], ...]
    identity_kind: str
    metric_identity_kind: str
    max_metric_count: int
    provider_reference_format: str
    factory: Callable = field(repr=False, compare=False)
    source_count_limit: int = field(kw_only=True)
    max_input_utf8: int | None = field(kw_only=True)
    resource_factory: Callable | None = field(default=None, kw_only=True, repr=False, compare=False)
    # The factory also receives the platform-managed ``region`` that the worker
    # bootstrap copied beside this provider's key. It is managed configuration,
    # never a constructor field and never written to a frozen document.
    requires_region: bool = field(default=False, kw_only=True)
    # The source runs only on the platform's own provider account: a customer
    # credential is refused before any secret is resolved for it.
    platform_managed_only: bool = field(default=False, kw_only=True)

    def meter(self) -> dict:
        return {"units": {"component": self.component, "provider": self.provider, "model": self.model,
                          "unit": self.unit, "max_native_units": dict(self.max_native_units)},
                "identity_kind": self.identity_kind, "metric_identity_kind": self.metric_identity_kind,
                "max_metric_count": self.max_metric_count, "provider_reference_format": self.provider_reference_format}


SOURCE_ADAPTERS: dict[tuple[str, str, str], SourceAdapterDeclaration] = {}


class WorkflowSourceAdapterUnavailable(ValueError):
    pass


# ``live`` is the persistent GPT-Live session speaking a workflow reply in the
# model's own voice and wording: a functional record of the languages a real
# call proved, keyed by provider and model only (every voice speaks the same
# languages), never an acoustic or exact-wording claim. How a live call hears
# the caller is ``LIVE_LISTENING`` below.
_ROLE_COMPONENT = {"listener": "stt", "authored": "tts", "live": "tts"}
_ROLE_KIND = {"listener": "functional", "authored": "functional", "live": "functional"}
QUALIFICATION_KINDS = frozenset(_ROLE_KIND.values())
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECORDED_AT = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")


def _text(value, maximum=256):
    return type(value) is str and bool(value) and value == value.strip() and len(value) <= maximum


@dataclass(frozen=True, slots=True)
class SourceQualification:
    """One recorded qualification of one source identity; never a capability claim.

    ``functional`` proves the adapter's contract: listener words, authored
    TTS, the live session speaking the listed locales. Locales use the workflow
    language codes stored on calls and deliveries (for example ``en``).
    """
    role: str
    component: str
    provider: str
    model: str
    kind: str
    evidence_sha256: str = field(kw_only=True)
    recorded_at: str = field(kw_only=True)
    authority: str = field(kw_only=True)
    voice_id: str | None = field(default=None, kw_only=True)
    locales: tuple[str, ...] = field(default=(), kw_only=True)

    def __post_init__(self):
        role = self.role
        if (role not in _ROLE_COMPONENT or self.component != _ROLE_COMPONENT[role]
                or not _text(self.provider) or not _text(self.model)
                or self.kind not in QUALIFICATION_KINDS or self.kind != _ROLE_KIND[role]
                or type(self.evidence_sha256) is not str or _SHA256.fullmatch(self.evidence_sha256) is None
                or type(self.recorded_at) is not str or _RECORDED_AT.fullmatch(self.recorded_at) is None
                or not _text(self.authority, 1024)
                or type(self.locales) is not tuple or len(self.locales) > 64
                or len(set(self.locales)) != len(self.locales)
                or any(not _text(locale, 64) or locale != locale.lower() for locale in self.locales)
                or not self.locales
                or self.voice_id is not None):
            raise ValueError("workflow source qualification record is malformed")

    def matches(self, role, provider, model, *, voice_id=None, locale=None):
        if (role, provider, model) != (self.role, self.provider, self.model):
            return False
        if locale is not None:
            # Every record answers only for the locales its evidence proved
            # (construction refuses a record that lists none). A live record
            # names no voice: ``voice_id`` is not part of it.
            return locale in self.locales
        return True


# Records are added only with recorded evidence (its SHA-256) and a lead
# ruling named in ``authority``. The first functional records (2026-10-20)
# come from one live functional smoke run through the production seams
# (``scripts/source_smoke``); ``evidence_sha256`` is that run's
# ``manifest.json``, which lists every case manifest's SHA-256. A source
# without a record here is untested (no badge); it is still admitted for the
# languages its catalogue row declares.
_SMOKE_2026_10_20 = dict(
    evidence_sha256="824a9e21ff6ccae3e04205df372221e5f7543bf90d51b9145d24248f2057a210",
    recorded_at="2026-10-20T02:58:03Z",
    authority=("founder-authorized functional smoke 2026-10-20; lead ruling: Claude lead implementer "
               "(takeover 2026-10-19), evidence archived skysay-reports/2026-10-20-source-smoke"),
)
QUALIFICATION_RECORDS: tuple[SourceQualification, ...] = (
    SourceQualification("listener", "stt", "deepgram", "nova-3", "functional",
                        locales=("en", "ta"), **_SMOKE_2026_10_20),
    SourceQualification("authored", "tts", "elevenlabs", "eleven_flash_v2_5", "functional",
                        locales=("en",), **_SMOKE_2026_10_20),
    SourceQualification("authored", "tts", "azure", "neural", "functional",
                        locales=("si", "en"), **_SMOKE_2026_10_20),
    SourceQualification("authored", "tts", "fishaudio", "s2-pro", "functional",
                        locales=("ta",), **_SMOKE_2026_10_20),
    # The Azure listener record (2026-10-20): its own functional smoke run on
    # the Linux voice image, heard against Azure neural (si) and ElevenLabs
    # (en) reference clips.
    SourceQualification("listener", "stt", "azure", "realtime", "functional", locales=("en", "si"),
        evidence_sha256="01e747b7cb880886c2c989e41b932a3fe83642f8caaba0f5389e25eee3655975",
        recorded_at="2026-10-20T06:01:50Z",
        authority=("founder-authorized functional smoke 2026-10-20 (Linux voice image; cases "
                   "listener-azure-realtime-en/si with Azure neural si and ElevenLabs en reference clips, all pass); "
                   "lead ruling: Claude lead implementer (takeover 2026-10-19), evidence archived "
                   "skysay-reports/2026-10-20-source-smoke")),
    # The first live record (2026-10-20): one founder-authorized live-role
    # workflow call on the local harness; ``evidence_sha256`` is its archived
    # evidence. It says the live session speaks English, never that it
    # reproduces configured wording.
    SourceQualification("live", "tts", "openai", "gpt-live-1", "functional", locales=("en",),
        evidence_sha256="423a6939529c7fb59d9438c079fef0c0c370a43e007c2e26d9c309909e1a81a1",
        recorded_at="2026-10-20T04:48:21Z",
        authority=("founder-authorized first live-role call on the local harness 2026-10-20 "
                   "(call_MCswkEWv5ox4rLm: 4 lines spoken in English, workflow advanced, barge-in journaled "
                   "interrupted, 4 live sources, no unsolicited speech); lead ruling: Claude lead implementer "
                   "(takeover 2026-10-19), evidence archived skysay-reports/2026-10-20-live-harness")),
    # The Soniox records (2026-10-20): two functional smoke runs on the voice
    # venv once the founder supplied the key; the listener heard the ElevenLabs
    # English reference clip. Each run is its own evidence archive.
    SourceQualification("authored", "tts", "soniox", "tts-rt-v2", "functional", locales=("en",),
        evidence_sha256="bfedc5e659af07da3589cedd1bb3749d3c3d064e42a7f5345afe091eda423235",
        recorded_at="2026-10-20T08:38:08Z",
        authority=("founder-authorized functional smoke 2026-10-20 (case authored-soniox-tts-rt-v2-en, pass; "
                   "the stream states no exact native usage); lead ruling: Claude lead implementer "
                   "(takeover 2026-10-19), evidence archived skysay-reports/2026-10-20-source-smoke-soniox")),
    SourceQualification("listener", "stt", "soniox", "stt-rt-v5", "functional", locales=("en",),
        evidence_sha256="b408fdc810706328a45f0c2a43a143c3324c5909e2ed80434a112e6d0da1f1f1",
        recorded_at="2026-10-20T09:51:07Z",
        authority=("founder-authorized functional smoke 2026-10-20 (case listener-soniox-stt-rt-v5-en heard "
                   "against the ElevenLabs en reference clip, pass; run with the per-turn caps the ruling sets); "
                   "lead ruling: Claude lead implementer "
                   "(takeover 2026-10-19), evidence archived skysay-reports/2026-10-20-source-smoke-soniox")),
    SourceQualification("listener", "stt", "openai", "gpt-live-transcribe", "functional", locales=("en",),
        evidence_sha256="280b94ac28167843869e6215986a728a5852ab718eea2e6db9bb0dfa9651abb3",
        recorded_at="2026-10-20T14:03:48Z",
        authority=("founder-authorized functional smoke 2026-10-20 on the Linux voice image; "
                   "lead ruling: Codex lead; English final, terminal acknowledgement and provider duration usage pass; "
                   "evidence archived skysay-reports/2026-10-20-shared-voice/source-smoke/20261020T140348Z-live")),
)

def source_qualification(role, provider, model, *, voice_id=None, locale=None):
    """The exact record for one source identity, or ``None``: the tested tier.

    Every record lists the locales its evidence proved, and its locale is
    checked whenever one is asked. No record names a voice. A record never
    admits anything; ``source_is_tested`` is the badge question.
    """
    for record in QUALIFICATION_RECORDS:
        if isinstance(record, SourceQualification) and record.matches(
                role, provider, model, voice_id=voice_id, locale=locale):
            return record
    return None


def source_is_tested(role, provider, model, locale) -> bool:
    """Whether Skysay has a functional record for this source in ``locale``.

    The "Tested by Skysay" badge, and nothing else: no admission seam asks it.
    A locale is required; the badge is per language.
    """
    if any(type(value) is not str or not value for value in (role, provider, model, locale)):
        return False
    record = source_qualification(role, provider, model, locale=locale)
    return (record is not None and record.kind == "functional"
            and record.component == _ROLE_COMPONENT.get(role) and record.voice_id is None)


def tested_locales(role, provider, model) -> tuple[str, ...]:
    """Every locale a functional record lists for this source, in record order (the badge set)."""
    if any(type(value) is not str or not value for value in (role, provider, model)):
        return ()
    found = []
    for record in QUALIFICATION_RECORDS:
        if (isinstance(record, SourceQualification) and (record.role, record.provider, record.model)
                == (role, provider, model) and record.kind == "functional"):
            found.extend(locale for locale in record.locales if locale not in found)
    return tuple(found)


def _catalogue():
    """The capability catalogue, or ``None`` where it cannot be reached.

    Imported only when a language is asked, never at module scope: the
    isolated packaging load has no package, so it gets ``None`` and every
    language question refuses there.
    """
    try:
        from .registry import load_registry
    except ImportError:
        return None
    return load_registry()


def declared_source_languages(role, provider, model):
    """What the catalogue declares for this source: its languages (a tuple), ``"any"``, or ``None`` (no row)."""
    catalogue = _catalogue()
    return None if catalogue is None else catalogue.source_language_scope(role, provider, model)


def source_declares_locale(role, provider, model, locale) -> bool:
    """The admission predicate: the catalogue row for this source declares ``locale``."""
    if any(type(value) is not str or not value for value in (role, provider, model, locale)):
        return False
    catalogue = _catalogue()
    return catalogue is not None and catalogue.declares_source_language(role, provider, model, locale)


# Declared adapters kept out of the registry for an operational reason, never
# a language one: each refuses at every seam until its reason is settled.
WITHHELD_SOURCE_ADAPTERS: dict[tuple[str, str, str], str] = {
    ("authored", "gemini", "gemini-3.1-flash-tts-preview"):
        "the workflow source's key environment name and the voice image are not settled",
}


# How a live-role call hears the caller: the one switch admission, run pinning
# and the source quote read. ``"listener"`` (founder ruling 2026-10-20): the
# GPT-Live session is muted and the workflow's qualified finite listener hears
# the caller, priced as its own STT line. ``"session"`` (the live session hears
# the caller, delegation-driven) is declared later in its own slice; it is not
# admitted here, so selecting it refuses every live workflow.
LIVE_LISTENING = "listener"
ADMITTED_LIVE_LISTENING = frozenset({"listener"})


def output_hears_through_listener(role) -> bool:
    """Whether a call with this output role hears through the finite listener.

    Every role but ``live`` always does. For ``live`` it is ``LIVE_LISTENING``,
    and a value outside ``ADMITTED_LIVE_LISTENING`` refuses rather than
    choosing a mode.
    """
    if role != "live":
        return True
    if LIVE_LISTENING not in ADMITTED_LIVE_LISTENING or LIVE_LISTENING not in ("listener", "session"):
        raise WorkflowSourceAdapterUnavailable("workflow_source_adapter_unqualified")
    return LIVE_LISTENING == "listener"


def live_output_admitted(provider, model, *, locale=None) -> bool:
    """Whether the live output source is admitted: registered and declared.

    With ``locale``, the catalogue's ``live_sources`` row declares it (``any``:
    every runtime language); without one, the row exists at all. Records are
    not asked; the voice is not part of the identity.
    """
    if (any(type(value) is not str or not value for value in (provider, model))
            or (locale is not None and (type(locale) is not str or not locale))):
        return False
    if locale is None:
        declaration = SOURCE_ADAPTERS.get(("live", provider, model))
        return (_admissible(declaration, ("live", provider, model))
                and declared_source_languages("live", provider, model) is not None)
    return source_admitted("live", provider, model, locale)


def _admissible(declaration, identity) -> bool:
    """A well-formed registered declaration of exactly ``identity`` that is not withheld."""
    return (isinstance(declaration, SourceAdapterDeclaration)
            and (declaration.role, declaration.provider, declaration.model) == identity
            and identity not in WITHHELD_SOURCE_ADAPTERS
            and declaration.component == _ROLE_COMPONENT.get(declaration.role))


def source_admitted(role, provider, model, locale) -> bool:
    """The server seams' admission question: registered, not withheld, and declared for ``locale``.

    Read directly from the registry and the catalogue (not through
    ``source_adapter_is_qualified``), so each seam keeps its own gate.
    """
    if any(type(value) is not str or not value for value in (role, provider, model, locale)):
        return False
    identity = (role, provider, model)
    return _admissible(SOURCE_ADAPTERS.get(identity), identity) and source_declares_locale(*identity, locale)


def live_output_qualification(provider, model, *, locale=None):
    """A functional live-output record for this provider and model, or ``None``: the tested tier.

    With ``locale`` it answers for that language; without it, whether any
    language is recorded. No voice is asked: a live record has none. The
    release smoke and the harness read it; admission asks ``live_output_admitted``.
    """
    if (any(type(value) is not str or not value for value in (provider, model))
            or (locale is not None and (type(locale) is not str or not locale))):
        return None
    for record in QUALIFICATION_RECORDS:
        if (isinstance(record, SourceQualification) and record.kind == "functional" and record.component == "tts"
                and (record.role, record.provider, record.model) == ("live", provider, model)
                and record.voice_id is None and (locale is None or locale in record.locales)):
            return record
    return None


def source_adapter_is_qualified(declaration: SourceAdapterDeclaration, *, voice_id=None, locale=None) -> bool:
    """Whether a declared adapter is admitted, and with ``locale`` for that language.

    Without a locale: a well-formed declaration that is not withheld. With
    one: also its catalogue row declares that locale. A record is never asked
    (that is ``source_is_tested``), and the voice is not part of a language
    declaration, so ``voice_id`` changes nothing.
    """
    if not isinstance(declaration, SourceAdapterDeclaration):
        return False
    identity = declaration.role, declaration.provider, declaration.model
    if not _admissible(declaration, identity):
        return False
    return locale is None or source_declares_locale(*identity, locale)


def qualified_source_adapter(role: str, provider: str, model: str, *, voice_id=None,
                             locale=None) -> SourceAdapterDeclaration:
    if any(type(value) is not str or not value for value in (role, provider, model)):
        raise WorkflowSourceAdapterUnavailable("workflow_source_adapter_unqualified")
    declaration = SOURCE_ADAPTERS.get((role, provider, model))
    if (not isinstance(declaration, SourceAdapterDeclaration)
            or role not in {"listener", "authored", "live"}
            or (declaration.role, declaration.provider, declaration.model) != (role, provider, model)
            or declaration.component != ("stt" if role == "listener" else "tts")
            or type(declaration.source_count_limit) is not int
            or not 1 <= declaration.source_count_limit <= MAX_CALL_ATTEMPTS
            or (type(declaration.max_input_utf8) is not int or not 1 <= declaration.max_input_utf8 <= 64000
                if role == "authored" else declaration.max_input_utf8 is not None)
            or not callable(declaration.factory)
            or type(declaration.requires_region) is not bool
            or type(declaration.platform_managed_only) is not bool
            or not source_adapter_is_qualified(declaration, voice_id=voice_id, locale=locale)):
        raise WorkflowSourceAdapterUnavailable("workflow_source_adapter_unqualified")
    return declaration


def _import_soniox_listener():
    try:
        from .. import workflow_soniox_listener as module
    except ImportError:
        import workflow_soniox_listener as module
    return module


def _import_azure_listener():
    try:
        from .. import workflow_azure_listener as module
    except ImportError:
        import workflow_azure_listener as module
    return module


def _import_deepgram_listener():
    try:
        from .. import workflow_deepgram_listener as module
    except ImportError:
        import workflow_deepgram_listener as module
    return module


def _import_openai_transcribe_listener():
    try:
        from .. import workflow_openai_transcribe_listener as module
    except ImportError:
        import workflow_openai_transcribe_listener as module
    return module


def _import_gemini_authored():
    try:
        from .. import workflow_gemini_authored as module
    except ImportError:
        import workflow_gemini_authored as module
    return module


def _import_elevenlabs_authored():
    try:
        from .. import workflow_elevenlabs_authored as module
    except ImportError:
        import workflow_elevenlabs_authored as module
    return module

def _import_azure_authored():
    try:
        from .. import workflow_azure_authored as module
    except ImportError:
        import workflow_azure_authored as module
    return module

def _import_fishaudio_authored():
    try:
        from .. import workflow_fishaudio_authored as module
    except ImportError:
        import workflow_fishaudio_authored as module
    return module

def _import_soniox_authored():
    try:
        from .. import workflow_soniox_authored as module
    except ImportError:
        import workflow_soniox_authored as module
    return module


def _import_live_output():
    try:
        from .. import workflow_live_output as module
    except ImportError:
        import workflow_live_output as module
    return module


# Static sibling imports, one per adapter module: the staged voice tree admits
# no module-loading machinery (the OSS export scan), and a name outside this
# table is a declaration error, not a computed import.
_WORKER_MODULES = {
    "workflow_soniox_listener": _import_soniox_listener,
    "workflow_azure_listener": _import_azure_listener,
    "workflow_deepgram_listener": _import_deepgram_listener,
    "workflow_openai_transcribe_listener": _import_openai_transcribe_listener,
    "workflow_gemini_authored": _import_gemini_authored,
    "workflow_elevenlabs_authored": _import_elevenlabs_authored,
    "workflow_azure_authored": _import_azure_authored,
    "workflow_fishaudio_authored": _import_fishaudio_authored,
    "workflow_soniox_authored": _import_soniox_authored,
    "workflow_live_output": _import_live_output,
}


def _worker_attribute(module, attribute):
    """Import a worker module only when a qualified source is built.

    The registry is imported by the backend and by the isolated packaging
    probe, neither of which may load a provider adapter or its SDK. The
    sibling is reached relative to this package in a checkout and flat on
    the voice image, exactly as the worker modules import each other.
    """
    return getattr(_WORKER_MODULES[module](), attribute)


def _lazy(module, attribute):
    def build(**fields):
        return _worker_attribute(module, attribute)(**fields)
    build.__qualname__ = f"lazy:{module}.{attribute}"
    return build


# Adapters that exist in this tree. Each is registered unless withheld
# (``WITHHELD_SOURCE_ADAPTERS``) and admitted per language by its catalogue
# row. Its meter caps are the ones the adapter's own tests exercise.
DECLARED_SOURCE_ADAPTERS: tuple[SourceAdapterDeclaration, ...] = (
    # Caps are one caller turn, not the call: the listener's PCM bound is 300 s
    # (MAX_SAMPLES at 24 kHz), and no audio-token rate the transport could
    # state reaches 1,000,000 input tokens or 100,000 text tokens for it. The
    # capture holds two overlapping sources while one finalizes, and the grant
    # layer counts each open source at its full cap, so a cap equal to the call
    # ceiling would refuse the second onset; these leave the ceiling to the
    # call. The stream states no native usage, so the caps bound capacity, not
    # settlement. Soniox record 2026-10-20: the ruling sets them.
    SourceAdapterDeclaration("listener", "stt", "soniox", "stt-rt-v5", "token",
        (("input_audio_tokens", 1_000_000), ("output_text_tokens", 100_000)), "adapter_stream", "stream_index", 128, "",
        _lazy("workflow_soniox_listener", "SonioxWorkflowListener"),
        source_count_limit=MAX_CALL_ATTEMPTS, max_input_utf8=None),
    # Deepgram bills socket seconds it never confirms on the stream; the cap is
    # the listener's own PCM bound (MAX_SAMPLES at 24 kHz), one caller turn.
    SourceAdapterDeclaration("listener", "stt", "deepgram", "nova-3", "audio_second",
        (("audio_second", 300),), "adapter_stream", "stream_index", 128, "",
        _lazy("workflow_deepgram_listener", "build_deepgram_listener"),
        source_count_limit=MAX_CALL_ATTEMPTS, max_input_utf8=None),
    # Azure realtime STT in its isolated child; it receives the platform region
    # beside the managed key (``requires_region``), so it never runs on a
    # customer key. Azure bills audio seconds the stream never states; the cap
    # is the listener's own PCM bound (MAX_SAMPLES at 24 kHz), one caller turn.
    SourceAdapterDeclaration("listener", "stt", "azure", "realtime", "audio_second",
        (("audio_second", 300),), "adapter_stream", "stream_index", 128, "",
        _lazy("workflow_azure_listener", "AzureWorkflowListener"),
        source_count_limit=MAX_CALL_ATTEMPTS, max_input_utf8=None, requires_region=True,
        platform_managed_only=True),
    # The completed item states billable seconds, including provider rounding.
    # Bound one turn by the adapter's 325-second absolute lifetime (300 seconds
    # of PCM plus finalization). The platform's own account only.
    SourceAdapterDeclaration("listener", "stt", "openai", "gpt-live-transcribe", "audio_second",
        (("audio_second", 300),), "adapter_stream",
        "stream_index", 128, "",
        _lazy("workflow_openai_transcribe_listener", "build_openai_transcribe_listener"),
        source_count_limit=MAX_CALL_ATTEMPTS, max_input_utf8=None, platform_managed_only=True),
    SourceAdapterDeclaration("authored", "tts", "gemini", "gemini-3.1-flash-tts-preview", "token",
        (("input_text_tokens", 100_000_000), ("output_audio_tokens", 100_000_000)),
        "adapter_stream", "stream_index", 128, "",
        _lazy("workflow_gemini_authored", "build_gemini_authored_source"),
        source_count_limit=MAX_CALL_ATTEMPTS, max_input_utf8=16384,
        resource_factory=_lazy("workflow_gemini_authored", "GeminiAuthoredWorkerPool")),
    # Finite in-process authored sources over the pinned TTS plugins. One
    # authored segment is at most 160 UTF-8 bytes: the gate's 15 s at 24 kHz.
    # The unit is what the released Cascade path bills for that provider.
    SourceAdapterDeclaration("authored", "tts", "elevenlabs", "eleven_flash_v2_5", "character",
        (("character", 160),), "adapter_stream", "stream_index", 128, "",
        _lazy("workflow_elevenlabs_authored", "build_elevenlabs_authored_source"),
        source_count_limit=MAX_CALL_ATTEMPTS, max_input_utf8=160),
    # Escaped SSML is at most six characters per input character (&quot;).
    SourceAdapterDeclaration("authored", "tts", "azure", "neural", "character",
        (("character", 960),), "adapter_stream", "stream_index", 128, "",
        _lazy("workflow_azure_authored", "build_azure_authored_source"),
        source_count_limit=MAX_CALL_ATTEMPTS, max_input_utf8=160, requires_region=True,
        platform_managed_only=True),
    # The text plus one separator per sentence, in UTF-8 bytes: the unit the
    # released Fish path bills and the source ledger stores (``utf8_byte``).
    SourceAdapterDeclaration("authored", "tts", "fishaudio", "s2-pro", "utf8_byte",
        (("utf8_byte", 320),), "adapter_stream", "stream_index", 128, "",
        _lazy("workflow_fishaudio_authored", "build_fishaudio_authored_source"),
        source_count_limit=MAX_CALL_ATTEMPTS, max_input_utf8=160),
    # Token caps are unproven; the first functional run sets them before any
    # record. Exact tokens exist only in the provider's usage log.
    SourceAdapterDeclaration("authored", "tts", "soniox", "tts-rt-v2", "token",
        (("input_text_tokens", 1000), ("output_audio_tokens", 100000)), "adapter_stream", "stream_index", 128, "",
        _lazy("workflow_soniox_authored", "build_soniox_authored_source"),
        source_count_limit=MAX_CALL_ATTEMPTS, max_input_utf8=160),
    # The call's persistent GPT-Live session speaks each reply; one source is
    # one reply's audio, under the same 120 s bound every output source has
    # (the source unit contract refuses a larger one). The call total stays
    # bounded by the source count and the reservation's minutes. The
    # platform's own OpenAI account only: never a customer key.
    SourceAdapterDeclaration("live", "tts", "openai", "gpt-live-1", "audio_second",
        (("audio_second", 120),), "provider_request", "stream_index", 128, "",
        _lazy("workflow_live_output", "build_live_workflow_output"),
        source_count_limit=MAX_CALL_ATTEMPTS, max_input_utf8=None, platform_managed_only=True),
)


def register_qualified_source_adapters() -> None:
    """The one writer of ``SOURCE_ADAPTERS``: every declaration not withheld.

    It reads no catalogue (the isolated packaging load has none) and no
    record; each language is asked of the catalogue when a seam asks it.
    """
    selected = {(item.role, item.provider, item.model): item for item in DECLARED_SOURCE_ADAPTERS
                if source_adapter_is_qualified(item)}
    SOURCE_ADAPTERS.clear()
    SOURCE_ADAPTERS.update(selected)


register_qualified_source_adapters()
