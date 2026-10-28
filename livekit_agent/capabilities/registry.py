# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Strict immutable reader for the checked-in voice capability artifact.

This module is intentionally dependency-free and is copied verbatim into the
API image as well as the separately built voice image.  It validates the
source document before exposing any selection, therefore a malformed or
partially edited capability file cannot become a permissive runtime fallback.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


class VoiceCapabilityError(RuntimeError):
    """The capability artifact is missing, malformed, or internally inconsistent."""


_COMPONENTS = frozenset({"stt", "tts"})
#: Components a *declaration* may cover.  Wider than ``_COMPONENTS`` by one:
#: the LLM has never had a capability record (it has no voice inventory and no
#: acoustic language limit), but a stack is only serveable in a language all
#: three of its components serve, so the LLM has to be able to say which
#: languages it is declared for.
_DECLARED_COMPONENTS = frozenset({"stt", "llm", "tts"})
_BILLING_UNITS = frozenset({"audio_second", "character", "utf8_byte", "token"})
#: Every credential-validation identity, and the one provider and billing unit
#: it is allowed to speak for.
#:
#: The enum and the provider/unit pinning used to be two literals two lines
#: apart in ``_validate_capability``.  They answer the same question -- "which
#: provider's evidence is this, and in what unit does it arrive" -- and a new
#: identity added to only one of them would either be rejected as unknown or,
#: worse, accepted with no unit pinned at all.  One table, read twice.
#:
#: A provider whose components bill in DIFFERENT units needs one identity per
#: unit rather than one per provider: Azure AI Speech meters real-time
#: transcription per audio second (``/sts/v1.0/issueToken`` is what proves a
#: key and region) and neural synthesis per character (``voices/list`` proves
#: the same pair plus the voice inventory), so a single ``azure_*`` identity
#: could not pin either one honestly.
_CREDENTIAL_VALIDATION_PROVIDER_UNIT: dict[str, tuple[str, str]] = {
    "soniox_models": ("soniox", "token"),
    "openai_models": ("openai", "audio_second"),
    "fish_wallet": ("fishaudio", "utf8_byte"),
    "cartesia_voices": ("cartesia", "character"),
    "azure_speech_token": ("azure", "audio_second"),
    "azure_speech_voices": ("azure", "character"),
}
_VOICE_INVENTORY_MODES = frozenset({"static", "provider_cache"})
#: The reviewed ways one STT construction can listen for several languages at
#: once.  A mode is a NAME FOR A MECHANISM, not for a provider: two providers
#: that identify the spoken language per result from a candidate list belong on
#: the same mode, and one provider offering both a fixed multi-language keyword
#: and a candidate list needs both.  Each entry has a validator branch below, a
#: worker builder, and -- for the modes that report which language they heard
#: -- a runtime evidence adapter; a value with no builder would admit a call
#: this platform cannot construct.
_SIMULTANEOUS_MODES = frozenset(
    {
        "language_multi",
        "language_identification",
        "strict_language_hints",
        "continuous_language_identification",
    }
)
_CONTINUOUS_IDENTIFICATION_MAX_CANDIDATES = 10
#: The modes whose provider construction names no language, so their
#: declaration may say ``language_scope: any`` instead of listing languages.
_OPEN_SCOPE_MODES = frozenset({"language_multi"})
_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_LANGUAGE = re.compile(r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
_VERSION = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_REVISION = re.compile(r"^[a-f0-9]{64}$")


def _root() -> Path:
    return Path(__file__).resolve().parent


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VoiceCapabilityError(message)


def _required_object(value: Any, *, where: str, keys: set[str]) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{where} must be an object")
    unknown = set(value) - keys
    _require(not unknown, f"{where} has unsupported keys: {sorted(unknown)!r}")
    return value


def _strings(value: Any, *, where: str, pattern: re.Pattern[str] | None = None, nonempty: bool = True) -> tuple[str, ...]:
    _require(isinstance(value, list), f"{where} must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        _require(isinstance(item, str) and item != "", f"{where}[{index}] must be a nonempty string")
        if pattern is not None:
            _require(pattern.fullmatch(item) is not None, f"{where}[{index}] is invalid")
        result.append(item)
    _require(bool(result) or not nonempty, f"{where} must not be empty")
    _require(len(result) == len(set(result)), f"{where} contains duplicates")
    return tuple(result)


def _validate_metadata(value: Any, *, where: str) -> None:
    metadata = _required_object(
        value,
        where=where,
        keys={"description", "tags", "gender", "accent_locale", "provenance"},
    )
    _require("provenance" in metadata, f"{where}.provenance is required")
    if "description" in metadata:
        _require(isinstance(metadata["description"], str), f"{where}.description must be text")
    if "tags" in metadata:
        _strings(metadata["tags"], where=f"{where}.tags", nonempty=False)
    if "gender" in metadata:
        _require(metadata["gender"] in {"female", "male", "nonbinary"}, f"{where}.gender is invalid")
    if "accent_locale" in metadata:
        _require(isinstance(metadata["accent_locale"], str) and _LANGUAGE.fullmatch(metadata["accent_locale"]) is not None, f"{where}.accent_locale is invalid")
    provenance = _required_object(
        metadata["provenance"],
        where=f"{where}.provenance",
        keys={"source", "observed_at"},
    )
    _require(isinstance(provenance.get("source"), str) and bool(provenance["source"]), f"{where}.provenance.source is required")
    _require(isinstance(provenance.get("observed_at"), str) and _VERSION.fullmatch(provenance["observed_at"]) is not None, f"{where}.provenance.observed_at is invalid")


#: STT adapters whose API refuses a bare ISO-639-1 code and therefore MUST
#: carry a locale for every language they can be asked to serve.
#:
#: Google publishes no bare code for any language this platform serves.  Both
#: of its published tables were parsed when this field was added: the v2 table
#: (which carries ``chirp_2``) lists exactly one bare entry across 170 codes
#: -- ``sw`` -- and the retired v1 table listed none at all.  So ``de``,
#: ``es``, ``fr``, ``ru``, ``et`` and the rest are not
#: "probably wrong" at Google, they are absent, and a session built with one
#: connects, debits the caller, and transcribes nothing.  Deepgram is the
#: opposite case and is deliberately not listed: ``nova-3`` takes the bare
#: ``et``, so its declaration below states the tags it wants rather than being
#: forced to invent regional ones.
#:
#: Matched against BOTH ``registry_name`` and ``provider``, which is not
#: belt-and-braces.  A declared row is IDENTIFIED by ``registry_name``/``model``
#: everywhere that matters -- admission (:meth:`declares_language`), the tag
#: lookup (:meth:`stt_locale`), row uniqueness, and the worker's legacy
#: adapter/model allowlist -- while ``provider`` names the vendor behind the
#: adapter and has almost no other reader.  Keying this requirement on
#: ``provider`` alone left a hole an adversarial review walked through: spell
#: that one field ``gcp`` and drop the table, and the artifact LOADS, admission
#: still answers yes for ``chirp_2``, and every German session goes out as a
#: bare ``de``.  Requiring a table when EITHER field names the adapter makes a
#: mislabel fail closed instead of silently disarming the guard.
#: Azure joins it on the provider's own answer rather than on documentation:
#: its ``speechtotext/.../locales`` endpoint returns 150 tags and not one bare
#: two-letter code, and a bare ``si`` closes the recognition socket outright.
#: Naming it here is what makes a capability row without a total locale table
#: fail to load, instead of failing on a caller's first Sinhala minute.
_LOCALE_REQUIRED_STT_ADAPTERS = frozenset({"google", "azure"})


def _validate_declared_stack_languages(value: Any, *, index: int) -> None:
    """One row of the provider-declared language table."""
    where = f"declared_stack_languages[{index}]"
    entry = _required_object(
        value,
        where=where,
        keys={
            "component",
            "provider",
            "registry_name",
            "model",
            "languages",
            "locales",
            "stt_interims_usable",
            "language_identification",
            "billing",
            "simultaneous_recognition",
        },
    )
    _require(entry["component"] in _DECLARED_COMPONENTS, f"{where}.component is unsupported")
    _require(
        ("billing" in entry) == (entry["component"] == "tts"),
        f"{where}.billing is required exactly on a declared TTS row",
    )
    if entry["component"] == "tts":
        _validate_declared_billing(entry["billing"], where=f"{where}.billing")
    for field in ("provider", "registry_name"):
        _require(
            isinstance(entry[field], str) and _ID.fullmatch(entry[field]) is not None,
            f"{where}.{field} is invalid",
        )
    _require(
        isinstance(entry["model"], str) and bool(entry["model"].strip()),
        f"{where}.model is required",
    )
    languages = _strings(entry["languages"], where=f"{where}.languages", pattern=_LANGUAGE)
    _validate_declared_locales(entry, where=where, languages=languages)
    _validate_declared_interims(entry, where=where)
    _validate_language_identification(entry, where=where)
    _validate_simultaneous_recognition(entry, where=where, languages=languages)


#: The one language scope a live (speech-to-speech) source may declare in
#: ``live_sources``: every language. A live provider that publishes no
#: per-language list for its model is admitted in any call language the
#: control plane accepts (founder ruling 2026-10-21). A named list, if one is
#: ever managed, is a ``languages`` field beside it, never an empty list.
LANGUAGE_SCOPE_ANY = "any"
_LIVE_SOURCE_KEYS = frozenset({"provider", "model", "language_scope", "basis"})


def _validate_live_source(value: Any, *, index: int) -> tuple[str, str]:
    """One row of ``live_sources``: a live output source and its declared language scope."""
    where = f"live_sources[{index}]"
    _require(isinstance(value, dict) and set(value) == _LIVE_SOURCE_KEYS, f"{where} has unsupported or missing keys")
    _require(isinstance(value["provider"], str) and _ID.fullmatch(value["provider"]) is not None,
             f"{where}.provider is invalid")
    _require(isinstance(value["model"], str) and bool(value["model"]) and value["model"] == value["model"].strip()
             and len(value["model"]) <= 128, f"{where}.model is required")
    _require(value["language_scope"] == LANGUAGE_SCOPE_ANY, f"{where}.language_scope is unsupported")
    _require(isinstance(value["basis"], str) and bool(value["basis"].strip()) and len(value["basis"]) <= 1024,
             f"{where}.basis is required")
    return value["provider"], value["model"]


def _validate_simultaneous_recognition(
    entry: Mapping[str, Any], *, where: str, languages: tuple[str, ...]
) -> None:
    """Validate an evidenced multi-language STT construction declaration.

    Ordinary per-language declarations answer whether a profile can run by
    itself. They are deliberately insufficient for Follow the caller: the
    worker must receive one provider construction which can listen for every
    selected language at the same time. Keeping the provider's fixed option
    alongside that declared set avoids a second adapter-local language list.
    """
    value = entry.get("simultaneous_recognition")
    if value is None:
        return
    _require(entry["component"] == "stt", f"{where}.simultaneous_recognition is only meaningful for STT")
    spec = _required_object(
        value,
        where=f"{where}.simultaneous_recognition",
        keys={"mode", "languages", "language_scope", "adapter_options", "max_candidates"},
    )
    _require(
        {"mode", "adapter_options"}.issubset(set(spec)),
        f"{where}.simultaneous_recognition is incomplete",
    )
    # This is intentionally a small typed protocol, not a bag of provider
    # kwargs carried from a customer request. New provider semantics require
    # a new explicitly reviewed registry mode and worker builder.
    mode = spec["mode"]
    _require(
        mode in _SIMULTANEOUS_MODES,
        f"{where}.simultaneous_recognition.mode is unsupported",
    )
    # Exactly one of a language list and an open scope. An open scope is only
    # declarable for the mode that sends the provider no language at all
    # (``language=multi``): a list there could only ever be a snapshot of what
    # the provider covered on the day it was typed. The modes that
    # send candidates or hints keep their list, because the list IS the
    # provider request.
    _require(
        ("languages" in spec) != ("language_scope" in spec),
        f"{where}.simultaneous_recognition needs exactly one of languages and language_scope",
    )
    if "language_scope" in spec:
        _require(
            spec["language_scope"] == LANGUAGE_SCOPE_ANY and mode in _OPEN_SCOPE_MODES,
            f"{where}.simultaneous_recognition.language_scope is unsupported",
        )
        simultaneous_languages = languages
    else:
        simultaneous_languages = _strings(
            spec["languages"],
            where=f"{where}.simultaneous_recognition.languages",
            pattern=_LANGUAGE,
        )
        _require(
            len(simultaneous_languages) >= 2,
            f"{where}.simultaneous_recognition.languages must contain at least two languages",
        )
        _require(
            set(simultaneous_languages).issubset(set(languages)),
            f"{where}.simultaneous_recognition.languages must be declared by this STT model",
        )
    options = _required_object(
        spec["adapter_options"],
        where=f"{where}.simultaneous_recognition.adapter_options",
        keys={
            "language",
            "language_hints",
            "language_hints_strict",
            "enable_language_identification",
            "language_id_mode",
            "candidate_locales",
        },
    )
    max_candidates = spec.get("max_candidates")
    if max_candidates is not None:
        _require(
            isinstance(max_candidates, int)
            and not isinstance(max_candidates, bool)
            and max_candidates >= 2,
            f"{where}.simultaneous_recognition.max_candidates is invalid",
        )
    if mode == "language_multi":
        _require(
            options == {"language": "multi"} and max_candidates is None,
            f"{where}.simultaneous_recognition.adapter_options is invalid",
        )
    elif mode == "language_identification":
        _require(
            options == {"enable_language_identification": True}
            and max_candidates is None,
            f"{where}.simultaneous_recognition.adapter_options is invalid",
        )
    elif mode == "continuous_language_identification":
        _validate_continuous_language_identification(
            entry,
            options,
            where=where,
            simultaneous_languages=simultaneous_languages,
            max_candidates=max_candidates,
        )
    else:
        _require(
            set(options) == {"language_hints", "language_hints_strict", "enable_language_identification"}
            and options["language_hints_strict"] is True
            and options["enable_language_identification"] is True,
            f"{where}.simultaneous_recognition.adapter_options is invalid",
        )
        _require(max_candidates is None, f"{where}.simultaneous_recognition.max_candidates is invalid")
        hints = _strings(
            options["language_hints"],
            where=f"{where}.simultaneous_recognition.adapter_options.language_hints",
            pattern=_LANGUAGE,
        )
        _require(
            tuple(hints) == simultaneous_languages,
            f"{where}.simultaneous_recognition adapter hints must exactly match declared languages",
        )


def _validate_continuous_language_identification(
    entry: Mapping[str, Any],
    options: Mapping[str, Any],
    *,
    where: str,
    simultaneous_languages: tuple[str, ...],
    max_candidates: Any,
) -> None:
    """The candidate-locale form of the same declaration.

    Structurally unlike the other two modes, which is why it is a mode of its
    own rather than a widened option bag.  ``language_multi`` sends one fixed
    keyword and ``strict_language_hints`` sends bare language codes; this mode
    sends the provider a LIST OF LOCALES and asks it to say, per final, which
    of them it heard.  The provider then reports a locale back, so the row has
    to name the exact tags it will listen with -- and those tags already exist
    on the row, in the ``locales`` table every locale-refusing adapter must
    carry.

    Copied beside the language set rather than resolved from that table at
    construction time, for the reason the neighbouring hint list is copied:
    the frozen declaration is what a reviewer, the control plane and the
    worker all read, and a construction that could only be reconstructed by
    re-deriving it is a construction nobody can check.  What keeps the copy
    from becoming a second authority is the equality below -- a candidate tag
    that disagrees with the row's own locale for that language fails the load,
    so the two cannot drift apart in the direction that matters (a socket
    opened on a tag the rest of the platform does not use).
    """
    _require(
        set(options) == {"language_id_mode", "candidate_locales"}
        and options["language_id_mode"] == "continuous",
        f"{where}.simultaneous_recognition.adapter_options is invalid",
    )
    # At-start identification decides once and never revisits, which is the
    # wrong shape for a call: a caller who opens in English and switches would
    # be transcribed as English for the rest of the conversation.  Only the
    # continuous form is declarable, and naming it in the artifact keeps the
    # worker from having to infer which one was meant.
    candidates = options["candidate_locales"]
    _require(
        isinstance(candidates, dict),
        f"{where}.simultaneous_recognition.adapter_options.candidate_locales must be an object",
    )
    _require(
        tuple(sorted(str(key) for key in candidates)) == tuple(sorted(simultaneous_languages)),
        f"{where}.simultaneous_recognition candidate locales must cover exactly the declared languages",
    )
    # The provider's own ceiling on a continuous-identification candidate set
    # (learn.microsoft.com/azure/ai-services/speech-service/language-identification,
    # read 2026-09-01: continuous identification accepts at most ten candidate
    # languages, at-start at most four).  The declaration publishes that
    # request-time ceiling beside the wider discoverable inventory;
    # ``simultaneous_stt_config`` enforces it against the exact selected set.
    _require(
        max_candidates == _CONTINUOUS_IDENTIFICATION_MAX_CANDIDATES,
        f"{where}.simultaneous_recognition must declare the continuous identification candidate limit",
    )
    locales = entry.get("locales")
    _require(
        isinstance(locales, dict),
        f"{where}.simultaneous_recognition requires this row's locale table",
    )
    for language in simultaneous_languages:
        candidate = candidates[language]
        _require(
            isinstance(candidate, str) and _LANGUAGE.fullmatch(candidate) is not None,
            f"{where}.simultaneous_recognition.adapter_options.candidate_locales[{language}] is invalid",
        )
        _require(
            candidate == locales.get(language),
            f"{where}.simultaneous_recognition candidate locale for {language} disagrees with this row's locale table",
        )


def _validate_declared_locales(
    entry: Mapping[str, Any],
    *,
    where: str,
    languages: tuple[str, ...],
    allow_tts_locales: bool = False,
) -> None:
    """The provider-specific tag each declared language is sent to STT as.

    Kept beside the declaration rather than in either consumer.  Before this
    field, the phone path and the browser demo each carried a hand-typed map
    of the handful of languages whose STT tag differs from the base code, and
    both were incomplete in the same direction: adding a language to a Google
    model widened what could be *offered* without widening what could be
    *recognised*, so the session connected, spent the caller's money, and
    heard nothing.  A declaration and its locale now move together or the
    image refuses to load.

    Three rules, all enforced here rather than only in ``schema.json`` (this
    stdlib reader is the gate that actually runs in both images):

    * A locale table belongs to an STT row.  An LLM or TTS declaration has no
      transcription tag to carry, and accepting one would create a second
      place to look for an answer neither consumer reads.
    * A row naming an adapter in :data:`_LOCALE_REQUIRED_STT_ADAPTERS`, by
      either ``registry_name`` or ``provider``, must have one.
    * A table that exists must be TOTAL over the languages its model can
      actually be HANDED, and carry nothing else.  Partial tables are how the
      two hand-typed maps drifted: "only the languages whose tag differs" is
      an invitation to forget one, and the forgotten one fails silently.

    The table covers exactly this model's declared languages. Historical
    language metadata cannot authorize an undeclared language or locale.
    """
    component = entry["component"]
    adapter_names = {str(entry["registry_name"]).lower(), str(entry["provider"]).lower()}
    locale_required = component == "stt" and bool(adapter_names & _LOCALE_REQUIRED_STT_ADAPTERS)
    if "locales" not in entry:
        _require(not locale_required, f"{where}.locales is required for this STT adapter")
        return
    # A DECLARATION carries the table for transcription only -- an LLM has no
    # tag and the pre-registry TTS adapters take bare codes -- and that
    # restriction is kept. A CAPABILITY may carry one for synthesis too: Azure
    # wants a tag in its SSML ``xml:lang`` for the same reason it wants one on
    # the wire, and answering that from a second table beside the adapter is
    # the duplication this field was introduced to delete.
    _require(
        component == "stt" or allow_tts_locales,
        f"{where}.locales is only meaningful for an STT declaration",
    )
    locales = entry["locales"]
    _require(isinstance(locales, dict), f"{where}.locales must be an object")
    for language, locale in locales.items():
        _require(
            isinstance(language, str) and _LANGUAGE.fullmatch(language) is not None,
            f"{where}.locales has an invalid language key",
        )
        _require(
            isinstance(locale, str) and _LANGUAGE.fullmatch(locale) is not None,
            f"{where}.locales[{language!r}] is not a valid language tag",
        )
    if locale_required:
        for language, locale in locales.items():
            # A locale that IS the bare code is a no-op dressed as an answer.
            # It satisfies totality, it is truthy, and it therefore walks
            # straight past the consumers' fail-closed refusal -- so an
            # adapter that refuses bare codes gets one anyway, from a table
            # written to prevent exactly that.  Adversarial review built it
            # and it is a realistic mistake, not a contrived one: the Deepgram
            # row's deliberate ``"et": "et"`` sits two rows above the Google
            # ones in the same file, and it is what a copy-paste starts from.
            # (Google's one bare v2 code is ``sw``, which nothing declares; a
            # row that ever needs it must name the exception here rather than
            # weaken the rule.)
            _require(
                locale != language,
                f"{where}.locales[{language!r}] is the bare code this adapter refuses",
            )
    expected = set(languages)
    _require(
        set(locales) == expected,
        f"{where}.locales must name exactly the languages this model can be asked to serve",
    )


def _validate_declared_interims(entry: Mapping[str, Any], *, where: str) -> None:
    """Whether this STT model's results arrive in time to count words against.

    Required on every STT declaration and refused on any other, for the same
    reason ``locales`` is: it answers a question only a transcriber has, and a
    row that may omit it is a row where "no record" and "measured false" look
    identical to a reader.  Stating it makes adding an STT model a decision
    rather than a default.

    It is a MEASUREMENT of the adapter's result cadence, not a claim that the
    provider supports streaming.  ``scripts/stt_interim_latency_probe.py`` is
    what produces it, and the distinction matters because the two answers
    differ: Google's streaming API does stream, but the models this platform
    dispatches (``chirp_2``) return their content late
    enough that a word requirement would refuse ordinary interruptions rather
    than only the wordless ones.  The Google row is therefore ``false``
    until somebody measures otherwise, and no consumer may read that as a
    statement about Google in general.

    Kept off the ``capabilities`` rows deliberately.  A registry-backed
    capability that carries no measurement reads as unusable through
    :meth:`stt_interims_usable`, which is the fail-closed direction: the
    consumer falls back to the duration floor alone, which is exactly what
    every managed call ran before this field existed.
    """

    present = "stt_interims_usable" in entry
    if entry["component"] != "stt":
        _require(not present, f"{where}.stt_interims_usable is only meaningful for an STT declaration")
        return
    _require(present, f"{where}.stt_interims_usable is required for an STT declaration")
    _require(
        isinstance(entry["stt_interims_usable"], bool),
        f"{where}.stt_interims_usable must be true or false",
    )


_LANGUAGE_IDENTIFICATION = ("none", "candidates", "any")


def _validate_language_identification(
    entry: Mapping[str, Any], *, where: str
) -> None:
    """How this transcriber is told which languages a call may contain.

    Three shapes, because there are three things an adapter's API can do with
    more than one language, and the worker cannot tell them apart from a
    provider's name:

    * ``none`` -- the adapter takes one language and one only.  A call opens
      on the locale it speaks.
    * ``candidates`` -- the adapter takes a LIST and compares the audio
      against every member.  The list is the agent's own configured
      languages, so an agent published for two languages hears both.
    * ``any`` -- the adapter has no list to give.  It either detects across
      everything it supports or it does not detect at all, which is a
      provider-wide switch rather than a per-call set.

    Required on every STT row and refused elsewhere, like ``locales`` and
    ``stt_interims_usable`` above: a transcriber that may omit it is one
    where "takes one language" and "nobody checked" read the same.

    ``candidates`` carries its own ceiling and its own locale table, because
    both are things the LIST needs and neither can be inferred.  A row
    declaring the shape without the number would be a promise to hand a
    provider as many tags as an agent happens to name, and both review
    engines independently built the same live-call failure out of that.
    Where a row ALSO declares a simultaneous ceiling the two must agree, so
    that a reader never has to ask which of them the socket obeys.
    """

    present = "language_identification" in entry
    if entry["component"] != "stt":
        _require(
            not present,
            f"{where}.language_identification is only meaningful for an STT declaration",
        )
        return
    _require(present, f"{where}.language_identification is required for an STT declaration")
    declaration = _required_object(
        entry["language_identification"],
        where=f"{where}.language_identification",
        keys={"mode", "max_candidates"},
    )
    mode = declaration.get("mode")
    _require(
        isinstance(mode, str) and mode in _LANGUAGE_IDENTIFICATION,
        f"{where}.language_identification.mode must be one of {list(_LANGUAGE_IDENTIFICATION)!r}",
    )
    ceiling = declaration.get("max_candidates")
    if mode != "candidates":
        _require(
            ceiling is None,
            f"{where}.language_identification.max_candidates is meaningless without a candidate list",
        )
        return
    _require(
        bool(entry.get("locales")),
        f"{where}.language_identification is 'candidates' without a locale table to build the list from",
    )
    _require(
        isinstance(ceiling, int) and not isinstance(ceiling, bool) and ceiling >= 2,
        f"{where}.language_identification.max_candidates must be an integer of at least two",
    )
    locales = entry["locales"]
    _require(
        len(set(locales.values())) == len(locales),
        f"{where}.locales maps two languages to one tag, which would spend two candidate slots on one socket",
    )
    simultaneous = entry.get("simultaneous_recognition")
    declared_elsewhere = (
        simultaneous.get("max_candidates") if isinstance(simultaneous, Mapping) else None
    )
    _require(
        declared_elsewhere is None
        or (isinstance(declared_elsewhere, int) and not isinstance(declared_elsewhere, bool)),
        f"{where}.simultaneous_recognition.max_candidates is not a number",
    )
    _require(
        declared_elsewhere is None or declared_elsewhere == ceiling,
        f"{where} declares two different candidate ceilings",
    )


_TOKEN_DIMENSIONS = {
    "stt": frozenset({"input_audio_tokens", "output_text_tokens"}),
    "tts": frozenset({"input_text_tokens", "output_audio_tokens"}),
}

# Raised from 16 to 96 when the call languages grew from 14 to
# 69. It still bounds every per-language projection of the artifact (catalog
# slices, the language picker, the Azure candidate table), and it sits above
# every language one of today's STT providers streams, so it is a ceiling
# against a malformed artifact rather than a product limit.
_MAX_RUNTIME_LANGUAGES = 96


#: TTS billing units the worker measures at its provider wire.
_WORKER_MEASURED_TTS_UNITS = frozenset({"character", "utf8_byte"})
#: TTS billing units only the provider measures, and their reportable dimensions.
_PROVIDER_REPORTED_TTS_UNITS = {"token": ("input_text_tokens", "output_audio_tokens")}
#: How one provider request can end.
_REQUEST_OUTCOMES = ("completed", "abandoned", "transport_failure", "provider_error")


def _validate_request_billing(value: Any, *, where: str, measure: str = "submitted_units") -> None:
    """What each request ending is charged: ``measure`` (what the unit counts) or nothing."""
    rules = _required_object(value, where=where, keys={*_REQUEST_OUTCOMES, "source"})
    _require(set(rules) == {*_REQUEST_OUTCOMES, "source"}, f"{where} is incomplete")
    for outcome in _REQUEST_OUTCOMES:
        _require(rules[outcome] in {measure, "none"}, f"{where}.{outcome} is invalid")
    # A request that ran to completion is always charged what it measured.
    _require(rules["completed"] == measure, f"{where}.completed must be {measure}")
    _require(isinstance(rules["source"], str) and bool(rules["source"].strip()), f"{where}.source is missing")


#: Which workflow source identities settle from the worker's own measured
#: duration and what that duration is. The measure is fixed by role.
_MEASURED_SOURCE_ROLES = {"listener": ("stt", "input_audio_seconds"),
                          "authored": ("tts", "output_audio_seconds"),
                          "live": ("tts", "output_audio_seconds")}


def _validate_workflow_source_measurement(value: Any, capabilities: list[Any]) -> None:
    """Measured-duration settlement rules: data, one entry per source identity.

    A token-billed row converts measured seconds at the provider usage
    reference its own capability row already reserves with, so the rule can
    never carry a second, drifting number.
    """
    _require(isinstance(value, list) and bool(value), "workflow_source_measurement must be a nonempty array")
    identities: set[tuple[str, str, str]] = set()
    for index, entry in enumerate(value):
        where = f"workflow_source_measurement[{index}]"
        rule = _required_object(entry, where=where, keys={
            "role", "component", "provider", "model", "measure", "token_units_per_minute", "source"})
        _require({"role", "component", "provider", "model", "measure", "source"} <= set(rule), f"{where} is incomplete")
        _require(rule["role"] in _MEASURED_SOURCE_ROLES, f"{where}.role is invalid")
        _require((rule["component"], rule["measure"]) == _MEASURED_SOURCE_ROLES[rule["role"]],
                 f"{where}.measure does not match its role")
        for key in ("provider", "model", "source"):
            _require(isinstance(rule[key], str) and bool(rule[key].strip()) and rule[key] == rule[key].strip(),
                     f"{where}.{key} is invalid")
        identity = (rule["role"], rule["provider"], rule["model"])
        _require(identity not in identities, "workflow_source_measurement repeats a source identity")
        identities.add(identity)
        row = next((item for item in capabilities if (item.get("component"), item.get("provider"), item.get("model"))
                    == (rule["component"], rule["provider"], rule["model"])), None)
        token_row = row is not None and isinstance(row.get("billing"), dict) and row["billing"].get("unit") == "token"
        if "token_units_per_minute" not in rule:
            _require(not token_row, f"{where}.token_units_per_minute is required for a token-billed row")
            continue
        per_minute = rule["token_units_per_minute"]
        _require(isinstance(per_minute, dict) and set(per_minute) == _TOKEN_DIMENSIONS[rule["component"]],
                 f"{where}.token_units_per_minute must name exactly the component's token dimensions")
        _require(all(isinstance(amount, int) and not isinstance(amount, bool) and amount > 0
                     for amount in per_minute.values()), f"{where}.token_units_per_minute is invalid")
        _require(token_row, f"{where}.token_units_per_minute belongs to a token-billed capability row")
        reserved = {item["name"]: item["reserve_units_per_minute"] for item in row["billing"]["rate_dimensions"]}
        _require(reserved == per_minute,
                 f"{where}.token_units_per_minute must equal the row's reserve usage references")


def _validate_declared_billing(value: Any, *, where: str) -> None:
    """A pre-registry TTS row's metering rule (its price lives in the catalog).

    Every production TTS adapter the worker builds outside Soniox's own usage
    tracker settles from the native unit meter, so a declared TTS row must say
    how its unit is measured and what each request ending is charged.
    """
    billing = _required_object(value, where=where, keys={"unit", "dimensions", "request_billing"})
    unit = billing.get("unit")
    if unit in _WORKER_MEASURED_TTS_UNITS:
        _require("dimensions" not in billing, f"{where}.dimensions belong to provider-reported units")
        _validate_request_billing(billing.get("request_billing"), where=f"{where}.request_billing")
        return
    _require(unit in _PROVIDER_REPORTED_TTS_UNITS, f"{where}.unit is invalid")
    dimensions = _strings(billing.get("dimensions"), where=f"{where}.dimensions")
    _require(
        all(dimension in _PROVIDER_REPORTED_TTS_UNITS[unit] for dimension in dimensions),
        f"{where}.dimensions are invalid",
    )
    _validate_request_billing(
        billing.get("request_billing"), where=f"{where}.request_billing", measure="reported_usage"
    )


def _validate_billing(
    value: Any, *, where: str, component: str, selectable: bool
) -> None:
    """Validate one exact native billing contract.

    A single token price is not truthful for Soniox: STT bills input audio and
    output text independently, while TTS bills input text and output audio
    independently.  Keep those dimensions in the immutable artifact rather
    than collapsing provider token usage back into a duration estimate.
    """
    billing = _required_object(
        value,
        where=where,
        keys={
            "unit",
            "price_micro_usd_per_unit",
            "reserve_units_per_minute",
            "rate_dimensions",
            "provider_price_basis",
            "request_billing",
        },
    )
    _require(billing.get("unit") in _BILLING_UNITS, f"{where}.unit is invalid")
    # What one provider request is charged once it ends is provider DATA, read
    # by the worker's one native unit meter: required exactly where the worker
    # measures the unit itself (TTS characters / UTF-8 bytes), refused
    # everywhere else so no other meter can appear to honour it.
    worker_measured = component == "tts" and billing["unit"] in _WORKER_MEASURED_TTS_UNITS
    _require(
        ("request_billing" in billing) == worker_measured,
        f"{where}.request_billing is required exactly for worker-measured TTS units",
    )
    if worker_measured:
        _validate_request_billing(billing["request_billing"], where=f"{where}.request_billing")
    is_token = billing["unit"] == "token"
    if not is_token:
        _require(
            {"unit", "price_micro_usd_per_unit", "reserve_units_per_minute"}.issubset(billing),
            f"{where} is incomplete",
        )
        _require("rate_dimensions" not in billing, f"{where}.rate_dimensions is only valid for token billing")
        for key in ("price_micro_usd_per_unit", "reserve_units_per_minute"):
            _require(
                isinstance(billing[key], int)
                and not isinstance(billing[key], bool)
                and billing[key] >= 0,
                f"{where}.{key} is invalid",
            )
        if selectable:
            _require(billing["price_micro_usd_per_unit"] > 0, f"{where} price is missing")
            _require(billing["reserve_units_per_minute"] > 0, f"{where} reserve is missing")
    else:
        _require(
            set(billing).issuperset({"unit", "rate_dimensions"})
            and "price_micro_usd_per_unit" not in billing
            and "reserve_units_per_minute" not in billing,
            f"{where} token dimensions are incomplete",
        )
        dimensions = billing["rate_dimensions"]
        _require(isinstance(dimensions, list), f"{where}.rate_dimensions must be an array")
        expected = _TOKEN_DIMENSIONS[component]
        observed: set[str] = set()
        for index, dimension in enumerate(dimensions):
            item_where = f"{where}.rate_dimensions[{index}]"
            item = _required_object(
                dimension,
                where=item_where,
                keys={"name", "price_micro_usd_per_unit", "reserve_units_per_minute"},
            )
            _require(set(item) == {"name", "price_micro_usd_per_unit", "reserve_units_per_minute"}, f"{item_where} is incomplete")
            name = item.get("name")
            _require(isinstance(name, str) and name in expected, f"{item_where}.name is invalid")
            _require(name not in observed, f"{where}.rate_dimensions contains duplicate names")
            observed.add(name)
            for key in ("price_micro_usd_per_unit", "reserve_units_per_minute"):
                _require(
                    isinstance(item[key], int)
                    and not isinstance(item[key], bool)
                    and item[key] >= 0,
                    f"{item_where}.{key} is invalid",
                )
                if selectable:
                    _require(item[key] > 0, f"{item_where}.{key} is missing")
        _require(observed == expected, f"{where}.rate_dimensions do not match the native {component} token contract")
    if "provider_price_basis" in billing:
        _require(
            isinstance(billing["provider_price_basis"], str) and bool(billing["provider_price_basis"]),
            f"{where}.provider_price_basis is invalid",
        )


def _validate_capability(value: Any, *, index: int) -> None:
    where = f"capabilities[{index}]"
    capability = _required_object(
        value,
        where=where,
        keys={
            "component", "provider", "registry_name", "model", "display", "languages",
            "streaming", "runnable", "selectable", "catalog", "typical_latency_ms",
            "credential", "billing", "voices", "voice_inventory", "locales",
            "language_identification", "simultaneous_recognition",
        },
    )
    required = {
        "component", "provider", "registry_name", "model", "display", "languages",
        "streaming", "runnable", "selectable", "catalog", "typical_latency_ms", "credential", "billing",
    }
    _require(not (required - set(capability)), f"{where} is missing required keys")
    _require(capability["component"] in _COMPONENTS, f"{where}.component is invalid")
    for key in ("provider", "registry_name"):
        _require(isinstance(capability[key], str) and _ID.fullmatch(capability[key]) is not None, f"{where}.{key} is invalid")
    for key in ("model", "display"):
        _require(isinstance(capability[key], str) and bool(capability[key]), f"{where}.{key} is required")
    _strings(capability["languages"], where=f"{where}.languages", pattern=_LANGUAGE)
    for key in ("streaming", "runnable", "selectable", "catalog"):
        _require(isinstance(capability[key], bool), f"{where}.{key} must be boolean")
    _require(not capability["selectable"] or capability["runnable"], f"{where} cannot be selectable without a runtime")
    _require(
        isinstance(capability["typical_latency_ms"], int)
        and not isinstance(capability["typical_latency_ms"], bool)
        and capability["typical_latency_ms"] >= 0,
        f"{where}.typical_latency_ms is invalid",
    )
    credential = _required_object(
        capability["credential"],
        where=f"{where}.credential",
        keys={"managed_env", "byok_supported", "validation"},
    )
    _require(set(credential) == {"managed_env", "byok_supported", "validation"}, f"{where}.credential is incomplete")
    _require(isinstance(credential["managed_env"], str) and _ID.fullmatch(credential["managed_env"].lower()) is not None and credential["managed_env"] == credential["managed_env"].upper(), f"{where}.credential.managed_env is invalid")
    _require(isinstance(credential["byok_supported"], bool), f"{where}.credential.byok_supported must be boolean")
    _require(
        credential["validation"] in _CREDENTIAL_VALIDATION_PROVIDER_UNIT,
        f"{where}.credential.validation is invalid",
    )
    # Native provider evidence must be locked to its immutable billing unit.
    # Otherwise a syntactically valid artifact could make worker admission
    # accept Soniox duration/character estimates or Fish character/token usage
    # even though settlement has only exact token/UTF-8-byte evidence.
    provider_unit = _CREDENTIAL_VALIDATION_PROVIDER_UNIT[str(credential["validation"])]
    _require(
        capability["provider"] == provider_unit[0],
        f"{where}.credential.validation does not match its provider",
    )
    _require(
        isinstance(capability["billing"], dict) and capability["billing"].get("unit") == provider_unit[1],
        f"{where}.billing must use {provider_unit[1]} units for its provider",
    )
    # A row may remain runnable after it is no longer selectable so an
    # already-pinned call stack can keep serving. It must still carry a
    # complete positive native billing contract: the worker's runtime check is
    # intentionally allowed to admit it and settlement must never guess.
    _validate_billing(
        capability["billing"],
        where=f"{where}.billing",
        component=str(capability["component"]),
        selectable=bool(capability["selectable"] or capability["runnable"]),
    )
    _validate_simultaneous_recognition(
        capability, where=where, languages=tuple(str(language) for language in capability["languages"])
    )
    # The same locale contract a declaration carries, on the same validator.
    # It was declaration-only while every capability-backed adapter happened to
    # take bare codes; Azure does not, and the alternative -- a locale map kept
    # in the worker beside the adapter -- is the second hand-typed table this
    # field exists to have deleted. A capability's table is total over its OWN
    # languages, exactly like declarations: neither path may inherit an
    # undeclared language from historical metadata.
    _validate_declared_locales(
        capability,
        where=where,
        languages=tuple(str(language) for language in capability["languages"]),
        allow_tts_locales=True,
    )
    _validate_language_identification(capability, where=where)
    voice_inventory = str(capability.get("voice_inventory") or "static")
    _require(voice_inventory in _VOICE_INVENTORY_MODES, f"{where}.voice_inventory is invalid")
    if voice_inventory == "provider_cache":
        # Exact provider IDs are mutable provider state.  The immutable
        # artifact may authorize the adapter/model/languages, but cannot
        # pretend a checked-in list of dynamic IDs is current.  This is
        # intentionally explicit rather than treating an empty ``voices`` list
        # as dynamic, so a typo cannot turn a static capability permissive.
        _require(capability["component"] == "tts", f"{where}.voice_inventory is only valid for TTS")
    voices = capability.get("voices", [])
    _require(isinstance(voices, list), f"{where}.voices must be an array")
    if voice_inventory == "provider_cache":
        _require(not voices, f"{where}.provider_cache inventory must not contain static voices")
    voice_ids: set[str] = set()
    for voice_index, voice in enumerate(voices):
        voice_where = f"{where}.voices[{voice_index}]"
        voice_data = _required_object(voice, where=voice_where, keys={"id", "name", "languages", "recommended", "metadata"})
        _require({"id", "name", "languages", "metadata"}.issubset(voice_data), f"{voice_where} is incomplete")
        _require(isinstance(voice_data["id"], str) and bool(voice_data["id"]), f"{voice_where}.id is invalid")
        _require(voice_data["id"] not in voice_ids, f"{where}.voices contains duplicate ids")
        voice_ids.add(voice_data["id"])
        _require(isinstance(voice_data["name"], str) and bool(voice_data["name"]), f"{voice_where}.name is invalid")
        languages = _strings(voice_data["languages"], where=f"{voice_where}.languages", pattern=_LANGUAGE)
        _require(set(languages).issubset(set(capability["languages"])), f"{voice_where}.languages are not supported by its capability")
        if "recommended" in voice_data:
            _require(isinstance(voice_data["recommended"], bool), f"{voice_where}.recommended must be boolean")
        _validate_metadata(voice_data["metadata"], where=f"{voice_where}.metadata")


def _validate_tts_voice_inventory(capabilities: list[Any]) -> None:
    """Reject ambiguous or unusable TTS selection data at artifact load time.

    ``default_voice`` and ``default_tts_capability`` retain defensive checks for
    callers holding an already-loaded registry. The artifact itself must not,
    however, boot successfully with a selectable/runnable TTS model that has
    no selectable voice for one of its declared languages, or with ambiguous
    recommendations that only fail later while a draft is being created.
    """
    default_provider_by_language: dict[str, int] = {}
    for index, capability in enumerate(capabilities):
        if capability["component"] != "tts":
            continue
        where = f"capabilities[{index}]"
        voices = capability.get("voices", [])
        provider_cache = capability.get("voice_inventory", "static") == "provider_cache"
        requires_inventory = bool(capability["selectable"] or capability["runnable"]) and not provider_cache
        if requires_inventory:
            _require(
                bool(voices),
                f"{where}.voices must not be empty for a runnable or selectable TTS capability",
            )
        for language in capability["languages"]:
            language_voices = [voice for voice in voices if language in voice["languages"]]
            if requires_inventory:
                _require(
                    bool(language_voices),
                    f"{where}.voices has no approved voice for language {language!r}",
                )
            recommended = [voice for voice in language_voices if voice.get("recommended") is True]
            _require(
                len(recommended) <= 1,
                f"{where}.voices has multiple recommended voices for language {language!r}",
            )
            if capability["catalog"] and capability["selectable"] and recommended:
                previous = default_provider_by_language.get(language)
                _require(
                    previous is None,
                    f"multiple catalog TTS capabilities declare a default voice for language {language!r}",
                )
                default_provider_by_language[language] = index


@dataclass(frozen=True)
class VoiceCapabilityRegistry:
    """One validated, immutable capability revision."""

    data: Mapping[str, Any]
    revision: str

    @property
    def registry_version(self) -> str:
        return str(self.data["registry_version"])

    @property
    def runnable_registry_names(self) -> frozenset[str]:
        """Names of worker adapters that legacy stored stacks may resolve.

        This is intentionally a provider-adapter inventory, not a claim that
        every model under an adapter is available or billable.  Registry-backed
        Soniox/Fish selections must use :meth:`is_capability_runnable`, which
        checks the exact component/provider/model record and its language.
        The extra historical adapter names preserve pinned English, Estonian,
        and Sinhala configurations that predate this registry.
        """
        return frozenset(str(item) for item in self.data["runnable_registry_names"])

    @property
    def capabilities(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.data["capabilities"])

    @property
    def runtime_languages(self) -> tuple[str, ...]:
        """Languages the packaged worker can resolve without another list.

        This covers both the historical, pinned production paths and newly
        catalogued capabilities.  The API uses it to reject a language before
        it can be persisted; the voice image uses the same artifact before it
        constructs a provider client.
        """
        return tuple(self.data["runtime_languages"])

    @property
    def legacy_runtime_languages(self) -> frozenset[str]:
        """Historical artifact metadata, never an admission or locale gate.

        Kept for artifact compatibility. Per-model declarations and capability
        rows are the only language evidence used to admit a stack.
        """
        return frozenset(str(item) for item in self.data["legacy_runtime_languages"])

    @property
    def declared_stack_languages(self) -> tuple[Mapping[str, Any], ...]:
        """Languages each pre-registry adapter/model pair is declared to serve.

        The pre-registry ("legacy") adapters have no per-model capability
        record, so before this table existed the only thing that could be said
        about them was the blanket ``legacy_runtime_languages`` set -- which is
        frozen, and which says nothing about WHICH provider or model carries
        which language.  Every consumer therefore kept its own hand-typed
        answer: the browser demo worker, the demo read model, the phone
        admission evaluator and the web tier each had one, and they drifted.

        This is that answer, once, as data: an explicit statement that this
        exact component/adapter/model is declared to serve these languages.  It
        is deliberately NOT a capability record -- it carries no billing
        contract, no voice inventory and no runnable flag, and it does not make
        its provider registry-backed -- so declaring one cannot silently change
        how an existing stack is metered or admitted.
        """
        return tuple(self.data["declared_stack_languages"])

    def declares_component_model(self, component: str, registry_name: str, model: str) -> bool:
        """Whether this image can construct this exact component/adapter/model.

        True for a registry-backed capability record and for a pre-registry
        adapter/model with an explicit declaration; false for everything else,
        including a real adapter asked for a model nobody wrote down.  The load
        step already refuses a declaration naming an adapter this image cannot
        construct, so a row here is also a statement that the adapter ships.

        Kept separate from :meth:`declares_language` so a refusal can say which
        of the two facts failed: "this worker has no such provider/model" and
        "that model does not speak this language" are different incidents.
        """
        component = str(component or "").strip().lower()
        registry_name = str(registry_name or "").strip().lower()
        model = str(model or "").strip()
        if not (component and registry_name and model):
            return False
        return any(
            item["component"] == component
            and str(item["registry_name"]).lower() == registry_name
            and item["model"] == model
            for item in (*self.capabilities, *self.declared_stack_languages)
        )

    def vendor_provider(self, component: str, registry_name: str, model: str) -> str:
        """The VENDOR behind an adapter name, or the adapter name itself.

        The two are usually the same word and occasionally are not: Gemini TTS
        is adapter ``gemini_tts`` from provider ``gemini``, and the catalog and
        every rate card key on the provider.  A caller that copied the adapter
        name into a provider field would look up a catalog entry that does not
        exist and silently price the component at nothing.
        """
        component = str(component or "").strip().lower()
        registry_name = str(registry_name or "").strip().lower()
        model = str(model or "").strip()
        for item in (*self.capabilities, *self.declared_stack_languages):
            if (
                item["component"] == component
                and str(item["registry_name"]).lower() == registry_name
                and item["model"] == model
            ):
                return str(item["provider"])
        return registry_name

    def is_registry_backed(self, component: str, registry_name: str, model: str) -> bool:
        """Whether this exact component/adapter/model has a capability record.

        These rows carry adapter construction, billing and voice-inventory
        requirements beyond the legacy model declarations.
        """
        component = str(component or "").strip().lower()
        registry_name = str(registry_name or "").strip().lower()
        model = str(model or "").strip()
        return any(
            item["component"] == component
            and str(item["registry_name"]).lower() == registry_name
            and item["model"] == model
            for item in self.capabilities
        )

    def declares_language(
        self, component: str, registry_name: str, model: str, language: str
    ) -> bool:
        """Whether this exact component/adapter/model serves ``language``.

        The one predicate both the separately built voice image and the control
        plane ask, so an agent can never be offered a stack the worker would
        refuse.  A registry-backed capability answers for itself; a
        pre-registry adapter answers from the declaration table above.  An
        adapter/model with neither is refused: "no record" is not evidence that
        something carries a language.

        Deliberately has no ``legacy_runtime_languages`` fallback.  That set is
        provider-blind -- it would let ANY installed adapter claim Estonian --
        and the whole point of asking per model is that ElevenLabs Flash and
        ElevenLabs v3 do not carry the same languages.
        """
        component = str(component or "").strip().lower()
        registry_name = str(registry_name or "").strip().lower()
        model = str(model or "").strip()
        language = str(language or "").strip().lower()
        if not (component and registry_name and model and language):
            return False
        for capability in self.capabilities:
            if (
                capability["component"] == component
                and str(capability["registry_name"]).lower() == registry_name
                and capability["model"] == model
            ):
                return bool(capability["runnable"]) and language in capability["languages"]
        for entry in self.declared_stack_languages:
            if (
                entry["component"] == component
                and str(entry["registry_name"]).lower() == registry_name
                and entry["model"] == model
            ):
                return language in entry["languages"]
        return False

    def simultaneous_stt_config(
        self, registry_name: str, model: str, languages: Any
    ) -> Mapping[str, Any] | None:
        """Return the one evidenced STT construction for a selected set.

        ``language=multi`` is provider syntax, whereas the selected languages
        are the customer's frozen publication contract. The latter may be a
        subset of the provider's documented multi-language set; it must never
        be inferred from the model's larger monolingual language list.

        The one exception is a declaration with ``language_scope: any``: its
        construction names no language, so there is no multi-language set to
        be a subset of, and the returned envelope says ``language_scope``
        instead of carrying a ``declared_languages`` copy.
        """
        registry_name = str(registry_name or "").strip().lower()
        model = str(model or "").strip()
        if not (registry_name and model) or not isinstance(languages, (list, tuple, set, frozenset)):
            return None
        selected = tuple(sorted({str(language or "").strip().lower() for language in languages}))
        if len(selected) < 2 or any(_LANGUAGE.fullmatch(language) is None for language in selected):
            return None
        for entry in (*self.declared_stack_languages, *self.capabilities):
            if (
                entry["component"] != "stt"
                or str(entry["registry_name"]).lower() != registry_name
                or entry["model"] != model
            ):
                continue
            if "runnable" in entry and not bool(entry["runnable"]):
                continue
            raw = entry.get("simultaneous_recognition")
            if not isinstance(raw, Mapping):
                # The artifact has both broad capability rows and the
                # worker's declared-stack rows.  A broad row without an
                # evidenced simultaneous construction must not hide a later
                # matching declaration that has one.
                continue
            if raw.get("language_scope") == LANGUAGE_SCOPE_ANY:
                # An open construction listens for whatever the provider
                # recognises; it sends no language. The selected set must
                # still be languages this model declares, which every caller
                # asks separately too, so no second list is consulted here.
                if not set(selected).issubset({str(language) for language in entry["languages"]}):
                    return None
                return MappingProxyType(
                    {
                        "provider": str(entry["provider"]),
                        "registry_name": str(entry["registry_name"]),
                        "model": str(entry["model"]),
                        "mode": str(raw["mode"]),
                        "languages": selected,
                        "language_scope": LANGUAGE_SCOPE_ANY,
                        "adapter_options": _freeze(_plain(raw["adapter_options"])),
                        "max_candidates": None,
                    }
                )
            declared = tuple(sorted(str(language) for language in raw["languages"]))
            if not set(selected).issubset(set(declared)):
                return None
            max_candidates = raw.get("max_candidates")
            if max_candidates is not None and len(selected) > int(max_candidates):
                return None
            adapter_options = _plain(raw["adapter_options"])
            if raw["mode"] == "strict_language_hints":
                # Strict hints are the provider-facing form of this call's
                # exact frozen set.  The declaration supplies evidence for a
                # larger supported set, but must not leave a caller-selected
                # subset listening for additional languages.
                adapter_options["language_hints"] = list(selected)
            elif raw["mode"] == "continuous_language_identification":
                # The same narrowing, and it matters more here than it does
                # for hints: an identification candidate the customer did not
                # select is a language the provider may ANSWER WITH, and the
                # call has no profile to switch into for it.  The switch would
                # be refused and the caller would keep being answered in a
                # language they stopped speaking, which is the failure this
                # whole path exists to avoid.
                adapter_options["candidate_locales"] = {
                    language: adapter_options["candidate_locales"][language]
                    for language in selected
                }
            return MappingProxyType(
                {
                    "provider": str(entry["provider"]),
                    "registry_name": str(entry["registry_name"]),
                    "model": str(entry["model"]),
                    "mode": str(raw["mode"]),
                    "languages": selected,
                    "declared_languages": declared,
                    "adapter_options": _freeze(adapter_options),
                    "max_candidates": int(max_candidates) if max_candidates is not None else None,
                }
            )
        return None

    def open_listening_config(self, registry_name: str, model: str) -> Mapping[str, Any] | None:
        """The construction that hears the far end in any language, for one STT, or ``None``.

        Only a mode that sends the provider no language can listen open
        (``_OPEN_SCOPE_MODES``: Deepgram ``language=multi``), so the answer is
        read off the row's simultaneous declaration -- its mode and its fixed
        adapter options -- and never off a language list, open or not. A
        transcriber without such a mode (candidate or hint lists, a single
        locale) has no open construction and keeps listening the way it does.
        """
        registry_name = str(registry_name or "").strip().lower()
        model = str(model or "").strip()
        for entry in (*self.declared_stack_languages, *self.capabilities):
            if (
                entry["component"] != "stt"
                or str(entry["registry_name"]).lower() != registry_name
                or entry["model"] != model
                or ("runnable" in entry and not bool(entry["runnable"]))
            ):
                continue
            raw = entry.get("simultaneous_recognition")
            if not isinstance(raw, Mapping) or raw.get("mode") not in _OPEN_SCOPE_MODES:
                continue
            return MappingProxyType(
                {
                    "provider": str(entry["provider"]),
                    "registry_name": str(entry["registry_name"]),
                    "model": str(entry["model"]),
                    "mode": str(raw["mode"]),
                    "adapter_options": _freeze(_plain(raw["adapter_options"])),
                }
            )
        return None

    def stt_language_detection(
        self, registry_name: str, model: str
    ) -> Mapping[str, Any] | None:
        """Return the customer-safe language switching capability for one STT.

        Provider constructor options remain private. The declared language set
        and real candidate ceiling are safe configuration facts used by both
        the API and editor, while ``language_multi`` stays request-only until
        ``runtime_language_switch.PROVIDER_LANGUAGE_EVIDENCE`` routes it to an
        evidence adapter. The pinned Deepgram plugin does report a per-final
        language under ``multi`` (``LanguageMultiEvidence`` reads it); turning
        automatic switching on is a separate, deliberate change. For an open
        scope the languages answered are every language the model declares.
        """
        registry_name = str(registry_name or "").strip().lower()
        model = str(model or "").strip()
        for entry in (*self.declared_stack_languages, *self.capabilities):
            if (
                entry["component"] != "stt"
                or str(entry["registry_name"]).lower() != registry_name
                or entry["model"] != model
                or ("runnable" in entry and not bool(entry["runnable"]))
            ):
                continue
            raw = entry.get("simultaneous_recognition")
            if not isinstance(raw, Mapping):
                continue
            return MappingProxyType(
                {
                    "behavior": (
                        "request_only" if raw["mode"] == "language_multi" else "automatic"
                    ),
                    # An open scope answers with every language this model
                    # declares: the switch set is still chosen from those.
                    "languages": tuple(sorted(str(value) for value in raw.get("languages", entry["languages"]))),
                    "max_candidates": (
                        int(raw["max_candidates"])
                        if raw.get("max_candidates") is not None
                        else None
                    ),
                }
            )
        return None

    def frozen_simultaneous_stt_config(
        self, value: Any, languages: Any
    ) -> Mapping[str, Any] | None:
        """Revalidate a frozen constructor, including released Soniox hints.

        New Soniox publications intentionally omit language restrictions. Old
        strict-hint envelopes remain an executable, immutable call contract;
        widening the current declaration must not strand them or rewrite them.
        """
        if not isinstance(value, Mapping):
            return None
        selected = tuple(sorted({str(item or "").strip().lower() for item in languages}))
        current = self.simultaneous_stt_config(
            str(value.get("registry_name") or ""), str(value.get("model") or ""), selected
        )
        if current is None:
            return None
        constructor_keys = {
            "provider", "registry_name", "model", "mode", "languages", "adapter_options"
        }
        if all(_plain(value.get(key)) == _plain(current.get(key)) for key in constructor_keys):
            # Revalidation proves constructibility; it does not rewrite the
            # publication's audit-only declared inventory or revision.
            return MappingProxyType({str(key): _freeze(_plain(item)) for key, item in value.items()})
        options = value.get("adapter_options")
        if (
            str(value.get("provider") or "").casefold() == "soniox"
            and str(value.get("registry_name") or "").casefold() == "soniox"
            and str(value.get("model") or "") == "stt-rt-v5"
            and value.get("mode") == "strict_language_hints"
            and current.get("mode") == "language_identification"
            and selected == ("en", "ta")
            and list(value.get("languages") or ()) == list(selected)
            and list(value.get("declared_languages") or ()) == ["en", "ta"]
            and isinstance(options, Mapping)
            and set(options) == {
                "language_hints", "language_hints_strict", "enable_language_identification"
            }
            and sorted(str(item) for item in options.get("language_hints") or ()) == list(selected)
            and options.get("language_hints_strict") is True
            and options.get("enable_language_identification") is True
        ):
            return MappingProxyType({str(key): _freeze(_plain(item)) for key, item in value.items()})
        return None

    def locale_for(
        self, component: str, registry_name: str, model: str, language: str
    ) -> str | None:
        """The tag this exact component/adapter/model wants for ``language``.

        The single lookup behind :meth:`stt_locale` and behind synthesis. Both
        questions are "what does this provider call this language", and Azure
        answers them with the same tag -- ``si-LK`` on the recognition socket
        and ``si-LK`` in the SSML ``xml:lang``. Keeping one table means a
        regional choice (``pt-PT`` rather than ``pt-BR``) cannot be made twice
        and differently.
        """
        component = str(component or "").strip().lower()
        registry_name = str(registry_name or "").strip().lower()
        model = str(model or "").strip()
        language = str(language or "").strip().lower()
        if not (component and registry_name and model and language):
            return None
        # Capabilities first, declarations second, and a row found in either is
        # the answer -- the two can never both describe one adapter, which the
        # load step enforces ("declared_stack_languages overlaps a
        # registry-backed adapter"). Searching both here is what lets a
        # capability-backed adapter that refuses bare codes exist at all
        # without a second table kept somewhere in the worker.
        for entry in (*self.capabilities, *self.declared_stack_languages):
            if (
                entry["component"] == component
                and str(entry["registry_name"]).lower() == registry_name
                and entry["model"] == model
            ):
                locales = entry.get("locales")
                if not locales:
                    return None
                locale = locales.get(language)
                return str(locale) if locale else None
        return None

    def stt_candidate_locales(
        self, registry_name: str, model: str, languages: Any
    ) -> tuple[str, ...] | None:
        """The locales one call opens on for an agent's configured languages.

        The whole rule, in one place, for every transcriber: ``None`` unless
        this adapter DECLARES that its API takes a candidate list, the agent
        names more than one language, and the artifact can name a tag for each
        one within whatever ceiling the adapter declares. A caller that gets
        ``None`` opens on the single locale it already had.

        Order is the caller's, not sorted, and the first mention of a
        language wins. That makes the order a stable property of a
        publication -- every call on it opens the same socket -- and it lets
        the caller put the language its call actually speaks in front,
        which is what a worker does. Neither adapter this platform ships
        today reads the first tag as primary: Azure compares the audio
        against every candidate, and Speech V2 takes ``language_codes`` as a
        set and returns the most likely. This ordering is what keeps that
        from being a thing anyone has to keep knowing.

        Deliberately separate from :meth:`simultaneous_stt_config`, which
        answers the frozen follow-the-caller construction and returns nothing
        for an adapter that declares no simultaneous block. This question is
        asked on ordinary calls, whose reply language never moves.
        """
        registry_name = str(registry_name or "").strip().lower()
        model = str(model or "").strip()
        if not (registry_name and model) or not isinstance(languages, (list, tuple)):
            return None
        selected: list[str] = []
        for item in languages:
            language = str(item or "").strip().lower()
            if language and language not in selected:
                selected.append(language)
        if len(selected) < 2:
            return None
        for entry in (*self.capabilities, *self.declared_stack_languages):
            if (
                entry["component"] != "stt"
                or str(entry["registry_name"]).lower() != registry_name
                or entry["model"] != model
                or ("runnable" in entry and not bool(entry["runnable"]))
            ):
                continue
            declaration = entry.get("language_identification")
            if not isinstance(declaration, Mapping) or declaration.get("mode") != "candidates":
                return None
            declared = {str(language) for language in entry["languages"]}
            # A row may transcribe more languages than it was
            # measured identifying between. Where it declares a simultaneous
            # block, only that block's languages are offered as candidates;
            # any other set keeps its one locale, exactly as a set over the
            # candidate ceiling does. (Azure's block is the fourteen measured
            # before Flash-Lite added languages; those were measured one at a
            # time.)
            simultaneous = entry.get("simultaneous_recognition")
            if isinstance(simultaneous, Mapping) and isinstance(simultaneous.get("languages"), (list, tuple)):
                declared &= {str(language) for language in simultaneous["languages"]}
            if not set(selected).issubset(declared):
                return None
            if len(selected) > int(declaration["max_candidates"]):
                return None
            locales = entry.get("locales") or {}
            resolved = tuple(str(locales[language]) for language in selected if locales.get(language))
            if len(resolved) != len(selected):
                return None
            return resolved
        return None

    def stt_locale(self, registry_name: str, model: str, language: str) -> str | None:
        """The tag this exact STT adapter/model wants for ``language``.

        The single answer both dispatch paths ask -- the phone/editor parser in
        ``production.profile_from_realtime_config`` and the homepage parser in
        ``browser_demo.demo_agent_profile``.  Each used to carry its own
        hand-typed map of the few languages whose tag differs from the base
        code, they disagreed (the browser demo had no Sinhala tag at all while
        the phone path did), and neither grew when later changes widened what
        Google is declared to serve.

        ``None`` means "this adapter/model has no locale table entry".  For
        most adapters that is a STATEMENT, not a gap: the bare declared
        language IS the tag to send.  Deepgram states its own tags and is not
        forced to invent regional ones, and a registry-backed capability
        (Soniox) carries no table at all, because Soniox identifies a language
        by its bare code and enforces it with a strict language hint.

        For an adapter in :data:`_LOCALE_REQUIRED_STT_ADAPTERS` it is the
        opposite, and a caller MUST NOT read it as permission to send the bare
        code -- ask :meth:`requires_stt_locale` and refuse instead.  The load
        step refuses a Google STT *declaration* with a missing or incomplete
        table, but it can only police rows that exist: a row renamed or
        deleted is simply not found here, while the worker's own legacy
        adapter/model allowlist goes on admitting ``google``/``chirp_2``
        for ``en``, ``et`` and ``si`` with no
        declaration at all.  Both constructions were built and run in
        adversarial review, and both produced bare codes before this
        distinction existed.
        """
        return self.locale_for("stt", registry_name, model, language)

    def stt_interims_usable(self, registry_name: str, model: str, language: str) -> bool:
        """Whether a word requirement can be enforced on this exact STT stack.

        The single predicate behind ``turn_taking.min_interruption_words``.
        The setting is language-agnostic in the API and in the worker; what is
        NOT agnostic is whether the transcriber puts a countable word in front
        of the barge-in decision, and that is a property of the adapter and
        model rather than of the customer's choice.  So the setting is accepted
        everywhere and made EFFECTIVE only here.

        Fails closed on every shape of "no record", and that is the whole
        point rather than defensiveness.  Component admission on the phone path
        has a provider-blind branch -- the frozen historical ``{en, et, si}``
        set admits a pre-registry adapter/model pair that declares nothing --
        so an undeclared stack reaches this method routinely.  Answering
        anything but ``False`` there would arm a word requirement against a
        transcriber whose cadence nobody has measured, and the failure is
        silent and total: the caller talks over the agent for the whole reply
        because the word that would release the floor never arrives.

        Also requires the row to actually declare the language it is asked
        about.  A row measured on the languages it serves says nothing about
        one it inherited from the historical set, and lending it the answer is
        the same drift ``stt_locale`` documents one method below.
        """
        registry_name = str(registry_name or "").strip().lower()
        model = str(model or "").strip()
        language = str(language or "").strip().lower()
        if not (registry_name and model and language):
            return False
        for entry in self.declared_stack_languages:
            if (
                entry["component"] == "stt"
                and str(entry["registry_name"]).lower() == registry_name
                and entry["model"] == model
            ):
                return bool(entry.get("stt_interims_usable")) and language in entry["languages"]
        return False

    def requires_stt_locale(self, registry_name: str) -> bool:
        """Whether this STT adapter refuses a bare ISO-639-1 code.

        The companion to :meth:`stt_locale`, and the reason that method's
        ``None`` is not self-describing.  A caller that resolves no tag for an
        adapter named here has NOT learned "send the bare code" -- it has
        learned the artifact cannot answer, which for Google means the session
        would connect, bill, and transcribe nothing.  Fail closed.

        Keyed on the adapter name alone, deliberately: it has to answer for an
        adapter/model pair with no declared row at all, which is exactly the
        case :meth:`stt_locale` cannot see.
        """
        return str(registry_name or "").strip().lower() in _LOCALE_REQUIRED_STT_ADAPTERS

    def is_registry_name_runnable(self, registry_name: str) -> bool:
        """Whether a historical adapter name remains constructible.

        Callers must not use this provider-wide fallback for a model represented
        by ``capabilities``; a provider can have an installed adapter while one
        of its models is unavailable, unselectable, or unmetered.
        """
        name = str(registry_name or "")
        return name in self.runnable_registry_names and not any(
            str(capability["registry_name"]) == name for capability in self.capabilities
        )

    def owns_component_provider(self, component: str, provider: str) -> bool:
        """Whether exact model admission is owned for this provider component.

        Historical adapter names are intentionally provider-wide only until a
        component/provider pair is represented here.  Once it is represented,
        every model for that pair must be an explicit artifact decision: a
        legacy adapter must not silently admit a future typo or withdrawn
        model.  Keeping this predicate beside ``capability`` lets the API and
        separately packaged worker enforce the same transition rule.
        """

        normalized_component = str(component or "").strip().lower()
        normalized_provider = str(provider or "").strip().lower()
        return any(
            str(item["component"]).lower() == normalized_component
            and str(item["provider"]).lower() == normalized_provider
            for item in self.capabilities
        )

    def is_capability_runnable(
        self,
        component: str,
        provider: str,
        model: str,
        *,
        language: str | None = None,
    ) -> bool:
        """Return exact-model runnable state from the immutable artifact."""
        capability = self.capability(component, provider, model)
        return bool(
            capability is not None
            and capability["runnable"]
            and (language is None or language in capability["languages"])
        )

    def source_language_scope(self, role: str, provider: str, model: str) -> str | tuple[str, ...] | None:
        """The languages this catalogue declares for one workflow source identity.

        ``role`` is the workflow source role: ``listener`` reads the STT row and
        ``authored`` the TTS row for this exact vendor ``provider``/``model``
        (a capability row that is not runnable declares nothing), ``live``
        reads ``live_sources`` and answers ``LANGUAGE_SCOPE_ANY``. A listed
        scope is the row's own ``languages`` in its order. ``None`` means no
        row: the identity declares no language at all.
        """
        if not all(isinstance(value, str) and value for value in (role, provider, model)):
            return None
        if role == "live":
            # A hand-built historical snapshot predating the section declares no live source.
            for item in self.data.get("live_sources", ()):
                if (item["provider"], item["model"]) == (provider, model):
                    return LANGUAGE_SCOPE_ANY
            return None
        component = {"listener": "stt", "authored": "tts"}.get(role)
        if component is None:
            return None
        for item in self.capabilities:
            if (item["component"], item["provider"], item["model"]) == (component, provider, model):
                return tuple(item["languages"]) if item["runnable"] else ()
        for item in self.declared_stack_languages:
            if (item["component"], item["provider"], item["model"]) == (component, provider, model):
                return tuple(item["languages"])
        return None

    def declares_source_language(self, role: str, provider: str, model: str, language: str) -> bool:
        """Whether a workflow source is declared for ``language``: the admission predicate.

        A listed language, or, for a live source whose row declares
        ``LANGUAGE_SCOPE_ANY``, any of the platform's ``runtime_languages``. Never a quality claim: what
        Skysay has tested is the functional record, read elsewhere.
        """
        # Membership is the whole grammar: a listed scope holds the row's own
        # codes and ``any`` the catalogue's validated ``runtime_languages``, so
        # no second pattern can disagree with what the catalogue declares.
        if not isinstance(language, str) or not language:
            return False
        scope = self.source_language_scope(role, provider, model)
        if scope == LANGUAGE_SCOPE_ANY:
            # Every language the platform runs calls in: "any" opens the live
            # source's own axis, not the call-language domain.
            return language in self.runtime_languages
        return isinstance(scope, tuple) and language in scope

    def capability(self, component: str, provider: str, model: str) -> Mapping[str, Any] | None:
        key = (str(component or ""), str(provider or "").lower(), str(model or ""))
        for capability in self.capabilities:
            if (capability["component"], capability["provider"], capability["model"]) == key:
                return capability
        return None

    def tts_metering(self, registry_name: str, model: str) -> Mapping[str, Any] | None:
        """The TTS row (capability or declared) whose ``billing`` meters this adapter/model.

        One lookup for both kinds of row, so the worker builds every TTS
        meter from data without asking which provider it is.
        """
        registry_name = str(registry_name or "").strip().lower()
        model = str(model or "").strip()
        for item in (*self.capabilities, *self.declared_stack_languages):
            if (
                item["component"] == "tts"
                and str(item["registry_name"]).lower() == registry_name
                and item["model"] == model
            ):
                return item
        return None

    def measured_source_rule(self, role: str, provider: str, model: str) -> Mapping[str, Any] | None:
        """The measured-duration rule for one workflow source identity, or ``None``.

        A source without a rule settles only provider-stated or worker-counted
        exact usage; one with a rule seals the worker's own measurement.
        """
        for rule in self.data["workflow_source_measurement"]:
            if (rule["role"], rule["provider"], rule["model"]) == (role, provider, model):
                return rule
        return None

    def tts_metering_by_vendor(self, provider: str, model: str) -> Mapping[str, Any] | None:
        """``tts_metering`` keyed by the catalog provider instead of the adapter name."""
        provider = str(provider or "").strip().lower()
        model = str(model or "").strip()
        for item in (*self.capabilities, *self.declared_stack_languages):
            if item["component"] == "tts" and str(item["provider"]).lower() == provider and item["model"] == model:
                return item
        return None

    def catalog_capabilities(self, *, component: str | None = None, language: str | None = None) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            capability
            for capability in self.capabilities
            if capability["catalog"]
            and (component is None or capability["component"] == component)
            and (language is None or language in capability["languages"])
        )

    def voices(self, provider: str, model: str, *, language: str | None = None) -> tuple[Mapping[str, Any], ...]:
        capability = self.capability("tts", provider, model)
        if capability is None:
            return ()
        return tuple(
            voice
            for voice in capability.get("voices", ())
            if language is None or language in voice["languages"]
        )

    def uses_provider_voice_inventory(self, provider: str, model: str) -> bool:
        """Whether exact IDs come from the scoped provider cache.

        This says nothing about an arbitrary ID being usable.  It only lets
        callers choose the shared cache/explicit-preview admission path instead
        of trying to validate an ID against the immutable bootstrap inventory.
        """

        capability = self.capability("tts", provider, model)
        return bool(capability is not None and capability.get("voice_inventory") == "provider_cache")

    def _validate_voice(
        self,
        provider: str,
        model: str,
        voice_id: str,
        *,
        language: str,
        require_selectable: bool,
    ) -> Mapping[str, Any]:
        capability = self.capability("tts", provider, model)
        if capability is None or not capability["runnable"]:
            raise VoiceCapabilityError("voice capability is not runnable")
        if require_selectable and not capability["selectable"]:
            raise VoiceCapabilityError("voice capability is not selectable")
        if language not in capability["languages"]:
            raise VoiceCapabilityError("voice capability does not support the requested language")
        if capability.get("voice_inventory") == "provider_cache":
            raise VoiceCapabilityError("voice ids are authorized by the provider inventory")
        for voice in self.voices(provider, model, language=language):
            if voice["id"] == voice_id:
                return voice
        raise VoiceCapabilityError("voice id is absent from the approved inventory")

    def validate_voice(self, provider: str, model: str, voice_id: str, *, language: str) -> Mapping[str, Any]:
        """Validate a fresh picker selection against the approved inventory."""

        return self._validate_voice(
            provider, model, voice_id, language=language, require_selectable=True
        )

    def validate_runnable_voice(
        self, provider: str, model: str, voice_id: str, *, language: str
    ) -> Mapping[str, Any]:
        """Validate an already-pinned voice without reopening the picker.

        A provider can withdraw a model from new selection while continuing to
        serve the exact billed model on existing credentials.  Runtime must
        still require its runnable capability, language, and inventory ID;
        only the fresh-selection flag is intentionally relaxed.
        """

        return self._validate_voice(
            provider, model, voice_id, language=language, require_selectable=False
        )

    def default_voice(self, provider: str, model: str, *, language: str) -> Mapping[str, Any] | None:
        """Return the one explicit recommended inventory item, if any.

        A provider with no recommended voice deliberately returns ``None`` so
        callers never invent a gender, name, locale, or default identifier.
        """
        voices = self.voices(provider, model, language=language)
        recommended = tuple(voice for voice in voices if voice.get("recommended") is True)
        if len(recommended) > 1:
            raise VoiceCapabilityError("voice inventory has multiple recommended voices for one language")
        return recommended[0] if recommended else None

    def default_tts_capability(self, *, language: str) -> Mapping[str, Any] | None:
        """Return the one catalogued TTS capability with a declared default voice.

        A recommendation belongs to the provider's inventory, not to a
        hand-maintained UI ordering rule.  Selecting the enclosing capability
        from the same immutable artifact keeps a fresh draft and a server-side
        default aligned.  More than one declared provider default for a
        language is ambiguous, so reject the artifact rather than selecting by
        incidental JSON order.
        """
        candidates = tuple(
            capability
            for capability in self.capabilities
            if capability["component"] == "tts"
            and capability["catalog"]
            and capability["selectable"]
            and language in capability["languages"]
            and self.default_voice(
                str(capability["provider"]), str(capability["model"]), language=language
            ) is not None
        )
        if len(candidates) > 1:
            raise VoiceCapabilityError("multiple TTS capabilities declare a default voice for one language")
        return candidates[0] if candidates else None

    def validate_registry_component_billing(
        self,
        config: Mapping[str, Any],
        *,
        stt_capability: Mapping[str, Any] | None,
        tts_capability: Mapping[str, Any] | None,
    ) -> None:
        """Require a complete pinned billing identity for new provider stacks.

        Registry-backed adapters are introduced with native usage contracts, so
        they may not take the historical unmetered worker path when a corrupt
        or forged per-call config reaches the voice host. Existing EN/ET/SI
        profiles remain compatible because neither selected component resolves
        through this new registry.
        """
        if stt_capability is None and tts_capability is None:
            return
        billing = config.get("component_billing")
        if not isinstance(billing, Mapping) or billing.get("usage_required") is not True:
            raise VoiceCapabilityError("registry-backed components require component-metered billing")
        components = billing.get("components")
        if not isinstance(components, list):
            raise VoiceCapabilityError("component billing configuration is incomplete")

        def selection_identity(
            name: str, *, capability: Mapping[str, Any] | None = None
        ) -> tuple[str, str]:
            selected = config.get(name)
            if not isinstance(selected, Mapping):
                return "", ""
            selected_provider = str(
                selected.get("provider") or selected.get("registry_name") or ""
            ).strip().lower()
            selected_model = str(selected.get("model") or "").strip()
            if capability is None:
                return selected_provider, selected_model

            # This method is deliberately a last line of defence for a
            # serialized realtime config.  Its production caller resolved the
            # capability from this same selection already, but do not let a
            # future caller accidentally pair a registry rate card with an
            # arbitrary model merely because its billing components agree with
            # one another.  ``provider`` may be omitted in favour of the
            # durable registry name, so accept either exact normalized value.
            capability_provider = str(capability.get("provider") or "").strip().lower()
            capability_registry = str(capability.get("registry_name") or "").strip().lower()
            capability_model = str(capability.get("model") or "").strip()
            if (
                selected_provider not in {capability_provider, capability_registry}
                or selected_model != capability_model
            ):
                raise VoiceCapabilityError("selected component does not match the capability billing identity")
            # The selector may identify an adapter by its durable registry
            # name, but component billing always pins the vendor/provider
            # identity that owns the native meter and rate card.
            return capability_provider, capability_model

        expected = {
            "stt": selection_identity("transcriber", capability=stt_capability),
            "llm": selection_identity("model"),
            "tts": selection_identity("voice", capability=tts_capability),
        }
        observed: dict[str, tuple[str, str]] = {}
        for component in components:
            if not isinstance(component, Mapping):
                raise VoiceCapabilityError("component billing configuration is incomplete")
            name = str(component.get("component") or "")
            if name not in expected or name in observed:
                raise VoiceCapabilityError("component billing configuration does not match the selected stack")
            observed[name] = (
                str(component.get("provider") or "").strip().lower(),
                str(component.get("model") or "").strip(),
            )
        if observed != expected or any(not all(identity) for identity in expected.values()):
            raise VoiceCapabilityError("component billing configuration does not match the selected stack")

    def managed_rate_contracts(self) -> dict[tuple[str, str, str], dict[str, Any]]:
        result: dict[tuple[str, str, str], dict[str, Any]] = {}
        for capability in self.capabilities:
            # A model may be removed from the picker while retained for a
            # previously pinned stack. ``_validate_capability`` requires the
            # same complete native billing contract for either runnable or
            # selectable rows, so settlement must export both rather than
            # turn an already-admitted model into an unmetered hole.
            if not (capability["selectable"] or capability["runnable"]):
                continue
            billing = capability["billing"]
            contract: dict[str, Any] = {"unit": billing["unit"]}
            if billing["unit"] == "token":
                for dimension in billing["rate_dimensions"]:
                    name = str(dimension["name"])
                    singular = name[:-1] if name.endswith("s") else name
                    contract[f"price_micro_usd_per_{singular}"] = int(
                        dimension["price_micro_usd_per_unit"]
                    )
                    contract[f"reserve_{name}_per_minute"] = int(
                        dimension["reserve_units_per_minute"]
                    )
            else:
                contract.update(
                    {
                        "price_micro_usd_per_unit": int(billing["price_micro_usd_per_unit"]),
                        "reserve_units_per_minute": int(billing["reserve_units_per_minute"]),
                    }
                )
            result[(capability["component"], capability["provider"], capability["model"])] = contract
        return result

    def byok_contracts(self) -> frozenset[tuple[str, str]]:
        return frozenset(
            (capability["component"], capability["provider"])
            for capability in self.capabilities
            if capability["credential"]["byok_supported"]
        )


def _load(path: Path) -> VoiceCapabilityRegistry:
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VoiceCapabilityError("voice capability source cannot be loaded") from exc
    _require(isinstance(data, dict), "voice capability source must be an object")
    expected_top_level = {
        "schema_version",
        "registry_version",
        "runtime_languages",
        "legacy_runtime_languages",
        "runnable_registry_names",
        "declared_stack_languages",
        "live_sources",
        "workflow_source_measurement",
        "capabilities",
    }
    _require(set(data) == expected_top_level, "voice capability source has unsupported or missing top-level keys")
    _require(
        isinstance(data["schema_version"], int)
        and not isinstance(data["schema_version"], bool)
        and data["schema_version"] == 1,
        "voice capability schema version is unsupported",
    )
    _require(isinstance(data["registry_version"], str) and _VERSION.fullmatch(data["registry_version"]) is not None, "voice capability registry version is invalid")
    runtime_languages = _strings(data["runtime_languages"], where="runtime_languages", pattern=_LANGUAGE)
    _require(
        len(runtime_languages) <= _MAX_RUNTIME_LANGUAGES,
        "runtime_languages exceeds the bounded catalog response contract",
    )
    legacy_runtime_languages = _strings(
        data["legacy_runtime_languages"],
        where="legacy_runtime_languages",
        pattern=_LANGUAGE,
        nonempty=False,
    )
    _require(
        set(legacy_runtime_languages).issubset(set(runtime_languages)),
        "legacy_runtime_languages references a language absent from the worker runtime contract",
    )
    runnable_registry_names = set(
        _strings(data["runnable_registry_names"], where="runnable_registry_names", pattern=_ID)
    )
    _require(
        isinstance(data["declared_stack_languages"], list),
        "declared_stack_languages must be an array",
    )
    declared_keys: set[tuple[str, str, str]] = set()
    declared_registry_names: set[str] = set()
    for index, entry in enumerate(data["declared_stack_languages"]):
        _validate_declared_stack_languages(entry, index=index)
        key = (entry["component"], entry["registry_name"], entry["model"])
        _require(
            key not in declared_keys,
            "declared_stack_languages contains duplicate component/adapter/model identities",
        )
        declared_keys.add(key)
        declared_registry_names.add(str(entry["registry_name"]))
        _require(
            str(entry["registry_name"]) in runnable_registry_names,
            "declared_stack_languages names an adapter this image cannot construct",
        )
        _require(
            set(entry["languages"]).issubset(set(runtime_languages)),
            "declared_stack_languages references a language absent from the worker runtime contract",
        )
    _require(isinstance(data["capabilities"], list) and bool(data["capabilities"]), "capabilities must be a nonempty array")
    keys: set[tuple[str, str, str]] = set()
    capability_registry_names: set[str] = set()
    byok_by_component_provider: dict[tuple[str, str], bool] = {}
    for index, capability in enumerate(data["capabilities"]):
        _validate_capability(capability, index=index)
        key = (capability["component"], capability["provider"], capability["model"])
        _require(key not in keys, "capabilities contains duplicate component/provider/model identities")
        keys.add(key)
        capability_registry_names.add(str(capability["registry_name"]))
        component_provider = (capability["component"], capability["provider"])
        byok_supported = bool(capability["credential"]["byok_supported"])
        existing_byok = byok_by_component_provider.setdefault(component_provider, byok_supported)
        _require(
            existing_byok == byok_supported,
            "capabilities disagree about BYOK support for one component/provider",
        )
        _require(
            set(capability["languages"]).issubset(set(runtime_languages)),
            "capability references a language absent from the worker runtime contract",
        )
    _require(isinstance(data["live_sources"], list), "live_sources must be an array")
    live_keys = [_validate_live_source(item, index=index) for index, item in enumerate(data["live_sources"])]
    _require(len(set(live_keys)) == len(live_keys), "live_sources contains duplicate provider/model identities")
    # Every runtime language needs named speech components. Historical metadata
    # grants no exemption from the per-model contract.
    #
    # This is THE both-sides rule (a founder decision): a language is a call
    # language only when a voice speaks it, a transcriber hears callers in it
    # and a language model is declared for it. It is enforced here, once, at
    # load, so every reader of ``runtime_languages`` -- the agent catalogue,
    # the voice library's call languages, the publish preflight and call
    # admission in both images -- inherits it instead of restating it. An
    # artifact that names a language without all three does not load.
    for language in runtime_languages:
        named_components = {
            str(capability["component"])
            for capability in data["capabilities"]
            if capability["runnable"] and language in capability["languages"]
        } | {
            str(entry["component"])
            for entry in data["declared_stack_languages"]
            if language in entry["languages"]
        }
        _require(
            {"stt", "tts"}.issubset(named_components),
            "runtime language lacks a named STT/TTS provider",
        )
        _require(
            "llm" in named_components,
            "runtime language lacks a declared language model",
        )
    _require(
        not (capability_registry_names & runnable_registry_names),
        "registry-backed capability overlaps the historical adapter allowlist",
    )
    # A declaration is for a pre-registry adapter only.  Once a component /
    # provider pair is registry-backed, every one of its models is an explicit
    # artifact decision with a billing contract behind it, and a second, weaker
    # statement about the same adapter would be a way to route around that.
    _require(
        not (declared_registry_names & capability_registry_names),
        "declared_stack_languages overlaps a registry-backed adapter",
    )
    _validate_tts_voice_inventory(data["capabilities"])
    _validate_workflow_source_measurement(data["workflow_source_measurement"], data["capabilities"])
    # Schema is intentionally versioned alongside the data.  Refusing a missing
    # schema catches an image that accidentally copied only one half of the
    # source artifact even though this stdlib reader enforces the same shape.
    schema = path.with_name("schema.json")
    try:
        schema_data = json.loads(schema.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VoiceCapabilityError("voice capability schema cannot be loaded") from exc
    _require(isinstance(schema_data, dict), "voice capability schema must be an object")
    _require(schema_data.get("$schema") == "https://json-schema.org/draft/2020-12/schema", "voice capability schema draft is unsupported")
    schema_properties = schema_data.get("properties")
    _require(isinstance(schema_properties, dict), "voice capability schema properties are invalid")
    schema_version_rule = schema_properties.get("schema_version")
    _require(isinstance(schema_version_rule, dict), "voice capability schema version contract is invalid")
    schema_version_const = schema_version_rule.get("const")
    _require(
        isinstance(schema_version_const, int)
        and not isinstance(schema_version_const, bool)
        and schema_version_const == 1,
        "voice capability schema version contract is invalid",
    )
    _require(
        schema_version_rule.get("type") == "integer",
        "voice capability schema version contract is invalid",
    )
    schema_required = schema_data.get("required")
    _require(isinstance(schema_required, list) and set(schema_required) == expected_top_level, "voice capability schema top-level contract drifted")
    return VoiceCapabilityRegistry(data=_freeze(data), revision=hashlib.sha256(raw).hexdigest())


@lru_cache(maxsize=1)
def load_registry() -> VoiceCapabilityRegistry:
    """Load the one artifact once per process and return an immutable revision."""
    return _load(_root() / "capabilities.json")


def public_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy without exposing mutable registry state."""
    return _plain(value)


__all__ = [
    "VoiceCapabilityError",
    "VoiceCapabilityRegistry",
    "load_registry",
    "public_dict",
]
