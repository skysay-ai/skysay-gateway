# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Root-owned native language evidence over the released stable policy.

This boundary schedules a target only. The Root's serial controller owns
business HTTP, exact published replay, profile generations and source prices.
"""
from __future__ import annotations

import copy
import re
from types import MappingProxyType, SimpleNamespace

try:
    from .runtime_language_switch import FollowCallerProfiles, LanguageSwitchError, PROVIDER_LANGUAGE_EVIDENCE
    from .workflow_input_recognition import CallerLanguageHint
    from .capabilities.workflow_sources import MAX_CALL_ATTEMPTS
except ImportError:
    from runtime_language_switch import FollowCallerProfiles, LanguageSwitchError, PROVIDER_LANGUAGE_EVIDENCE
    from workflow_input_recognition import CallerLanguageHint
    from capabilities.workflow_sources import MAX_CALL_ATTEMPTS


class RootOutputLanguage:
    """Synchronous state used only at the Root's serial output commit point.

    Scheduling buys no source and advances no workflow. One authenticated
    request retains its first language, including every later presentation.
    """

    def __init__(self, *, languages, opening_language, max_requests=MAX_CALL_ATTEMPTS):
        if (type(languages) is not tuple or not 2 <= len(languages) <= 32
                or any(type(language) is not str or not 1 <= len(language) <= 64
                       or language != language.strip().lower() for language in languages)
                or len(set(languages)) != len(languages) or opening_language not in languages
                or type(max_requests) is not int or not 1 <= max_requests <= MAX_CALL_ATTEMPTS):
            raise LanguageSwitchError("Root output requires bounded frozen languages and request ownership")
        self._languages = frozenset(languages)
        self._active = opening_language
        self._generation = 0
        self._pending = None
        self._terminal = False
        self._requests, self._limit = {}, max_requests

    @property
    def active_language(self):
        return self._active

    @property
    def generation(self):
        return self._generation

    def transition_pending(self, language):
        return type(language) is str and self._pending == language

    def schedule(self, language):
        if self._terminal or type(language) is not str or language not in self._languages:
            return "rejected"
        if language == self._active:
            return "already_active"
        self._pending = language
        return "scheduled"

    def mark_terminal(self):
        self._terminal = True
        self._pending = None

    def clear_pending(self):
        self._pending = None

    def choose(self, request_id, *, requested_language=None, terminal=False):
        if (type(request_id) is not str or re.fullmatch(r"[A-Za-z0-9_-]{16,96}", request_id) is None
                or type(terminal) is not bool):
            raise LanguageSwitchError("Root output requires its authenticated reply identity and terminal state")
        if terminal:
            self.mark_terminal()
        if request_id in self._requests:
            return self._requests[request_id]
        if len(self._requests) >= self._limit:
            raise LanguageSwitchError("Root output exceeded its bounded request history")
        language = self._active
        if not self._terminal:
            if type(requested_language) is str and requested_language in self._languages:
                language = requested_language
            elif self._pending is not None:
                language = self._pending
        if language != self._active:
            self._active = language
            self._generation += 1
        self._pending = None
        self._requests[request_id] = language
        return language


class RootLanguageEvidence:
    """Translate one typed capture hint; never inspect caller words or audio.

    The existing evidence adapters compose the shared _StableLanguageLatch.
    Its active/pending/schedule ports belong to the Root, rather than to a
    freeform AgentSession or a second language-switching actor.
    """

    def __init__(self, frozen: FollowCallerProfiles, *, active_language, transition_pending, schedule):
        if (not isinstance(frozen, FollowCallerProfiles)
                or not 1 <= len(frozen.profiles) <= 32
                or any(type(language) is not str for language in frozen.profiles)
                or any(not callable(port) for port in (active_language, transition_pending, schedule))):
            raise LanguageSwitchError("root language evidence requires frozen profiles and owned ports")
        self._frozen = SimpleNamespace(
            profiles=MappingProxyType(dict.fromkeys(frozen.profiles)),
            simultaneous_stt=copy.deepcopy(frozen.simultaneous_stt))
        self._active_language = active_language
        self._transition_pending = transition_pending
        self._schedule = schedule
        mode = self._frozen.simultaneous_stt.get("mode")
        factory = PROVIDER_LANGUAGE_EVIDENCE.get(mode)
        self._evidence = factory(self) if factory is not None else None
        self._locale_mode = mode == "continuous_language_identification"

    @property
    def active_language(self):
        language = self._active_language()
        if type(language) is not str or language not in self._frozen.profiles:
            raise LanguageSwitchError("root language generation has no frozen active profile")
        return language

    def transition_pending(self, language):
        pending = self._transition_pending(language)
        if type(pending) is not bool:
            raise LanguageSwitchError("root language generation has no typed pending state")
        return pending

    def provider_stable_evidence(self, *, language):
        if language not in self._frozen.profiles:
            raise LanguageSwitchError("root native evidence has no frozen target profile")
        return self._schedule(language)

    def observe(self, hint: CallerLanguageHint | None):
        if self._evidence is None:
            return None
        typed = isinstance(hint, CallerLanguageHint)
        if self._locale_mode:
            locale = (hint.locale if typed and hint.source_languages is None
                      and (hint.locale is None or type(hint.locale) is str) else None)
            return self._evidence.observe_detected_locale(locale)
        sources = (hint.source_languages if typed and hint.locale is None
                   and type(hint.source_languages) is tuple
                   and all(type(language) is str for language in hint.source_languages) else None)
        return self._evidence.observe_source_languages(sources)
