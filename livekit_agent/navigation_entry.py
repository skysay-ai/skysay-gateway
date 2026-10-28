# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""The optional call feature this build does not include.

Generated into the public export as ``livekit_agent/navigation_entry.py`` (template:
``scripts/oss-release/templates/navigation_entry.py``). The worker reaches the feature through
this one module. Here it is a stand-in: it always reports the feature off, builds nothing,
and redacts nothing. That is safe because a call in this build never carries a held value to
redact: nothing here can ask a caller for one, and a self-hosted worker refuses the keys that
would deliver one.

The names are exactly those of the full build's ``navigation_entry.py``; the private suite pins
the two lists equal, and pins every inert answer below to the full build's answer for a call
that has the feature off. A name that only a call with the feature on can reach is a loud
``NavigationUnavailable``, never a quiet no-op.
"""

from __future__ import annotations

import contextlib
import enum
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence


class NavigationUnavailable(RuntimeError):
    """Raised by anything only a call with the feature on could reach."""


def _unavailable(name: str) -> Callable[..., Any]:
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise NavigationUnavailable(f"{name} is not available in this build")

    refuse.__name__ = name
    return refuse


def _unavailable_class(name: str) -> type:
    def refuse(self: Any, *_args: Any, **_kwargs: Any) -> None:
        raise NavigationUnavailable(f"{name} is not available in this build")

    return type(name, (), {"__init__": refuse, "__doc__": "Not available in this build."})


# -- vocabulary the worker names -------------------------------------------------------------------
DIGIT_GAP_SECONDS = 0.12
GROUP_GAP_SECONDS = 0.45
DETAIL_DATA_NOT_AUTHORISED = "data_not_authorised"
DETAIL_DURING_HOLD = "during_hold"
DETAIL_MENU_NO_PATH = "menu_no_path"
DETAIL_NAVIGATION_DETECTOR_LOST = "navigation_detector_lost"
DETAIL_RECORDED_MESSAGE_ONLY = "recorded_message_only"
ENDER_ARBITER = "arbiter_endings"
ENDER_CLOSING_TEXT_FALLBACK = "closing_text_fallback"
ENDER_END_CALL_TOOL = "end_call_tool"
ENDER_MODEL_UNRESPONSIVE = "model_unresponsive"
ENDER_MUTUAL_SILENCE = "mutual_silence"
ENDER_NO_CALLER_SPEECH = "no_caller_speech"
ENGINE_CASCADE = "cascade"
ENGINE_LIVE = "live"
EVENT_INTERVALS = "navigation_intervals"
LIMIT_DEADLINE = "deadline"
LIMIT_HOLD = "hold_limit"
LIMIT_NAVIGATION = "navigation_limit"
SPEECH_ACKNOWLEDGEMENT = "acknowledgement"
SPEECH_CHECK_IN = "check_in"
SPEECH_LANGUAGE_SWITCH = "language_switch"
SPEECH_RECOVERY = "recovery_speech"
STATE_SETTLING = "settling"
WITHHELD_TEXT = "far end spoke, content withheld"

_STATE_CONVERSATION = "conversation"
_ENDING_MAX_CALL = "max_call_duration_reached"


# -- the settings resolver: always off ---------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class NavigationSettings:
    enabled: bool = False


def navigation_settings(realtime_config: Any = None) -> NavigationSettings:
    return NavigationSettings()


def navigation_engine_refused(realtime_config: Any = None) -> bool:
    return False


class NavigationPolicy:
    """A policy that is never enabled and never suspends anything."""

    __slots__ = ()

    @classmethod
    def from_settings(cls, settings: Any = None, **_options: Any) -> "NavigationPolicy":
        return cls()

    enabled = False
    in_episode = False
    state = _STATE_CONVERSATION

    def suspends(self, name: str) -> bool:
        return False

    def duration_ending(self) -> tuple[str, str]:
        return _ENDING_MAX_CALL, ""

    def remote_hangup_detail(self) -> str:
        return ""

    def set_deadline(self, at: float | None) -> None:
        return None

    def bind_far_end_audio(self, source: Callable[[], bool] | None) -> None:
        return None

    def bind_conversation_activity(self, source: Callable[[], bool], *, pending: Callable[[], bool] | None = None) -> None:
        return None

    def open_entry_window(self, why: str = "answer") -> bool:
        return False

    def add_episode_listener(self, listener: Callable[[int], None]) -> None:
        return None

    def person_reached_state(self, *, unresolved: bool = False) -> bool | None:
        return None

    def recorded_only_due(self, *, human_seen: bool = False, unresolved: bool = False) -> bool:
        return False


def follow_far_end(navigation: Any, *, observer: Any = None, arbiter: Any = None, music: Any = None) -> str:
    return ""


def leave_episode(navigation: Any, detector: Any) -> None:
    return None


def keypad_fields(config: Any) -> list:
    """No navigation, so no keypad value: a call here never holds one."""
    return []


def observer_saw_a_person(observer: Any) -> bool:
    return False


# -- the tool gate: nothing is ever held or refused ----------------------------------------------------
class NavigationToolGate:
    def __init__(self, **_options: Any) -> None:
        pass

    def applies(self, tool_name: str) -> bool:
        return False

    def unresolved(self) -> bool:
        return False

    async def hold(self, tool_name: str) -> bool:
        return True

    async def refusal(self, tool_name: str) -> dict[str, Any] | None:
        return None

    async def opening_failure_excused(self, site: str, **_options: Any) -> bool:
        return False


async def line_is_a_menu(*_args: Any, **_kwargs: Any) -> bool:
    return False


async def line_is_holding_music(*_args: Any, **_kwargs: Any) -> bool:
    return False


# -- redaction: a call here holds no value, so nothing is masked -----------------------------------------
def scrub_registered_structure(value: Any) -> Any:
    return value


class Consumer(str, enum.Enum):
    TRANSCRIPT_ARTIFACT = "transcript_artifact"


@dataclass(frozen=True)
class FarEnd:
    text: str
    language: str = ""
    final: bool = True


@dataclass(frozen=True)
class Own:
    text: str
    language: str = ""


class Redactor:
    """A redactor that is never in value mode and passes every text through."""

    def __init__(self, call_id: str = "", *, default_language: Any = "", call_languages: Sequence[Any] = ()) -> None:
        pass

    @property
    def value_mode(self) -> bool:
        return False

    def has_numeral(self, text: str, language: Any = "") -> bool:
        return False

    def set_default_language(self, language: Any) -> None:
        return None

    def effective_language(self, tag: Any = "") -> Any:
        return tag

    def deliver(self, consumer: Any, event: Any) -> str | None:
        return getattr(event, "text", event)

    def remask_ranges(self, texts: Sequence[str]) -> list[list[tuple[int, int]]]:
        return [[] for _ in texts]

    @staticmethod
    def replace(text: str, ranges: Sequence[tuple[int, int]]) -> str:
        return text

    def close(self) -> None:
        return None


class NavigationTimeline:
    def __init__(self, recorder: Any = None, **_options: Any) -> None:
        pass

    def on_policy_event(self, event: str, **details: Any) -> None:
        return None


# -- building: nothing is built -----------------------------------------------------------------------------
def build_navigation_actions(**_options: Any) -> None:
    return None


def build_perception(**_options: Any) -> None:
    return None


def cascade_model_feed(*_args: Any, **_kwargs: Any) -> None:
    return None


async def start_controller_track(room: Any) -> None:
    return None


async def flush_controller_track(*_args: Any, **_kwargs: Any) -> None:
    return None


def build_hold_safe_gemini(**_options: Any) -> Any:
    raise ImportError("the hold-safe Gemini model is not available in this build")


# -- helpers the worker calls on every call ------------------------------------------------------------------
def fed_prompt(chat_ctx: Any) -> int | None:
    return None


def with_a_user_turn(chat_ctx: Any, make_message: Callable[[str], Any] | None = None) -> Any:
    return chat_ctx


@contextlib.contextmanager
def greeting_site() -> Iterator[None]:
    yield


def without_preemptive_generation(session_options: Any) -> dict[str, Any]:
    return dict(session_options)


def navigation_base_language(tag: Any) -> str:
    code = str(tag or "").strip().lower().replace("_", "-").split("-", 1)[0]
    return "" if code in ("und", "zxx", "mis", "mul") else code


# -- the capability facts: this build trusts no source for navigation and releases no spoken value ------------
def navigation_untrusted_sources(registry: Any, sources: Any, languages: Any) -> tuple[str, ...]:
    return ()


def navigation_plan_sources(plan: Any) -> tuple[tuple[str, str, str], ...]:
    return ()


def navigation_menu_language_trusted(*_args: Any, **_kwargs: Any) -> bool:
    return False


def voice_says_held_value(registry: Any, registry_name: Any, model: Any, language: Any) -> bool:
    return False


# -- live-engine input backends: held by the worker, installed only by a call with the feature on ----------
class SilencedInput:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def install(self, session: Any) -> None:
        return None


class GptLiveInput:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass


# -- reachable only with the feature on --------------------------------------------------------------------------
ProbeLine = _unavailable_class("ProbeLine")
ProbeSpeech = _unavailable_class("ProbeSpeech")
FarEndWatch = _unavailable_class("FarEndWatch")
NavigationActions = _unavailable_class("NavigationActions")
NavDecider = _unavailable_class("NavDecider")
WorkflowFinalGate = _unavailable_class("WorkflowFinalGate")
WorkflowNavigator = _unavailable_class("WorkflowNavigator")
WorkflowSettling = _unavailable_class("WorkflowSettling")
build_far_end_classifier = _unavailable("build_far_end_classifier")
cascade_value_covers = _unavailable("cascade_value_covers")
cascade_voice = _unavailable("cascade_voice")
hand_native_opening_to_navigation = _unavailable("hand_native_opening_to_navigation")
listeners_of = _unavailable("listeners_of")
navigation_intervals = _unavailable("navigation_intervals")
sip_identities_of = _unavailable("sip_identities_of")
yield_to_far_end = _unavailable("yield_to_far_end")

__all__ = [
    "DIGIT_GAP_SECONDS",
    "GROUP_GAP_SECONDS",
    "ProbeLine",
    "ProbeSpeech",
    "DETAIL_DATA_NOT_AUTHORISED",
    "DETAIL_DURING_HOLD",
    "DETAIL_MENU_NO_PATH",
    "DETAIL_NAVIGATION_DETECTOR_LOST",
    "DETAIL_RECORDED_MESSAGE_ONLY",
    "ENDER_ARBITER",
    "ENDER_CLOSING_TEXT_FALLBACK",
    "ENDER_END_CALL_TOOL",
    "ENDER_MODEL_UNRESPONSIVE",
    "ENDER_MUTUAL_SILENCE",
    "ENDER_NO_CALLER_SPEECH",
    "ENGINE_CASCADE",
    "ENGINE_LIVE",
    "EVENT_INTERVALS",
    "LIMIT_DEADLINE",
    "LIMIT_HOLD",
    "LIMIT_NAVIGATION",
    "SPEECH_ACKNOWLEDGEMENT",
    "SPEECH_CHECK_IN",
    "SPEECH_LANGUAGE_SWITCH",
    "SPEECH_RECOVERY",
    "STATE_SETTLING",
    "WITHHELD_TEXT",
    "Consumer",
    "FarEnd",
    "FarEndWatch",
    "GptLiveInput",
    "NavDecider",
    "NavigationActions",
    "NavigationPolicy",
    "NavigationSettings",
    "NavigationTimeline",
    "NavigationToolGate",
    "Own",
    "Redactor",
    "SilencedInput",
    "WorkflowFinalGate",
    "WorkflowNavigator",
    "WorkflowSettling",
    "build_far_end_classifier",
    "build_hold_safe_gemini",
    "build_navigation_actions",
    "build_perception",
    "cascade_model_feed",
    "cascade_value_covers",
    "cascade_voice",
    "fed_prompt",
    "flush_controller_track",
    "follow_far_end",
    "greeting_site",
    "hand_native_opening_to_navigation",
    "keypad_fields",
    "leave_episode",
    "line_is_a_menu",
    "line_is_holding_music",
    "listeners_of",
    "navigation_base_language",
    "navigation_engine_refused",
    "navigation_intervals",
    "navigation_menu_language_trusted",
    "navigation_plan_sources",
    "navigation_settings",
    "navigation_untrusted_sources",
    "observer_saw_a_person",
    "scrub_registered_structure",
    "sip_identities_of",
    "start_controller_track",
    "voice_says_held_value",
    "with_a_user_turn",
    "without_preemptive_generation",
    "yield_to_far_end",
]
