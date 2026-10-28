# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Liveness witnesses on a duplex (voice-to-voice) provider session.

When a GPT-Live agent goes silent while the caller keeps talking, two causes
look identical from everywhere the worker already watches: the provider
stopped answering, or our own send of the caller's audio stalled. Neither
reports an error. These witnesses sit where the two separate:

* **input, after the runtime and at the socket.** ``push_audio`` is wrapped
  on the plugin's session, which the runtime calls with its silence
  substitution already applied, so a stretch of substituted silence reads as
  silence here even while the caller is talking. Each audio append is then
  seen again when the plugin hands it to the websocket
  (``openai_client_event_queued`` fires inside the send, immediately before
  the write), so appends queued but never handed over, and how long the
  oldest has waited, show a send that stalled.
* **output, from the provider.** Every server event is counted by TYPE only,
  from ``openai_server_event_received``. The raw events carry transcript
  text; nothing here keeps, logs or forwards anything but the type string.

So the next silent-agent incident answers itself: appends still reaching the
socket with nothing coming back is the provider; appends piling up unsent is
our send; silence handed over while the caller talks is the substitution.

**Output that is speech, not the stream.** A GPT-Live session streams output
audio frames for the whole call, speaking or not, so "output still arriving"
says only that the socket is alive: 50 output events per 5 s were measured
right through a 20 s silent stall. What says the MODEL spoke is kept apart:
the last output-transcript delta, and the last output audio delta that
carried sound.
Beside the last input transcript, that answers "did the model hear the caller
after it last spoke?" without any text.

**The backend delegation lifecycle.** A GPT-Live voice model may hand a turn
to its backend Responses model (``session.delegation.created``), whose events
arrive wrapped in ``response.event``. A delegation still open at a silent
stall would look exactly like a dead model, so the lifecycle is counted: each
delegation, each backend ``response.created`` / ``completed`` / ``failed`` /
``incomplete``, whether one is open and for how long. Delegation ids are held
only as dictionary keys to pair a response with its delegation; none is
logged. The client side is counted too, by TYPE from an allowlist: context
appends (commentary, instructions, thinking) and ``response.create`` are what
the worker and the plugin send mid-call, and when the last one went out is the
other half of "what happened just before the model went quiet".

The plugin hooks are not on the ``DuplexSession`` interface, so installation
is a capability check and a session without them is reported as
``unsupported``, never guessed at. Every witness is observation only: a
failure inside one is swallowed, because media must never depend on it.
"""
from __future__ import annotations

import array
import base64
import binascii
import math
import sys
import time
from collections import Counter, deque
from typing import Any, Callable, Mapping

#: Server event types counted by name: every type the pinned plugin's
#: GPT-Live handler reads or acknowledges. Anything else is counted as
#: "other", so a type is never copied from the wire into evidence.
SERVER_EVENT_TYPES = frozenset(
    {
        "session.started",
        "session.updated",
        "session.closed",
        "session.usage.updated",
        "session.input_transcript.delta",
        "session.output_transcript.delta",
        "session.output_audio.delta",
        "session.input_audio.muted",
        "session.input_audio.unmuted",
        "session.instructions.appended",
        "session.thinking.appended",
        "session.commentary.appended",
        "session.delegation.created",
        "response.event",
        "error",
    }
)
#: The client event that carries caller audio to the provider.
INPUT_AUDIO_APPEND = "session.input_audio.append"
#: Server events that mean the provider produced something for the caller.
OUTPUT_EVENT_TYPES = frozenset(
    {
        "session.output_audio.delta",
        "session.output_transcript.delta",
        "response.event",
    }
)
#: The provider transcribing the caller: proof it is hearing the input.
INPUT_TRANSCRIPT_EVENT = "session.input_transcript.delta"
#: The model's own speech, as text: proof it SAID something (the audio stream
#: runs whether or not it speaks).
OUTPUT_TRANSCRIPT_EVENT = "session.output_transcript.delta"
OUTPUT_AUDIO_EVENT = "session.output_audio.delta"
#: The voice model handing a turn to its backend, and the backend's events.
DELEGATION_CREATED_EVENT = "session.delegation.created"
RESPONSE_EVENT_ENVELOPE = "response.event"
USAGE_UPDATED_EVENT = "session.usage.updated"
#: Backend response lifecycle types, as the plugin reads them off the envelope.
BACKEND_OPENED = "response.created"
BACKEND_TERMINAL_TYPES = ("response.completed", "response.failed", "response.incomplete")
BACKEND_COUNTED_TYPES = (BACKEND_OPENED, *BACKEND_TERMINAL_TYPES, "response.output_item.done")
#: What ``duplex_backend_last_status`` can say besides "".
BACKEND_STATUSES = (BACKEND_OPENED, *BACKEND_TERMINAL_TYPES)
#: Where a delegation is sent; anything else is recorded as "other".
DELEGATION_TARGETS = ("responses", "client")
#: Client events counted by type. Audio appends have their own counters; an
#: event not listed here is counted as "other" and its type is not kept.
CLIENT_EVENT_TYPES = frozenset(
    {
        "session.start",
        "session.update",
        "session.close",
        "session.input_audio.mute",
        "session.input_audio.unmute",
        "session.instructions.append",
        "session.thinking.append",
        "session.commentary.append",
        "response.create",
        "response.item.create",
    }
)
#: The client events that put context in front of the model mid-call.
CONTEXT_APPEND_TYPES = frozenset(
    {"session.instructions.append", "session.thinking.append", "session.commentary.append"}
)
#: Delegations tracked as open at once. Past it an open event is counted as
#: untracked rather than remembered, so a saturated map says so.
_MAX_OPEN_DELEGATIONS = 64
#: Trailing window for the "is it still flowing" counts.
WINDOW_SECONDS = 5.0
#: PCM16 RMS at or above which a frame is treated as carrying sound. Low on
#: purpose: the question is "silence or not", not "speech or not".
AUDIBLE_RMS = 100.0
#: Unsent appends whose queue times are kept. A stalled send at ten appends a
#: second reaches this after about half an hour; past it the age still reads
#: from the oldest kept stamp, which is the true oldest, and the rest are
#: only counted.
_MAX_PENDING_STAMPS = 20_000
#: Timestamps kept per trailing window, far above any real event rate.
_MAX_WINDOW_STAMPS = 4_096


def pcm16_rms(payload: bytes | bytearray | memoryview) -> float:
    """RMS of little-endian signed 16-bit PCM; 0.0 for anything unreadable."""

    data = bytes(payload)
    if len(data) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(data[: len(data) - len(data) % 2])
    if sys.byteorder != "little":
        samples.byteswap()
    # Every fourth sample is plenty to tell sound from silence, and this runs
    # on every input frame of the call.
    if len(samples) > 256:
        samples = samples[::4]
    if not samples:
        return 0.0
    return math.sqrt(sum(value * value for value in samples) / len(samples))


def _event_type(event: Any) -> str:
    value = _field(event, "type")
    return value if isinstance(value, str) else ""


def _field(event: Any, name: str) -> Any:
    """One field of a raw (dict) or parsed (model) event; ``None`` when absent."""

    if event is None:
        return None
    if isinstance(event, Mapping):
        return event.get(name)
    return getattr(event, name, None)


def _delegation_key(value: Any) -> str:
    """The key a delegation is paired under. A null id is one key of its own:
    the plugin files a response the application started itself under it."""

    return value if isinstance(value, str) and value else "<none>"


class DuplexLivenessWitness:
    """Per-call counters for what reached, and came back from, the provider."""

    def __init__(
        self,
        *,
        relative_ms: Callable[[float], int],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._relative_ms = relative_ms
        self._clock = clock
        self._installed = False
        self._install_failure = ""
        self._frames_pushed = 0
        self._last_push_at: float | None = None
        self._last_audible_push_at: float | None = None
        self._appends_queued = 0
        self._appends_sent = 0
        self._pending_since: deque[float] = deque()
        self._pending_overflow = 0
        self._max_send_lag = 0.0
        self._last_sent_at: float | None = None
        self._last_audible_sent_at: float | None = None
        self._sent_window: deque[float] = deque(maxlen=_MAX_WINDOW_STAMPS)
        self._server_events = 0
        self._event_types: Counter[str] = Counter()
        self._last_server_event_at: float | None = None
        self._last_output_at: float | None = None
        self._last_input_transcript_at: float | None = None
        self._output_window: deque[float] = deque(maxlen=_MAX_WINDOW_STAMPS)
        self._reconnects = 0
        self._withholding: Callable[[], bool] | None = None
        self._frames_withheld = 0
        self._last_withheld_at: float | None = None
        self._last_output_transcript_at: float | None = None
        # An input delta that carried words: an empty one transcribed nothing.
        self._last_input_words_at: float | None = None
        self._last_audible_output_at: float | None = None
        self._context_usage_ratio: float | None = None
        self._delegations_created = 0
        self._delegation_targets: Counter[str] = Counter()
        # delegation key -> when it (or its latest backend response) opened
        self._open_delegations: dict[str, float] = {}
        self._untracked_opens = 0
        self._delegations_abandoned = 0
        self._backend_events: Counter[str] = Counter()
        self._backend_item_types: Counter[str] = Counter()
        self._backend_tool_choice = "unknown"
        self._backend_end_call_declared: bool | None = None
        self._backend_responses_tools_disabled = 0
        self._backend_end_call_items = 0
        self._last_backend_end_call_at: float | None = None
        self._backend_last_status = ""
        self._last_backend_event_at: float | None = None
        self._client_events: Counter[str] = Counter()
        self._last_client_event = ""
        self._last_client_event_at: float | None = None
        self._last_context_append_at: float | None = None

    @property
    def installed(self) -> bool:
        return self._installed

    def note_withheld_by(self, withholding: Callable[[], bool]) -> None:
        """Name who deliberately silences the model's input, and when.

        A hold that swaps caller frames for silence upstream of this witness
        (the duplex opening hold) makes the input read as pushed but never
        audible -- the same shape as a dead line. Frames pushed while
        ``withholding()`` is true are counted as withheld, so the record says
        the silence was chosen. A predicate that raises counts nothing.
        """

        self._withholding = withholding

    def install(self, session: Any) -> bool:
        """Attach to one plugin session. Returns whether every hook was found.

        Refuses a session without ``push_audio``, ``send_event`` and an
        emitter ``on``: a partial install would report a half-blind witness as
        a healthy one.
        """

        if self._installed:
            return True
        push_audio = getattr(session, "push_audio", None)
        send_event = getattr(session, "send_event", None)
        on = getattr(session, "on", None)
        if not (callable(push_audio) and callable(send_event) and callable(on)):
            self._install_failure = "hooks_absent"
            return False
        try:
            on("openai_client_event_queued", self._on_client_event)
            on("openai_server_event_received", self._on_server_event)
            on("session_reconnected", self._on_reconnected)

            def witnessed_push_audio(frame: Any) -> Any:
                self._note_push(frame)
                return push_audio(frame)

            def witnessed_send_event(event: Any) -> Any:
                self._note_queued(event)
                return send_event(event)

            # Instance attributes shadow the class methods, so the plugin's own
            # ``push_audio`` -> ``self.send_event`` path is witnessed too.
            session.push_audio = witnessed_push_audio
            session.send_event = witnessed_send_event
        except Exception as exc:  # noqa: BLE001 - a witness never breaks media
            self._install_failure = type(exc).__name__
            return False
        self._installed = True
        return True

    # hooks

    def _note_push(self, frame: Any) -> None:
        try:
            now = self._clock()
            self._frames_pushed += 1
            self._last_push_at = now
            if self._withheld_now():
                self._frames_withheld += 1
                self._last_withheld_at = now
            if pcm16_rms(getattr(frame, "data", b"")) >= AUDIBLE_RMS:
                self._last_audible_push_at = now
        except Exception:  # noqa: BLE001
            pass

    def _withheld_now(self) -> bool:
        if self._withholding is None:
            return False
        try:
            return bool(self._withholding())
        except Exception:  # noqa: BLE001 - an unreadable hold claims nothing
            return False

    def _note_queued(self, event: Any) -> None:
        try:
            if _event_type(event) != INPUT_AUDIO_APPEND:
                return
            self._appends_queued += 1
            if len(self._pending_since) < _MAX_PENDING_STAMPS:
                self._pending_since.append(self._clock())
            else:
                self._pending_overflow += 1
        except Exception:  # noqa: BLE001
            pass

    def _on_client_event(self, raw: Any) -> None:
        try:
            if _event_type(raw) != INPUT_AUDIO_APPEND:
                if _event_type(raw) in {"session.start", "session.update"}:
                    config = _field(_field(_field(raw, "session"), "delegation"), "responses")
                    if isinstance(config, Mapping):
                        choice = config.get("tool_choice", "auto" if _event_type(raw) == "session.start" else None)
                        if choice is not None:
                            self._backend_tool_choice = (
                                choice if isinstance(choice, str) and choice in {"auto", "none", "required"}
                                else "function" if isinstance(choice, Mapping) and choice.get("type") == "function"
                                else "unknown"
                            )
                        tools = config.get("tools")
                        if isinstance(tools, list):
                            self._backend_end_call_declared = any(
                                isinstance(tool, Mapping) and tool.get("name") == "end_call"
                                for tool in tools
                            )
                self._note_client_event(_event_type(raw))
                return
            now = self._clock()
            self._appends_sent += 1
            self._last_sent_at = now
            self._sent_window.append(now)
            # FIFO: the socket sends appends in the order they were queued.
            if self._pending_since:
                self._max_send_lag = max(self._max_send_lag, now - self._pending_since.popleft())
            elif self._pending_overflow:
                self._pending_overflow -= 1
            audio = raw.get("audio") if isinstance(raw, Mapping) else None
            if isinstance(audio, str) and audio:
                try:
                    decoded = base64.b64decode(audio, validate=False)
                except (binascii.Error, ValueError):
                    decoded = b""
                if pcm16_rms(decoded) >= AUDIBLE_RMS:
                    self._last_audible_sent_at = now
        except Exception:  # noqa: BLE001
            pass

    def _on_server_event(self, event: Any) -> None:
        try:
            now = self._clock()
            kind = _event_type(event)
            self._server_events += 1
            self._last_server_event_at = now
            self._event_types[kind if kind in SERVER_EVENT_TYPES else ("other" if kind else "untyped")] += 1
            if kind in OUTPUT_EVENT_TYPES:
                self._last_output_at = now
                self._output_window.append(now)
            if kind == INPUT_TRANSCRIPT_EVENT:
                self._last_input_transcript_at = now
                if _field(event, "delta"):
                    self._last_input_words_at = now
            elif kind == OUTPUT_TRANSCRIPT_EVENT:
                if _field(event, "delta"):
                    self._last_output_transcript_at = now
            elif kind == OUTPUT_AUDIO_EVENT:
                self._note_output_audio(event, now)
            elif kind == DELEGATION_CREATED_EVENT:
                self._note_delegation(event, now)
            elif kind == RESPONSE_EVENT_ENVELOPE:
                self._note_backend(event, now)
            elif kind == USAGE_UPDATED_EVENT:
                self._note_usage(event)
        except Exception:  # noqa: BLE001
            pass

    def _note_output_audio(self, event: Any, now: float) -> None:
        delta = _field(event, "delta")
        if not isinstance(delta, str) or not delta:
            return
        try:
            decoded = base64.b64decode(delta, validate=False)
        except (binascii.Error, ValueError):
            return
        if pcm16_rms(decoded) >= AUDIBLE_RMS:
            self._last_audible_output_at = now

    def _note_delegation(self, event: Any, now: float) -> None:
        self._delegations_created += 1
        delegation = _field(event, "delegation")
        target = _field(delegation, "target")
        self._delegation_targets[target if target in DELEGATION_TARGETS else "other"] += 1
        self._open(_field(delegation, "id"), now)

    def _note_backend(self, envelope: Any, now: float) -> None:
        inner = _field(envelope, "event")
        kind = _field(inner, "type")
        kind = kind if kind in BACKEND_COUNTED_TYPES else "other"
        self._backend_events[kind] += 1
        if kind == "response.output_item.done":
            item = _field(inner, "item")
            item_type = _field(item, "type")
            self._backend_item_types[
                item_type if item_type in {"message", "reasoning", "function_call"} else "other"
            ] += 1
            if (item_type == "function_call" and _field(item, "name") == "end_call"
                    and _field(item, "status") == "completed"):
                self._backend_end_call_items += 1
                self._last_backend_end_call_at = now
        self._last_backend_event_at = now
        key = _field(envelope, "delegation_id")
        if kind == BACKEND_OPENED:
            if self._backend_tool_choice == "none":
                self._backend_responses_tools_disabled += 1
            self._backend_last_status = kind
            self._open(key, now)
        elif kind in BACKEND_TERMINAL_TYPES:
            self._backend_last_status = kind
            self._open_delegations.pop(_delegation_key(key), None)

    def _open(self, key: Any, now: float) -> None:
        name = _delegation_key(key)
        if name in self._open_delegations:
            # A continuation under an open delegation: the delegation has been
            # open since it first opened, not since this response.
            return
        if len(self._open_delegations) >= _MAX_OPEN_DELEGATIONS:
            self._untracked_opens += 1
            return
        self._open_delegations[name] = now

    def _note_usage(self, event: Any) -> None:
        ratio = _field(_field(event, "context_window"), "usage_ratio")
        # Finite only: the worker's log line is strict JSON, and one infinite
        # value would cost the whole decision line.
        if isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and math.isfinite(ratio):
            self._context_usage_ratio = round(float(ratio), 4)

    def _note_client_event(self, kind: str) -> None:
        try:
            name = kind if kind in CLIENT_EVENT_TYPES else "other"
            now = self._clock()
            self._client_events[name] += 1
            self._last_client_event = name
            self._last_client_event_at = now
            if name in CONTEXT_APPEND_TYPES:
                self._last_context_append_at = now
        except Exception:  # noqa: BLE001
            pass

    def _on_reconnected(self, *_args: Any) -> None:
        self._reconnects += 1
        # A replacement session never delivers the old one's terminal
        # responses, so what was open is abandoned, not still open.
        self._delegations_abandoned += len(self._open_delegations)
        self._open_delegations.clear()

    # reads

    def _ms(self, at: float | None) -> int | None:
        return None if at is None else self._relative_ms(at)

    @staticmethod
    def _within(window: deque[float], now: float) -> int:
        return sum(1 for stamp in window if now - stamp <= WINDOW_SECONDS)

    def _input_heard_since_output(self) -> bool:
        heard = self._last_input_words_at
        said = [at for at in (self._last_output_transcript_at, self._last_audible_output_at) if at is not None]
        return heard is not None and (not said or heard > max(said))

    def _oldest_open_delegation_age(self, now: float) -> float:
        if not self._open_delegations:
            return 0.0
        return max(0.0, now - min(self._open_delegations.values()))

    def _unsent(self) -> int:
        return max(0, self._appends_queued - self._appends_sent)

    def _oldest_unsent_age(self, now: float) -> float:
        if not self._unsent() or not self._pending_since:
            return 0.0
        return max(0.0, now - self._pending_since[0])

    def decision_details(self, *, now: float | None = None) -> dict[str, object]:
        """Flat fields for a single log line, e.g. ``model_unresponsive_decision``."""

        if not self._installed:
            return {"duplex_liveness": "unsupported"}
        at = self._clock() if now is None else float(now)
        return {
            "duplex_last_input_sent_ms": self._ms(self._last_sent_at),
            "duplex_last_audible_input_sent_ms": self._ms(self._last_audible_sent_at),
            "duplex_last_output_received_ms": self._ms(self._last_output_at),
            "duplex_last_input_transcript_ms": self._ms(self._last_input_transcript_at),
            "duplex_appends_sent_last_5s": self._within(self._sent_window, at),
            "duplex_output_events_last_5s": self._within(self._output_window, at),
            "duplex_unsent_appends": self._unsent(),
            "duplex_oldest_unsent_age_ms": round(self._oldest_unsent_age(at) * 1000),
            "duplex_reconnects": self._reconnects,
            "duplex_frames_withheld_by_hold": self._frames_withheld,
            # What the MODEL did, as opposed to the socket.
            "duplex_last_output_transcript_ms": self._ms(self._last_output_transcript_at),
            "duplex_last_audible_output_ms": self._ms(self._last_audible_output_at),
            "duplex_input_heard_since_output": self._input_heard_since_output(),
            "duplex_delegations_created": self._delegations_created,
            "duplex_delegation_open": bool(self._open_delegations),
            "duplex_delegation_open_ms": round(self._oldest_open_delegation_age(at) * 1000),
            "duplex_backend_last_status": self._backend_last_status,
            "duplex_backend_end_call_items": self._backend_end_call_items,
            "duplex_backend_tool_choice": self._backend_tool_choice,
            "duplex_backend_responses_tools_disabled": self._backend_responses_tools_disabled,
            "duplex_last_backend_end_call_ms": self._ms(self._last_backend_end_call_at),
            "duplex_backend_failed": self._backend_events["response.failed"]
            + self._backend_events["response.incomplete"],
            "duplex_context_usage_ratio": self._context_usage_ratio,
            "duplex_last_client_event": self._last_client_event,
            "duplex_last_client_event_ms": self._ms(self._last_client_event_at),
            "duplex_last_context_append_ms": self._ms(self._last_context_append_at),
        }

    def section(self, *, now: float | None = None) -> dict[str, object]:
        """The per-call record for the evidence bundle. Counts and times only."""

        if not self._installed:
            return {"status": "unsupported", "reason": self._install_failure or "not_installed"}
        at = self._clock() if now is None else float(now)
        return {
            "status": "observed",
            "clock": "recording_monotonic_zero_v1",
            "window_seconds": WINDOW_SECONDS,
            "input": {
                "frames_pushed": self._frames_pushed,
                "last_push_ms": self._ms(self._last_push_at),
                "last_audible_push_ms": self._ms(self._last_audible_push_at),
                # Frames a hold upstream deliberately turned into silence.
                "frames_withheld_by_hold": self._frames_withheld,
                "last_withheld_push_ms": self._ms(self._last_withheld_at),
                "appends_queued": self._appends_queued,
                "appends_sent": self._appends_sent,
                "last_input_sent_ms": self._ms(self._last_sent_at),
                "last_audible_input_sent_ms": self._ms(self._last_audible_sent_at),
                "unsent_appends": self._unsent(),
                "oldest_unsent_age_ms": round(self._oldest_unsent_age(at) * 1000),
                "max_send_lag_ms": round(self._max_send_lag * 1000),
                "appends_sent_last_window": self._within(self._sent_window, at),
            },
            "output": {
                "server_events": self._server_events,
                "last_server_event_ms": self._ms(self._last_server_event_at),
                "last_output_received_ms": self._ms(self._last_output_at),
                "last_input_transcript_ms": self._ms(self._last_input_transcript_at),
                "output_events_last_window": self._within(self._output_window, at),
                "event_types": dict(sorted(self._event_types.items())),
                "last_output_transcript_ms": self._ms(self._last_output_transcript_at),
                "last_audible_output_ms": self._ms(self._last_audible_output_at),
                "input_heard_since_output": self._input_heard_since_output(),
                "context_usage_ratio": self._context_usage_ratio,
            },
            "delegation": {
                "created": self._delegations_created,
                "targets": dict(sorted(self._delegation_targets.items())),
                "open": len(self._open_delegations),
                "oldest_open_ms": round(self._oldest_open_delegation_age(at) * 1000),
                "abandoned_at_reconnect": self._delegations_abandoned,
                # Open EVENTS refused at the cap (a delegation and its
                # backend response are two), not delegations.
                "untracked_opens": self._untracked_opens,
                "backend_events": dict(sorted(self._backend_events.items())),
                "backend_item_types": dict(sorted(self._backend_item_types.items())),
                "tool_choice": self._backend_tool_choice,
                "end_call_declared": self._backend_end_call_declared,
                "responses_started_with_tools_disabled": self._backend_responses_tools_disabled,
                "end_call_items": self._backend_end_call_items,
                "last_end_call_item_ms": self._ms(self._last_backend_end_call_at),
                "backend_last_status": self._backend_last_status,
                "last_backend_event_ms": self._ms(self._last_backend_event_at),
            },
            "client_events": {
                "counts": dict(sorted(self._client_events.items())),
                "last": self._last_client_event,
                "last_ms": self._ms(self._last_client_event_at),
                "last_context_append_ms": self._ms(self._last_context_append_at),
            },
            "reconnects": self._reconnects,
        }


__all__ = [
    "AUDIBLE_RMS",
    "BACKEND_COUNTED_TYPES",
    "BACKEND_STATUSES",
    "CLIENT_EVENT_TYPES",
    "DELEGATION_CREATED_EVENT",
    "DuplexLivenessWitness",
    "INPUT_AUDIO_APPEND",
    "INPUT_TRANSCRIPT_EVENT",
    "OUTPUT_EVENT_TYPES",
    "OUTPUT_TRANSCRIPT_EVENT",
    "RESPONSE_EVENT_ENVELOPE",
    "SERVER_EVENT_TYPES",
    "WINDOW_SECONDS",
    "pcm16_rms",
]
