# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Typed transcript events for the workspace browser test panel.

The test panel beside the agent editor shows the live transcript straight
from the LiveKit room: the SDK publishes every turn on the ``lk.transcription``
text stream and the browser subscribes. The panel is also built to show,
inline between those turns, the moments a customer cannot otherwise see: a
tool ran, and the AI agent switched to the caller's language. Nothing fed
them. This module is the worker half of that feed.

It publishes one bounded JSON envelope per moment on a second text-stream
topic, ``skysay.transcript-events``, addressed to the workspace participant
only. The channel inherits the room token the control plane minted for that
tenant's session; there is no new API surface and nothing is persisted.

Every payload that reaches the customer's browser is redacted with the
worker's crash-report scrubber BEFORE it is truncated (cutting first would
manufacture the fragment the scrubber can no longer recognise), and the
session is capped in count and per-event size so a chatty tool loop cannot
turn the room into a firehose.

This runs from ``session.on`` callbacks on the media loop. Publishing is
scheduled, never awaited there, and every failure is swallowed into a named
worker log event: a note about a call must never end one.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

try:  # package import in tests; flat /app import in the pinned worker image
    from .error_tracking import redact_text
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from error_tracking import redact_text  # type: ignore[no-redef]

SCHEMA = "skysay.test-transcript-event.v1"
TOPIC = "skysay.transcript-events"

EVENT_TOOL_CALL = "tool_call"
EVENT_LANGUAGE_SWITCH = "language_switch"

#: Hard ceilings on what one browser test may receive. The panel enforces the
#: same count so a worker/web skew cannot lift either side's bound alone.
MAX_EVENTS_PER_SESSION = 200
MAX_EVENT_BYTES = 2048
MAX_NAME_CHARS = 80
MAX_SUMMARY_CHARS = 240
MAX_LANGUAGE_CHARS = 16
#: Sends that may be pending at once; beyond this a moment is dropped and
#: logged rather than queued without bound behind a slow data channel.
MAX_IN_FLIGHT = 8

#: The approved panel renders an explicit switch as one ``language`` note, so
#: the ``switch_language`` tool itself does not also produce a tool row.
SUPPRESSED_TOOLS = frozenset({"switch_language"})

#: Argument keys whose values are never summarised, whatever they hold.
_SECRET_ARGUMENT_MARKERS = (
    "secret",
    "password",
    "passwd",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "private_key",
    "ssn",
    "card",
    "cvv",
)

#: How the panel labels a switch, keyed by the switcher's ``source``.
REASON_BY_SOURCE = {
    "explicit_request": "explicit",
    "provider_stable_evidence": "follow_caller",
}

_TRUNCATED = "…"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + _TRUNCATED


def _redact_then_clip(value: Any, limit: int) -> str:
    """Redact the WHOLE value, then cut. Order is the contract, see module doc."""

    return _clip(redact_text(value), limit)


def _is_secret_key(key: Any) -> bool:
    lowered = str(key).casefold()
    return any(marker in lowered for marker in _SECRET_ARGUMENT_MARKERS)


def _drop_secret_arguments(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): ("[redacted]" if _is_secret_key(key) else _drop_secret_arguments(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_drop_secret_arguments(item) for item in value]
    return value


def summarize_arguments(raw: Any) -> str:
    """One bounded, redacted line describing a tool call's arguments.

    The model hands the SDK a JSON string; a dict reaches here from tests and
    from any future in-process caller. Either way secret-looking keys lose
    their values before the text scrubber runs over the rest.
    """

    parsed: Any = raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return ""
        try:
            parsed = json.loads(stripped)
        except ValueError:
            return _redact_then_clip(stripped, MAX_SUMMARY_CHARS)
    if parsed is None:
        return ""
    if isinstance(parsed, (dict, list)):
        text = json.dumps(
            _drop_secret_arguments(parsed), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    else:
        text = str(parsed)
    return _redact_then_clip(text, MAX_SUMMARY_CHARS)


def summarize_result(raw: Any) -> str:
    """One bounded, redacted line describing what a tool returned."""

    if raw is None:
        return ""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str) and raw.lstrip()[:1] in ("{", "["):
        # A structured result gets the same key-based secret drop as the
        # arguments; a plain sentence goes through the text scrubber only.
        try:
            raw = json.loads(raw)
        except ValueError:
            pass
    if isinstance(raw, (dict, list)):
        text = json.dumps(
            _drop_secret_arguments(raw), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    else:
        text = str(raw)
    text = " ".join(text.split())
    if not text:
        return ""
    return _redact_then_clip(text, MAX_SUMMARY_CHARS)


def _tool_name(value: Any) -> str:
    """A tool name is a customer-defined identifier; it still goes through
    the scrubber before the cut, like every other string in an envelope."""

    return _clip(redact_text(" ".join(str(value or "").split())), MAX_NAME_CHARS)


#: A language is a BCP-47-shaped tag or nothing: never free text.
_LANGUAGE_TAG = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8}){0,3}$")


def _language(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) > MAX_LANGUAGE_CHARS or not _LANGUAGE_TAG.match(text):
        return ""
    return text


class TranscriptEventPublisher:
    """Turn SDK session events into bounded envelopes on the room text stream.

    ``send_text`` is the only I/O: an awaitable that publishes one string to
    the workspace participant. It is injected so the module never imports
    LiveKit and so a test can record exactly what would have left the worker.
    """

    def __init__(
        self,
        *,
        send_text: Callable[[str], Awaitable[Any]],
        emit: Callable[..., None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        schedule: Callable[[Awaitable[Any]], Any] | None = None,
    ) -> None:
        self._send_text = send_text
        self._emit = emit or (lambda event, **details: None)
        self._clock = clock
        self._schedule = schedule
        self._started_at = clock()
        self._seq = 0
        self._published = 0
        self._dropped = 0
        self._closed = False
        self._tool_started_at: dict[str, float] = {}
        self._in_flight: set[Any] = set()

    # -- observed state ---------------------------------------------------

    @property
    def published(self) -> int:
        return self._published

    @property
    def dropped(self) -> int:
        return self._dropped

    def close(self) -> None:
        """Stop publishing; a late tool result after teardown goes nowhere."""

        self._closed = True
        self._tool_started_at.clear()

    # -- session hooks (sync, media-loop safe) ---------------------------

    def on_tool_execution_updated(self, event: Any) -> None:
        """Remember when each tool call started; the executed event has no clock."""

        try:
            update = getattr(event, "update", None)
            if str(getattr(update, "type", "") or "") != "tool_call_started":
                return
            call = getattr(update, "function_call", None)
            call_id = str(getattr(call, "call_id", "") or "")
            if call_id and len(self._tool_started_at) < MAX_EVENTS_PER_SESSION:
                self._tool_started_at.setdefault(call_id, self._clock())
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._note("transcript_event_hook_failed", hook="tool_execution_updated", error=type(exc).__name__)

    def on_function_tools_executed(self, event: Any) -> None:
        """One ``tool_call`` envelope per executed call, in the SDK's order."""

        try:
            calls = list(getattr(event, "function_calls", None) or [])
            outputs = {
                str(getattr(output, "call_id", "") or ""): output
                for output in (getattr(event, "function_call_outputs", None) or [])
            }
            ended_at = self._clock()
            for call in calls:
                name = _tool_name(getattr(call, "name", ""))
                call_id = str(getattr(call, "call_id", "") or "")
                started_at = self._tool_started_at.pop(call_id, None)
                if not name or name in SUPPRESSED_TOOLS:
                    continue
                output = outputs.get(call_id)
                # A call the SDK executed but returned nothing for (cancelled,
                # lost) did not succeed; only a present, non-error output is ok.
                ok = output is not None and not bool(getattr(output, "is_error", False))
                self.publish(
                    self.tool_call_envelope(
                        name=name,
                        arguments=getattr(call, "arguments", None),
                        result=getattr(output, "output", None) if output is not None else None,
                        ok=ok,
                        duration_ms=(
                            max(0, int(round((ended_at - started_at) * 1000)))
                            if started_at is not None
                            else None
                        ),
                    )
                )
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._note("transcript_event_hook_failed", hook="function_tools_executed", error=type(exc).__name__)

    def on_language_switch(self, event: str, **details: Any) -> None:
        """Tee of the switcher's emit: only an APPLIED switch is a moment."""

        try:
            if event != "language_switch_applied":
                return
            self.publish(
                self.language_switch_envelope(
                    from_language=details.get("old_language"),
                    to_language=details.get("language"),
                    source=str(details.get("source") or ""),
                )
            )
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._note("transcript_event_hook_failed", hook="language_switch", error=type(exc).__name__)

    # -- envelopes -------------------------------------------------------

    def _base(self, kind: str) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "type": kind,
            "seq": self._seq + 1,
            "at_ms": max(0, int(round((self._clock() - self._started_at) * 1000))),
        }

    def tool_call_envelope(
        self,
        *,
        name: str,
        arguments: Any,
        result: Any,
        ok: bool,
        duration_ms: int | None,
    ) -> dict[str, Any]:
        envelope = self._base(EVENT_TOOL_CALL)
        envelope.update(
            {
                "name": _tool_name(name),
                "arguments": summarize_arguments(arguments),
                "result": summarize_result(result),
                "ok": bool(ok),
                "duration_ms": duration_ms if isinstance(duration_ms, int) and duration_ms >= 0 else None,
            }
        )
        return envelope

    def language_switch_envelope(
        self, *, from_language: Any, to_language: Any, source: str
    ) -> dict[str, Any]:
        envelope = self._base(EVENT_LANGUAGE_SWITCH)
        envelope.update(
            {
                "from": _language(from_language),
                "to": _language(to_language),
                "reason": REASON_BY_SOURCE.get(source, "follow_caller"),
            }
        )
        return envelope

    # -- publishing ------------------------------------------------------

    def publish(self, envelope: dict[str, Any]) -> bool:
        """Serialise, bound, and schedule one envelope. True when it left."""

        if self._closed:
            return False
        if self._published >= MAX_EVENTS_PER_SESSION:
            self._drop("session_cap")
            return False
        if len(self._in_flight) >= MAX_IN_FLIGHT:
            self._drop("in_flight_cap")
            return False
        try:
            text = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            self._drop("unserializable")
            return False
        if len(text.encode("utf-8")) > MAX_EVENT_BYTES:
            self._drop("event_bytes")
            return False
        try:
            pending = self._send_text(text)
            handle = self._schedule(pending) if self._schedule is not None else asyncio.ensure_future(pending)
        except Exception as exc:  # noqa: BLE001 - scheduling must not raise into the SDK
            # Nothing left, so nothing is counted: a transient scheduler
            # failure must not spend a slot the session cap then refuses.
            self._note("transcript_event_publish_failed", error=type(exc).__name__, stage="schedule")
            return False
        self._seq = int(envelope.get("seq", self._seq + 1))
        self._published += 1
        self._in_flight.add(handle)
        done = getattr(handle, "add_done_callback", None)
        if callable(done):
            done(self._settle)
        else:
            self._in_flight.discard(handle)
        return True

    def _settle(self, handle: Any) -> None:
        self._in_flight.discard(handle)
        try:
            if handle.cancelled():
                return
            exc = handle.exception()
        except Exception:  # noqa: BLE001 - a foreign future shape is not a failure to report twice
            return
        if exc is not None:
            self._note("transcript_event_publish_failed", error=type(exc).__name__, stage="send")

    def _drop(self, reason: str) -> None:
        self._dropped += 1
        self._note("transcript_event_dropped", reason=reason)

    def _note(self, event: str, **details: Any) -> None:
        try:
            self._emit(event, **details)
        except Exception:  # noqa: BLE001 - the log sink is not allowed to take the call down
            pass


def install_transcript_events(session: Any, publisher: TranscriptEventPublisher) -> None:
    """Register the two SDK hooks. The language tee is wired by the caller
    that owns the switcher's emit, because that emit exists before the
    switcher does."""

    session.on("tool_execution_updated", publisher.on_tool_execution_updated)
    session.on("function_tools_executed", publisher.on_function_tools_executed)
