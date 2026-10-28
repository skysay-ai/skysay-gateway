# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Recover a never-spoken opening once, only where a retry is provably safe.

The pinned adapter deliberately fails its pending ask on reconnection. An
opening may be asked once more only after its command left the old queue and
the replacement accepted the same initial context. Continuous input stays on
the SDK path. This is not a retry of an autonomous turn or a business action.

A request-scoped pipeline (cascade STT/LLM/TTS) is the other proven case:
each opening is its own speech request, and a request that completed is over,
so none of its output can play later. A failed LLM puts its error on the
request; an empty generation reports nothing at all. Either way the evidence
of an empty opening is the playout itself: completed, not interrupted,
nothing audible.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from typing import Any

from duplex_liveness import AUDIBLE_RMS, pcm16_rms

GENERATION_START_SECONDS = 10.0  # Pinned duplex adapter's original ask budget.
RECONNECTED_REPLY_ERROR = "the session reconnected before the model replied"


class EmptyOpeningError(RuntimeError):
    """A completed, uninterrupted opening request that produced nothing audible."""


class _Superseded(Exception):
    """A pending retry whose ownership moved to the caller or a terminal owner."""


def _seed(session: Any) -> str:
    config = session._session_start_event().session.model_dump(exclude_none=True)
    # Tool choice changes when AgentSession admits the opening. Its caller
    # always requests tool_choice=none; tools are never replayed here.
    initial = {key: config.get(key) for key in ("model", "instructions", "input", "audio")}
    return hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()


class OpeningReplyRecovery:
    """One per-call intent, one retry, and one unchanged generation deadline.

    Only GPTLiveSession currently proves this contract: reconnect closes the
    old websocket/tasks, clears the pending adapter ask, and seeds a new
    session from its own history. Other adapters keep their existing outcome.
    """

    def __init__(self, *, ended, caller_revision, may_generate, emit=lambda **d: None,
                 clock=time.monotonic, audible=None) -> None:
        self.ended = ended
        # Request-scoped pipelines only: a monotonic count of the agent's
        # audible playout starts. A duplex engine is recovered by install().
        self.audible = audible
        self.caller_revision = caller_revision
        self.may_generate = may_generate
        self.emit = emit
        self.clock = clock
        self.duplex = None
        self.realtime = None
        self.epoch = 0
        self.used = False
        self.blocked = ""
        self.changed = asyncio.Event()
        self.output = asyncio.Event()
        self.started_seed = ""
        self.sent_seed = ""
        self.expected_command = ""
        self.command_sent_epoch: int | None = None
        self.active = False
        self._subscriptions = []

    def install(self, realtime) -> bool:
        try:
            from livekit.plugins.openai.realtime.gpt_live_model import GPTLiveSession
            from livekit.agents.llm.duplex_adapter import _DuplexRealtimeSession
            if not isinstance(realtime, _DuplexRealtimeSession) or not isinstance(
                    realtime.duplex_session, GPTLiveSession):
                return False
            self.realtime = realtime
            self.duplex = realtime.duplex_session
            for source, event, callback in (
                    (self.duplex, "session_reconnected", self._reconnected),
                    (self.duplex, "openai_client_event_queued", self._client),
                    (self.duplex, "openai_server_event_received", self._server),
                    (realtime, "generation_created", self._generation)):
                # Remember before registering: an emitter can register and
                # then raise. Cleanup must undo that partial installation.
                self._subscriptions.append((source, event, callback))
                source.on(event, callback)
            return True
        except Exception as exc:
            self.close()
            self.emit(stage="install_failed", error_type=type(exc).__name__)
            return False

    def close(self) -> None:
        for source, event, callback in self._subscriptions:
            try:
                source.off(event, callback)
            except Exception as exc:
                self.emit(stage="cleanup_failed", error_type=type(exc).__name__)
        self._subscriptions.clear()
        self.duplex = self.realtime = None

    def _block(self, reason: str) -> None:
        self.blocked = self.blocked or reason
        self.changed.set()

    def _generation(self, _event) -> None:
        self.output.set()
        self._block("generation_observed")

    def _reconnected(self, _event) -> None:
        self.epoch += 1
        self.started_seed = self.sent_seed = ""
        self.changed.set()

    def _client(self, event) -> None:
        kind = event.get("type")
        if kind == "session.input_audio.append" and not self.blocked:
            # Spoken pickup audio is not restored by a text-only history seed.
            # Use the existing audible-input threshold alongside caller VAD
            # and captions; quiet transport noise is not a spoken pickup.
            try:
                if pcm16_rms(base64.b64decode(event.get("audio", ""))) >= AUDIBLE_RMS:
                    self._block("non_silent_input")
            except (ValueError, TypeError):
                self._block("unproven_input")
        elif kind == "session.start":
            config = event.get("session", {})
            initial = {key: config.get(key) for key in ("model", "instructions", "input", "audio")}
            self.sent_seed = hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()
        elif kind == "session.commentary.append":
            content_hash = hashlib.sha256(str(event.get("content", "")).encode()).hexdigest()
            if self.active and content_hash == self.expected_command and self.command_sent_epoch is None:
                # This hook is inside _ws_send, AFTER dequeue. A command still
                # in the SDK queue must never be supplemented by another ask.
                self.command_sent_epoch = self.epoch
            else:
                self._block("context_changed")
        elif kind in {"session.thinking.append", "session.instructions.append", "response.create",
                      "response.item.create", "session.close"}:
            self._block("context_or_action_changed")

    def _server(self, event) -> None:
        kind = event.get("type")
        if kind == "session.started":
            self.started_seed = self.sent_seed
            self.changed.set()
        elif kind == "session.output_audio.delta" and not self.blocked:
            # The adapter consumes audio asynchronously. Sound already queued
            # from the old epoch must decline replay even before a typed burst.
            try:
                if pcm16_rms(base64.b64decode(event.get("delta", ""))) >= AUDIBLE_RMS:
                    self._block("audible_provider_output")
            except (ValueError, TypeError):
                self._block("unproven_output")
        elif kind in {"session.input_transcript.delta", "session.output_transcript.delta"}:
            if event.get("delta"):
                self._block("caller_context_changed" if kind == "session.input_transcript.delta"
                            else "output_caption_observed")
        elif kind in {"session.delegation.created", "response.event", "session.closed"}:
            self._block("action_or_close_observed")

    def _terminal_owned(self) -> bool:
        return self.ended.is_set() or not self.may_generate()

    async def run(self, *, instructions, generate, wait, attach=lambda h: None):
        """Return the original failure unless a single safe recovery succeeds."""
        if self.duplex is None and self.audible is not None and not self.used:
            return await self._run_request_scoped(generate, wait, attach)
        if self.duplex is None or self.used:
            return await self._attempt(generate, wait, attach)
        self.used = True
        self.active = True
        revision = self.caller_revision()
        initial_epoch = self.epoch
        deadline = self.clock() + GENERATION_START_SECONDS
        try:
            # A spoken pickup has audio context not fully represented in the
            # reconnect seed. Recover only before any caller turn or history.
            if revision or any(getattr(item, "role", "") not in {"system", "developer"}
                               for item in self.duplex._history.items):
                self._block("non_initial_context")
            seed = _seed(self.duplex)
            # Pinned SDK command identity; the image reconnect selftest drives
            # generate_reply through the real adapter and proves the match.
            from livekit.agents.llm._realtime.gpt_live import _ASK_INSTRUCTED
            self.expected_command = hashlib.sha256(f"{_ASK_INSTRUCTED}\n\n{instructions}".encode()).hexdigest()
            error = await self._attempt(generate, wait, attach)
            from livekit.agents.llm import RealtimeError
            if not isinstance(error, RealtimeError) or str(error) != RECONNECTED_REPLY_ERROR:
                return error
            while True:
                self.changed.clear()
                remaining = deadline - self.clock()
                if self._terminal_owned():
                    self.emit(stage="superseded", epoch=self.epoch, attempts=0)
                    return None  # An existing terminal/tool owner owns the call.
                if self.caller_revision() != revision:
                    self._block("caller_revision_changed")
                if (self.blocked or self.epoch != initial_epoch + 1
                        or self.command_sent_epoch != initial_epoch or remaining <= 0):
                    self.emit(stage="not_recovered", epoch=self.epoch, attempts=0,
                              reason=self.blocked or "unproven_replacement_or_deadline")
                    return error
                if self.started_seed:
                    if self.started_seed != seed:
                        self.emit(stage="not_recovered", epoch=self.epoch, attempts=0,
                                  reason="replacement_context_changed")
                        return error
                    break
                try:
                    await asyncio.wait_for(self.changed.wait(), min(remaining, .05))
                except asyncio.TimeoutError:
                    pass
            self.emit(stage="retry_requested", epoch=self.epoch, attempts=1,
                      remaining_ms=max(0, round((deadline - self.clock()) * 1000)))
            # A second matching command is expected on the new connection.
            self.command_sent_epoch = None
            try:
                handle = generate()
            except Exception as exc:
                self.emit(stage="retry_refused", epoch=self.epoch, attempts=1,
                          error_type=type(exc).__name__)
                return exc
            attach(handle)
            playing = asyncio.create_task(wait(handle, ended=self.ended))
            output = asyncio.create_task(self.output.wait())
            try:
                while not playing.done() and not output.done():
                    if self._terminal_owned():
                        handle.interrupt(force=True)
                        self.emit(stage="superseded", epoch=self.epoch, attempts=1)
                        return None
                    if self.caller_revision() != revision:
                        self._block("caller_revision_changed")
                    if self.blocked in {
                            "non_silent_input", "unproven_input", "caller_context_changed",
                            "caller_revision_changed", "context_changed", "context_or_action_changed",
                            "action_or_close_observed"}:
                        handle.interrupt(force=True)
                        self.emit(stage="not_recovered", epoch=self.epoch, attempts=1,
                                  reason=self.blocked)
                        # Audio/VAD alone does not prove a new reply owner,
                        # especially while the initial input hold is engaged.
                        return error
                    remaining = deadline - self.clock()
                    if remaining <= 0:
                        handle.interrupt(force=True)
                        self.emit(stage="expired", epoch=self.epoch, attempts=1)
                        return TimeoutError("recovered opening did not start within the original reply budget")
                    await asyncio.wait((playing, output), timeout=min(remaining, .05),
                                       return_when=asyncio.FIRST_COMPLETED)
                # The original deadline bounds start, not the spoken line's
                # length. Normal playout/terminal ownership applies thereafter.
                try:
                    await playing
                except Exception as exc:
                    return exc
                result = handle.exception() if handle.done() else None
                self.emit(stage="settled", epoch=self.epoch, attempts=1,
                          failed=result is not None)
                return result
            finally:
                for task in (playing, output):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(playing, output, return_exceptions=True)
        finally:
            self.active = False
            self.expected_command = ""

    async def _run_request_scoped(self, generate, wait, attach):
        """One empty opening is asked once more; anything else is final.

        A retry needs all of: the request completed, uninterrupted, with
        nothing audible; no caller speech since the opening was asked (a
        caller turn owns the next reply); no terminal or tool owner; and the
        original reply budget not yet spent. Otherwise an empty opening is
        reported, so the caller of this seam can end the call truthfully
        instead of leaving the line silent.
        """
        self.used = True
        revision = self.caller_revision()
        deadline = self.clock() + GENERATION_START_SECONDS
        attempts = 0
        while True:
            heard = self.audible()
            try:
                handle = generate()
            except Exception as exc:
                # Refused before anything was scheduled: the session is not
                # running, and a second ask would be refused the same way.
                return exc
            attach(handle)
            try:
                if attempts:
                    # The retry shares the original budget: it must START to
                    # be heard before the deadline. Once it is audible, normal
                    # playout and interruption own it, however long it is.
                    finished = await self._wait_audible_by(handle, wait, heard, deadline, revision)
                else:
                    finished = await wait(handle, ended=self.ended)
            except _Superseded as superseded:
                self.emit(stage="superseded", attempts=attempts, reason=str(superseded))
                return None
            except asyncio.TimeoutError:
                self.emit(stage="expired", attempts=attempts)
                return TimeoutError("recovered opening did not start within the original reply budget")
            except Exception as exc:
                return exc
            if handle is None or finished is False or self.ended.is_set():
                return None
            try:
                error = handle.exception() if handle.done() else None
            except Exception:
                error = None
            interrupted = bool(getattr(handle, "interrupted", False))
            if self.audible() != heard or interrupted:
                # The opening reached the wire, or the caller took the floor:
                # the conversation owns the call from here, as it always has.
                if attempts:
                    self.emit(stage="settled", attempts=attempts, failed=False)
                return None
            error = error or EmptyOpeningError("the opening produced no audible output")
            if self._terminal_owned():
                self.emit(stage="superseded", attempts=attempts)
                return None
            if self.caller_revision() != revision:
                self.emit(stage="superseded", attempts=attempts, reason="caller_revision_changed")
                return None
            reason = ("retry_failed" if attempts else
                      "budget_spent" if self.clock() >= deadline else "")
            if reason:
                self.emit(stage="not_recovered", attempts=attempts, reason=reason,
                          error_type=type(error).__name__)
                return error
            attempts = 1
            self.emit(stage="retry_requested", attempts=attempts,
                      error_type=type(error).__name__,
                      remaining_ms=max(0, round((deadline - self.clock()) * 1000)))

    async def _wait_audible_by(self, handle, wait, heard, deadline, revision):
        """Await one request's playout until it is audible, bounded by ``deadline``.

        Ownership is decided before the budget: while nothing is audible, a
        caller who starts speaking, or a terminal or tool owner, supersedes
        the retry (it is withdrawn and nothing is reported as failed). Only a
        retry nobody superseded is cut at the deadline.
        """
        playing = asyncio.create_task(wait(handle, ended=self.ended))
        try:
            while not playing.done():
                if self.audible() != heard:
                    return await playing
                if self._terminal_owned() or self.caller_revision() != revision:
                    handle.interrupt(force=True)
                    raise _Superseded("terminal_owner" if self._terminal_owned() else "caller_revision_changed")
                if self.clock() >= deadline:
                    handle.interrupt(force=True)
                    raise asyncio.TimeoutError
                await asyncio.wait((playing,), timeout=.05)
            return playing.result()
        finally:
            if not playing.done():
                playing.cancel()
                await asyncio.gather(playing, return_exceptions=True)

    async def _attempt(self, generate, wait, attach):
        try:
            handle = generate()
            attach(handle)
            await wait(handle, ended=self.ended)
            return handle.exception() if handle is not None and handle.done() else None
        except Exception as exc:
            return exc
