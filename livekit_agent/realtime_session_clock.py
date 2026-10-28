# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Session time for a realtime provider whose protocol reports none.

A voice-to-voice engine is priced on its session's duration: the pinned TTS
line is ``audio_second`` and the worker reads it from the SDK's
``session_duration`` (``component_billing``). GPT-Live's plugin fills that
field with the provider's own session seconds. Gemini Live's
(livekit-plugins-google 1.8.8) reports tokens only, so every Gemini Live
report carried zero seconds: the control plane refused it, or settled it at no
Gemini charge when a tool unit happened to make it look complete.

This adapter is the provider's missing half. The quantity it reports is the
one GPT-Live's provider reports: the seconds a provider connection was open,
summed over the call's connections, silence included
(``docs/HOSTED_AI_COMPONENT_BILLING.md``, "Voice-to-voice engines"):

* a connection opens when the plugin reports it acquired, and closes at the
  plugin's next connection close or when the plugin first sees the error that
  ended it, so a failed connect, the close of a socket that dropped while the
  plugin's receive/send tasks were serving it, and the wait before a reconnect
  are never counted. A failure raised while the plugin is still setting a
  connection up, before those tasks start, is seen only at ``_emit_error``,
  after that socket's close;
* every ``RealtimeModelMetrics`` the plugin emits is ASSIGNED the seconds
  accrued since the previous report, never added to, so nothing is counted
  twice (the pinned plugin sets no session time of its own; a proof in
  ``tests/test_navigation_engines_sdk_proofs.py`` fails first if a bump does);
* ``aclose`` reports the remainder before it awaits anything, so a close the
  worker cancels at its bound cannot lose it;
* the worker calls :func:`report_session_time` immediately before it takes its
  billing snapshot, which covers a close it skipped or one still running.
  Seconds after that snapshot -- the provider socket's own teardown -- are not
  billed.

The SDK's usage collector sums these reports into the same one model summary
the plugin's tokens go to, so the billing normalizer stays engine-neutral.
"""

from __future__ import annotations

import time
from typing import Any, Callable


def session_clock_classes(
    model_cls: Any, session_cls: Any, *, clock: Callable[[], float] = time.monotonic
) -> tuple[Any, Any]:
    """``(model_class, session_class)``: the given pair, reporting connected seconds."""

    class ClockedSession(session_cls):  # type: ignore[misc, valid-type]
        def __init__(self, realtime_model: Any) -> None:
            # Before the base class: its constructor starts the connection task.
            self._skysay_open_since: float | None = None
            self._skysay_unreported = 0.0
            self._skysay_closing = False
            super().__init__(realtime_model)

        def _skysay_end_connection(self) -> None:
            if self._skysay_open_since is not None:
                self._skysay_unreported += max(0.0, clock() - self._skysay_open_since)
                self._skysay_open_since = None

        def _skysay_take_seconds(self) -> float:
            """Connected seconds not yet reported; an open connection keeps running."""

            if self._skysay_open_since is not None:
                now = clock()
                self._skysay_unreported += max(0.0, now - self._skysay_open_since)
                self._skysay_open_since = now
            taken, self._skysay_unreported = self._skysay_unreported, 0.0
            return taken

        def _report_connection_acquired(self, acquire_time: float) -> None:
            if not self._skysay_closing:
                self._skysay_end_connection()  # a connect with no close seen: never overlap two
                self._skysay_open_since = clock()
            super()._report_connection_acquired(acquire_time)

        def _mark_restart_needed(self, on_error: bool = False) -> None:
            # The receive/send task that saw the connection fail: the plugin closes the dead
            # socket (up to the websockets close timeout) before it reports the error.
            if on_error:
                self._skysay_end_connection()
            super()._mark_restart_needed(on_error)

        def _emit_error(self, error: Exception, recoverable: bool) -> None:
            # The plugin reports the error that ended a connection before it waits to retry.
            self._skysay_end_connection()
            super()._emit_error(error, recoverable)

        async def _close_active_session(self) -> None:
            self._skysay_end_connection()
            await super()._close_active_session()

        def emit(self, event: str, *args: Any) -> None:
            # The SDK's own discriminator for a realtime model's usage report.
            if event == "metrics_collected" and args and getattr(args[0], "type", None) == "realtime_model_metrics":
                args[0].session_duration = self._skysay_take_seconds()
            super().emit(event, *args)

        def skysay_report_session_time(self) -> None:
            """Report the connected seconds accrued so far, now."""

            seconds = self._skysay_take_seconds()
            if seconds <= 0:
                return
            from livekit.agents.metrics.base import Metadata, RealtimeModelMetrics

            # Through ``super().emit``: the value is already this interval's own.
            super().emit(
                "metrics_collected",
                RealtimeModelMetrics(
                    request_id="",
                    timestamp=time.time(),
                    label=self._realtime_model.label,
                    session_duration=seconds,
                    input_token_details=RealtimeModelMetrics.InputTokenDetails(),
                    output_token_details=RealtimeModelMetrics.OutputTokenDetails(),
                    # The plugin's own identity, so the collector merges this into the
                    # one summary its token reports go to.
                    metadata=Metadata(
                        model_name=self._realtime_model.model,
                        model_provider=self._realtime_model.provider,
                    ),
                ),
            )

        async def aclose(self) -> None:
            # Before the first await: the worker may cancel a slow close, and nothing
            # after the close request is billed.
            self._skysay_closing = True
            self._skysay_end_connection()
            self.skysay_report_session_time()
            await super().aclose()

    class ClockedModel(model_cls):  # type: ignore[misc, valid-type]
        def session(self, *args: Any, **kwargs: Any) -> Any:
            session = ClockedSession(self)
            self._sessions.add(session)
            return session

    return ClockedModel, ClockedSession


def report_session_time(model: Any) -> None:
    """At the billing snapshot: every clocked session of ``model`` reports what it accrued.

    A model whose sessions keep no clock -- a cascade stack's, GPT-Live's -- is
    left alone, so the caller needs no engine branch.
    """

    for session in list(getattr(model, "_sessions", None) or ()):
        report = getattr(session, "skysay_report_session_time", None)
        if callable(report):
            report()


__all__ = ["report_session_time", "session_clock_classes"]
