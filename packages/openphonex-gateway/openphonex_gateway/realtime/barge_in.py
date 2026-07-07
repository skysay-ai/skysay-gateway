# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

from .frames import AudioFrame
from .metrics import CallMetrics
from .transport import CancellationToken


@dataclass(frozen=True)
class BargeInConfig:
    speech_rms: float = 650.0
    poll_interval_seconds: float = 0.01


class BargeInMonitor:
    """Reads caller frames during assistant playback and cancels output on speech."""

    def __init__(
        self,
        *,
        read_frame: Callable[[], AudioFrame | None],
        cancellation: CancellationToken,
        metrics: CallMetrics,
        config: BargeInConfig | None = None,
    ):
        self.read_frame = read_frame
        self.cancellation = cancellation
        self.metrics = metrics
        self.config = config or BargeInConfig()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._captured: list[AudioFrame] = []
        self._interrupted = False

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def captured_frames(self) -> list[AudioFrame]:
        with self._lock:
            return list(self._captured)

    def start(self) -> None:
        if self._thread:
            return
        self.metrics.mark("barge_in_monitor_started")
        self._thread = threading.Thread(target=self._run, name="openphonex-barge-in-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.metrics.mark("barge_in_monitor_stopped", interrupted=self._interrupted, captured_frames=len(self.captured_frames()))

    def _run(self) -> None:
        while not self._stop.is_set() and not self.cancellation.cancelled:
            frame = self.read_frame()
            if frame is None:
                time.sleep(self.config.poll_interval_seconds)
                continue
            with self._lock:
                self._captured.append(frame)
            if frame.rms >= self.config.speech_rms:
                self._interrupted = True
                self.metrics.mark("barge_in_speech_detected", rms=round(frame.rms, 3))
                self.cancellation.cancel()
                self.metrics.mark("barge_in_output_cancelled")
                return
