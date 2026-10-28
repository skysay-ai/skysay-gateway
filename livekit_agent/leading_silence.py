# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Start the opening at its first sound, not at the provider's leading silence.

A synthesis can begin with silence the provider generated before the first
word: measured on the worker's own builder, Gemini 3.8 Flash-Lite TTS opens
with 200-310 ms of it, ElevenLabs Flash v2.5 with 40-60 ms. Played as it is,
that silence is dead air the callee waits through after their hello, and it
moves the speaking edge (``opening_first_audio_ms``) onto a frame nobody can
hear. :func:`trim_leading_silence` drops it from the front of one synthesis.

Bounded both ways, so it can only ever remove quiet:

* a window counts as sound once it rises above :data:`LEADING_SILENCE_FLOOR_DBFS`
  (the decider's own speech floor), and :data:`LEADING_SILENCE_PREROLL_SECONDS`
  before that window are kept so a soft onset is not clipped;
* at most :data:`LEADING_SILENCE_MAX_SECONDS` are dropped; past that the rest
  of the synthesis passes untouched, silent or not.

Only the front is trimmed. A pause inside the line is the voice's own
phrasing, and every frame after the first sound passes exactly as produced.
Frame metadata is never lost: the userdata of a dropped frame is carried onto
the first frame that plays.

Pure: frames are read through their ``data``/``sample_rate``/
``num_channels``/``samples_per_channel``/``userdata`` attributes and built
through the ``make_frame`` the caller supplies (``rtc.AudioFrame`` in the
worker), so the rule is tested without the LiveKit SDK.
"""

from __future__ import annotations

import array
import inspect
import math
from typing import Any, AsyncIterator, Callable

#: A 10 ms window above this is sound. The opening decider's speech floor.
LEADING_SILENCE_FLOOR_DBFS = -50.0
#: The most silence dropped from the front of one synthesis.
LEADING_SILENCE_MAX_SECONDS = 0.5
#: Kept before the first window of sound, so its onset is not clipped.
LEADING_SILENCE_PREROLL_SECONDS = 0.02
_WINDOW_SECONDS = 0.01
_FULL_SCALE = 32768.0


def first_sound_sample(samples: array.array, *, sample_rate: int, num_channels: int) -> int | None:
    """The first interleaved sample index of a window above the floor, or ``None``."""

    window = max(1, int(sample_rate * _WINDOW_SECONDS)) * max(1, num_channels)
    for start in range(0, len(samples), window):
        chunk = samples[start : start + window]
        if not chunk:
            break
        rms = math.sqrt(sum(value * value for value in chunk) / len(chunk))
        if rms > 0 and 20 * math.log10(rms / _FULL_SCALE) > LEADING_SILENCE_FLOOR_DBFS:
            return start
    return None


async def trim_leading_silence(
    stream: Any, *, make_frame: Callable[..., Any], trimmed: list[float] | None = None
) -> AsyncIterator[Any]:
    """Yield ``stream``'s frames without the silence before its first sound.

    ``trimmed``, when given, holds the seconds dropped in ONE slot of its
    own (appended once, then overwritten), for the caller's evidence. The
    slot is current before every await: a reader that looks while the front
    is still being searched (a call ending in the provider's silence) sees
    what has been dropped so far, never nothing.
    """

    if inspect.isawaitable(stream):
        stream = await stream
    if stream is None:
        return
    dropped = 0.0
    carried: dict[Any, Any] = {}
    #: The latest silent samples not yet dropped: the preroll the first
    #: sound is played with, even when it arrives in a frame of its own.
    tail = array.array("h")
    searching = True
    shape = (0, 1)
    slot: int | None = None

    def record() -> None:
        nonlocal slot
        if trimmed is None:
            return
        if slot is None:
            slot = len(trimmed)
            trimmed.append(round(dropped, 3))
        else:
            trimmed[slot] = round(dropped, 3)

    try:
        async for frame in stream:
            if not searching:
                yield frame
                continue
            sample_rate = int(frame.sample_rate)
            channels = max(1, int(frame.num_channels))
            per_channel = int(frame.samples_per_channel)
            samples = array.array("h", bytes(frame.data)[: per_channel * channels * 2])
            onset = first_sound_sample(samples, sample_rate=sample_rate, num_channels=channels)
            preroll = int(LEADING_SILENCE_PREROLL_SECONDS * sample_rate) * channels
            shape = (sample_rate, channels)
            combined = tail + samples
            if onset is None:
                keep = combined[max(0, len(combined) - preroll) :] if preroll else array.array("h")
                releasing = (len(combined) - len(keep)) / channels / sample_rate
                if dropped + releasing <= LEADING_SILENCE_MAX_SECONDS:
                    dropped += releasing
                    tail = keep
                    carried.update(_userdata(frame))
                    record()
                    continue
            searching = False
            sound_at = len(tail) + (len(samples) if onset is None else onset)
            budget = max(0, int(round((LEADING_SILENCE_MAX_SECONDS - dropped) * sample_rate))) * channels
            cut = min(budget, max(0, sound_at - preroll))
            cut -= cut % channels
            dropped += cut / channels / sample_rate
            if cut == 0 and not tail:
                out = frame
            else:
                kept = combined[cut:]
                out = make_frame(
                    data=kept.tobytes(),
                    sample_rate=sample_rate,
                    num_channels=channels,
                    samples_per_channel=len(kept) // channels,
                )
                carried.update(_userdata(frame))
            if carried:
                _carry(out, carried)
            # The front is final: recorded before the first frame plays.
            record()
            yield out
        if searching and (tail or carried) and shape[0]:
            # A synthesis that never rose above the floor still plays its
            # kept tail, so it produces a frame whenever it produced one before.
            sample_rate, channels = shape
            out = make_frame(
                data=tail.tobytes(),
                sample_rate=sample_rate,
                num_channels=channels,
                samples_per_channel=len(tail) // channels,
            )
            _carry(out, carried)
            record()
            yield out
    finally:
        # An empty or cancelled synthesis still leaves its (possibly zero) slot.
        record()


def _userdata(frame: Any) -> dict[Any, Any]:
    try:
        return dict(frame.userdata or {})
    except Exception:  # noqa: BLE001 - a frame without userdata carries none
        return {}


def _carry(frame: Any, carried: dict[Any, Any]) -> None:
    try:
        for key, value in carried.items():
            frame.userdata.setdefault(key, value)
    except Exception:  # noqa: BLE001 - metadata never decides whether audio plays
        pass


__all__ = [
    "LEADING_SILENCE_FLOOR_DBFS",
    "LEADING_SILENCE_MAX_SECONDS",
    "LEADING_SILENCE_PREROLL_SECONDS",
    "first_sound_sample",
    "trim_leading_silence",
]
