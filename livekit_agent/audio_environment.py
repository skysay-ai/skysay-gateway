# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Optional, outgoing-only LiveKit ambience for a resolved agent selection.

The control plane validates and pins the selection.  This worker module repeats
only the integrity/format checks needed to make a missing or damaged packaged
asset an optional-media failure rather than a voice-call failure.  It does not
open an input stream, alter STT, or synthesize/retime the agent's TTS track.

``BackgroundAudioPlayer`` is a pinned LiveKit primitive: it publishes its own
local ``background_audio`` track and loops a local WAV.  That is intentionally
different from a PCM mixer.  The distinction is important: callers and
evidence can honestly describe this as an independently delivered outgoing
track, not as a receiver-side post-mix we cannot observe.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import wave
from pathlib import Path
from typing import Any, Mapping


ASSET_VERSION = "v1"
OFF_PRESET_ID = "off"
BACKGROUND_TRACK_NAME = "background_audio"
SAMPLE_RATE_HZ = 24_000
RUNTIME_MAX_AMBIENCE_GAIN = 0.20
# An immutable local 20-second WAV should verify and publish immediately.  A
# bounded optional-media setup keeps a damaged filesystem or SDK publication
# from holding the already-admitted caller/call reservation on the startup
# path.  It is intentionally not a new call-admission gate.
BACKGROUND_AUDIO_START_TIMEOUT_SECONDS = 2.0
_PRESET_IDS = frozenset({"quiet-office", "call-centre", "cafe", "street", "television"})


class AudioEnvironmentUnavailable(RuntimeError):
    """The optional ambience cannot start; core media must continue."""


def asset_root(environ: Mapping[str, str] | None = None) -> Path:
    source = os.environ if environ is None else environ
    explicit = str(source.get("SKYSAY_AUDIO_ENVIRONMENT_ASSET_ROOT") or "").strip()
    if explicit:
        return Path(explicit)
    return Path(__file__).resolve().parent / "audio-environment" / ASSET_VERSION


def _normalise_resolved(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AudioEnvironmentUnavailable("audio environment selection is missing")
    allowed = {
        "preset_id",
        "asset_version",
        "volume",
        "runtime_gain_linear",
        "runtime_gain_db",
        "channel",
    }
    if set(value) - allowed:
        raise AudioEnvironmentUnavailable("audio environment selection has unsupported fields")
    preset_id = str(value.get("preset_id") or "").strip().lower()
    version = str(value.get("asset_version") or "").strip()
    volume = value.get("volume")
    channel = str(value.get("channel") or "")
    if version != ASSET_VERSION or type(volume) is not int:
        raise AudioEnvironmentUnavailable("audio environment selection is invalid")
    if preset_id == OFF_PRESET_ID:
        if volume != 0 or channel != "none":
            raise AudioEnvironmentUnavailable("off audio environment is malformed")
        return {"preset_id": OFF_PRESET_ID, "asset_version": ASSET_VERSION, "volume": 0}
    if preset_id not in _PRESET_IDS or not 1 <= volume <= 100:
        raise AudioEnvironmentUnavailable("audio environment selection is unsupported")
    expected_gain = round(RUNTIME_MAX_AMBIENCE_GAIN * volume / 100, 6)
    gain = value.get("runtime_gain_linear")
    if type(gain) not in (int, float) or isinstance(gain, bool) or not math.isfinite(float(gain)):
        raise AudioEnvironmentUnavailable("audio environment gain is invalid")
    if round(float(gain), 6) != expected_gain or channel != "outgoing_background_track":
        raise AudioEnvironmentUnavailable("audio environment selection is not control-plane resolved")
    return {
        "preset_id": preset_id,
        "asset_version": ASSET_VERSION,
        "volume": volume,
        "runtime_gain_linear": expected_gain,
    }


def _manifest(root: Path) -> dict[str, Any]:
    try:
        document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise AudioEnvironmentUnavailable("audio environment manifest is unavailable") from exc
    if not isinstance(document, dict) or document.get("asset_version") != ASSET_VERSION:
        raise AudioEnvironmentUnavailable("audio environment manifest is unsupported")
    return document


def _manifest_asset_records(root: Path) -> dict[str, dict[str, Any]]:
    """Return exactly one immutable manifest row for every supported preset.

    The control-plane catalogue rejects duplicate, incomplete and extra rows
    before advertising a preset.  The worker must enforce the same identity
    contract rather than allowing a different first matching row to choose the
    WAV it publishes.  This is local package validation, not a second
    catalogue or a provider lookup.
    """

    assets = _manifest(root).get("assets")
    if not isinstance(assets, list):
        raise AudioEnvironmentUnavailable("audio environment manifest has no assets")
    records: dict[str, dict[str, Any]] = {}
    for item in assets:
        if not isinstance(item, dict):
            raise AudioEnvironmentUnavailable("audio environment manifest asset is invalid")
        preset_id = str(item.get("preset_id") or "")
        if preset_id not in _PRESET_IDS:
            raise AudioEnvironmentUnavailable("audio environment manifest has an unsupported asset")
        if preset_id in records:
            raise AudioEnvironmentUnavailable("audio environment manifest has a duplicate asset")
        records[preset_id] = item
    if set(records) != _PRESET_IDS:
        raise AudioEnvironmentUnavailable("audio environment catalogue is incomplete")
    return records


def verified_preset_asset(
    preset_id: str, asset_version: str, *, root: Path | None = None
) -> Path | None:
    """Verify an immutable asset identity without assuming its delivery channel.

    Both an outgoing agent track and a Simulation Lab caller-input mixer use
    the same signed-in-control-plane catalogue bytes, but deliberately use
    different gain/channel contracts. This narrow helper shares only the
    asset-integrity fact; it cannot turn caller noise into output ambience.
    """

    preset_id = str(preset_id or "").strip().lower()
    if asset_version != ASSET_VERSION:
        raise AudioEnvironmentUnavailable("audio environment version is unsupported")
    if preset_id == OFF_PRESET_ID:
        return None
    if preset_id not in _PRESET_IDS:
        raise AudioEnvironmentUnavailable("audio environment selection is unsupported")
    root = asset_root() if root is None else root
    record = _manifest_asset_records(root).get(preset_id)
    if not isinstance(record, dict):
        raise AudioEnvironmentUnavailable("audio environment asset is unavailable")
    filename = str(record.get("filename") or "")
    if not filename or Path(filename).name != filename or not filename.endswith(".wav"):
        raise AudioEnvironmentUnavailable("audio environment filename is invalid")
    path = root / filename
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AudioEnvironmentUnavailable("audio environment asset is unavailable") from exc
    if hashlib.sha256(payload).hexdigest() != str(record.get("sha256") or ""):
        raise AudioEnvironmentUnavailable("audio environment asset integrity check failed")
    try:
        # Python 3.14's stdlib wave reader treats a Path as a file-like
        # object, unlike older releases; pass the string explicitly so the
        # worker image and local tests take the same path.
        with wave.open(str(path), "rb") as handle:
            actual = (
                handle.getnchannels(),
                handle.getsampwidth(),
                handle.getframerate(),
                handle.getcomptype(),
                handle.getnframes(),
            )
    except (OSError, wave.Error, EOFError) as exc:
        raise AudioEnvironmentUnavailable("audio environment asset format is invalid") from exc
    if (
        actual != (1, 2, SAMPLE_RATE_HZ, "NONE", int(record.get("frame_count") or 0))
        or actual[4] != SAMPLE_RATE_HZ * 20
    ):
        raise AudioEnvironmentUnavailable("audio environment asset format is unsupported")
    if int(record.get("duration_ms") or 0) != 20_000:
        raise AudioEnvironmentUnavailable("audio environment asset duration is unsupported")
    return path


def verified_asset_path(resolved: Mapping[str, Any], *, root: Path | None = None) -> Path | None:
    """Return a verified WAV for a resolved outgoing environment, or Off."""

    selection = _normalise_resolved(resolved)
    return verified_preset_asset(
        str(selection["preset_id"]), str(selection["asset_version"]), root=root
    )


async def start_outgoing_audio_environment(
    resolved: Mapping[str, Any] | None,
    *,
    room: Any,
    agent_session: Any,
    root: Path | None = None,
    player_factory: Any = None,
) -> Any | None:
    """Start a separately-published background track, never an STT input.

    The late import keeps control-plane/unit-test imports independent of the
    LiveKit SDK and makes the pinned primitive obvious at this sole seam.
    """

    # A dispatch created before this optional field existed has no selection.
    # That historic absence means the established quiet default, not a guessed
    # named preset and not an optional-media error on every legacy call. A
    # present malformed value still fails closed below and is reported by the
    # production worker.
    if resolved is None:
        return None
    player: Any | None = None
    try:
        # Verification does local file I/O, hashing and WAV parsing.  Keep it
        # out of the async media loop and give the whole optional setup one
        # small budget.  If it misses that budget, the core call continues
        # after the best-effort cleanup below.
        async with asyncio.timeout(BACKGROUND_AUDIO_START_TIMEOUT_SECONDS):
            selection = _normalise_resolved(resolved)
            path = await asyncio.to_thread(verified_asset_path, resolved, root=root)
            if path is None:
                return None
            if player_factory is None:
                from livekit.agents.voice.background_audio import AudioConfig, BackgroundAudioPlayer
                player = BackgroundAudioPlayer(
                    ambient_sound=AudioConfig(
                        str(path),
                        volume=float(selection["runtime_gain_linear"]),
                        fade_in=0.25,
                        fade_out=0.25,
                    )
                )
            else:
                player = player_factory(
                    str(path),
                    float(selection["runtime_gain_linear"]),
                )
            await player.start(room=room, agent_session=agent_session)
    except asyncio.CancelledError:
        # A caller/job cancellation can interrupt `start` after the player
        # has published its loop but before this function returns it to the
        # owner.  Do not orphan that player: close its owned task before
        # preserving cancellation semantics for the worker supervisor.
        await _close_after_unsuccessful_start(player)
        raise
    except AudioEnvironmentUnavailable:
        await _close_after_unsuccessful_start(player)
        raise
    except Exception as exc:
        # An optional track may fail while importing its SDK primitive,
        # constructing it, or publishing it.  None of those failures may
        # escape as a core-call failure; if construction got far enough to
        # allocate media, release it before reporting one safe optional-media
        # error to the production worker.
        await _close_after_unsuccessful_start(player)
        raise AudioEnvironmentUnavailable("background audio could not start") from exc
    return player


async def start_filler_background_player(
    *,
    room: Any,
    agent_session: Any,
    player_factory: Any = None,
) -> Any | None:
    """Publish the background track with no ambience, for acknowledgements only.

    The acknowledgement speaker needs a separately published track of its own
    (never the ambience player: cutting a filler flushes the player's queue,
    which must not touch ambience).  Same track name, so the evidence sidecar
    keeps ignoring it.  Failures leave the core call untouched and are
    reported to the caller as ``None``.
    """

    player: Any | None = None
    try:
        async with asyncio.timeout(BACKGROUND_AUDIO_START_TIMEOUT_SECONDS):
            if player_factory is None:
                from livekit.agents.voice.background_audio import BackgroundAudioPlayer

                player = BackgroundAudioPlayer()
            else:
                player = player_factory()
            await player.start(room=room, agent_session=agent_session)
    except asyncio.CancelledError:
        await _close_after_unsuccessful_start(player)
        raise
    except Exception:  # noqa: BLE001 - optional media only
        await _close_after_unsuccessful_start(player)
        return None
    return player


async def _close_after_unsuccessful_start(player: Any | None) -> None:
    """Release a partially started optional player without masking its cause."""

    if player is None:
        return
    try:
        await close_outgoing_audio_environment(player)
    except asyncio.CancelledError:
        # `close_outgoing_audio_environment` has already waited for its owned
        # close task; this is a second external cancellation and must surface.
        raise
    except Exception:
        # The enclosing call still owns room teardown, which removes its local
        # publications.  The original optional-start error is more useful than
        # a secondary cleanup implementation detail.
        pass


async def close_outgoing_audio_environment(player: Any | None) -> None:
    """Close only the optional track; callers keep their core teardown path."""

    if player is None:
        return
    closer = getattr(player, "aclose", None)
    if not callable(closer):
        # This feature deliberately relies on the pinned LiveKit primitive's
        # async lifecycle.  Silently accepting a different object would turn
        # an unpublish failure into a false successful teardown and leave its
        # local track alive until the room happens to disappear.  The caller
        # treats this as optional-media diagnostic, then continues its normal
        # session and room teardown.
        raise AudioEnvironmentUnavailable("background audio player has no asynchronous close")
    try:
        close_task = asyncio.ensure_future(closer())
    except Exception as exc:
        raise AudioEnvironmentUnavailable("background audio player close is invalid") from exc

    cancellation_requested = False
    while not close_task.done():
        try:
            # Shield the owned close so a job cancellation cannot strand the
            # player between terminal start/teardown and room cleanup.  We
            # re-raise that cancellation after the close task reaches done.
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            cancellation_requested = True

    try:
        close_task.result()
    except asyncio.CancelledError as exc:
        if cancellation_requested:
            raise asyncio.CancelledError from exc
        raise AudioEnvironmentUnavailable("background audio player close was cancelled") from exc
    except Exception:
        if cancellation_requested:
            raise asyncio.CancelledError
        raise
    if cancellation_requested:
        raise asyncio.CancelledError


__all__ = [
    "ASSET_VERSION",
    "BACKGROUND_TRACK_NAME",
    "start_filler_background_player",
    "AudioEnvironmentUnavailable",
    "asset_root",
    "close_outgoing_audio_environment",
    "start_outgoing_audio_environment",
    "verified_asset_path",
    "verified_preset_asset",
]
