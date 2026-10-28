# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""The room recorder every Skysay call runs: one sidecar participant per call.

A separate LiveKit room participant subscribes to the caller's and the agent's
tracks and feeds :class:`livekit_evidence.SynchronizedEvidenceRecorder`, so the
agent's audio is captured after playout and interruptions rather than when TTS
merely generated it. The phone worker (``production_agent.py``) and the
homepage demo worker (``browser_demo_agent.py``) both record through this one
implementation; it lives in its own module so neither imports the other.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Callable

try:  # package import in tests; flat /app import in the pinned worker image
    from .audio_environment import BACKGROUND_TRACK_NAME
    from .evidence_contract import is_noncaller_ambience_track
    from .livekit_evidence import SynchronizedEvidenceRecorder, inbound_rtp_receive_stats
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from audio_environment import BACKGROUND_TRACK_NAME
    from evidence_contract import is_noncaller_ambience_track
    from livekit_evidence import SynchronizedEvidenceRecorder, inbound_rtp_receive_stats

# How often the caller track's cumulative inbound-RTP counters are re-read.
# The counters are cumulative, so this is a staleness bound on the artifact's
# worst case (a hang-up between samples), not a sampling rate anything is
# averaged over. Ten seconds keeps at most ten seconds of a call unaccounted
# while costing one FFI round-trip per ~500 audio frames.
RECEIVE_STATS_SAMPLE_SECONDS = 10.0
# Hard deadline on one stats read. The capture task is awaited by the teardown
# BEFORE the component-usage report is posted, so a read that never returns
# would strand the reservation exactly the way a metering failure does. A hang
# is not an exception, so the surrounding try/except cannot cover it.
RECEIVE_STATS_READ_BUDGET_SECONDS = 2.0


def safe_component(value: str) -> str:
    return "".join(character for character in value if character.isalnum() or character in "-_")[:96]


def evidence_sidecar_identity(caller_identity: str) -> str:
    """The recorder's room identity, built in ONE place.

    The occupancy probe has to exclude this participant to decide the room is
    empty, and it used to rebuild the string from the same inputs at its own
    call site. Both spellings agreed, but nothing held them to it, and a
    divergence would have been silent in the worst way: the recorder would
    count as a remaining party, the room would never read empty, and the
    occupancy rule -- the direct fix for the incident this all began with --
    would simply never fire, with nothing red anywhere.
    """

    return f"evidence-{safe_component(caller_identity)}"


async def start_evidence_sidecar(
    *,
    room_name: str,
    caller_identity: str,
    agent_identity: str,
    recorder: SynchronizedEvidenceRecorder,
    on_caller_frame: Callable[[bytes], None] | None = None,
    on_media_frame: Callable[..., None] | None = None,
) -> tuple[rtc.Room, set[asyncio.Task[None]], str]:
    # The SDK is the worker's; imported here so the identity and naming
    # helpers above stay importable without it.
    from livekit import api, rtc

    identity = evidence_sidecar_identity(caller_identity)
    token = (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    room = rtc.Room()
    tasks: set[asyncio.Task[None]] = set()

    async def sample_receive_stats(track: rtc.Track, source: str) -> None:
        """Refresh one caller publication's inbound-RTP snapshot. Never raises.

        ``track`` is the SIDECAR's subscription, not the agent session's: this
        function runs inside the sidecar room, and the session subscribes to
        the same publication over its own peer connection. Loss and jitter are
        per subscriber transport, so these counters are a close proxy for what
        STT received and not a measurement of it. ``RTP_RECEIVE_STATS_SOURCE``
        says so on every snapshot; do not relabel it without moving the read.

        Media evidence must never be able to break media capture, so every
        failure here is silent: a missing snapshot degrades the artifact to
        the ``{}`` it already carried, while an exception escaping into the
        teardown would cost the recording itself.
        """
        try:
            stats = await asyncio.wait_for(
                track.get_stats(), timeout=RECEIVE_STATS_READ_BUDGET_SECONDS
            )
            recorder.record_media_receive_stats(
                inbound_rtp_receive_stats(stats), source=source
            )
        except Exception:  # noqa: BLE001 - evidence is never a media gate
            # ``asyncio.TimeoutError`` is the builtin ``TimeoutError`` here, so
            # the deadline lands in this branch. ``CancelledError`` is a
            # BaseException and still propagates, which is correct.
            pass

    async def sample_receive_stats_until_cancelled(track: rtc.Track, source: str) -> None:
        """Poll one caller publication's counters on its OWN task.

        Deliberately NOT awaited from inside the frame loop. A slow
        ``get_stats()`` there would stop draining the ``AudioStream`` for up to
        the read budget; the frames queued during that pause are then stamped
        with the later arrival time, which inserts silence that never happened
        into the synchronized WAVs and stretches every turn offset derived from
        them. Evidence about the media must not deform the media.
        """
        while True:
            await sample_receive_stats(track, source)
            await asyncio.sleep(RECEIVE_STATS_SAMPLE_SECONDS)

    async def capture(track: rtc.Track, role: str) -> None:
        # One cursor per concrete subscription keeps queued frames in media
        # order. A republished track gets a new wall-clock anchor.
        stream_id = object()
        stream = rtc.AudioStream(
            track,
            sample_rate=8_000,
            num_channels=1,
            frame_size_ms=20,
        )
        # Sampled DURING the call, not only at teardown: the caller track is
        # unsubscribed the moment the far end hangs up, and a stats read after
        # that returns nothing. Counters are cumulative within a publication,
        # so the last sample before the hang-up is a complete measurement of
        # everything that publication received.
        stats_source = str(getattr(track, "sid", "") or id(track))
        sampler: asyncio.Task[None] | None = None
        if role == "caller":
            sampler = asyncio.create_task(
                sample_receive_stats_until_cancelled(track, stats_source)
            )
            tasks.add(sampler)
            sampler.add_done_callback(tasks.discard)
        try:
            async for event in stream:
                frame = bytes(event.frame.data)
                received_at = time.monotonic()
                placement = recorder.append_stream_pcm(
                    role,  # type: ignore[arg-type]
                    frame,
                    stream_id=stream_id,
                    received_at=received_at,
                )
                if on_media_frame is not None:
                    # Voicemail detection: both roles, with where the recorder placed the
                    # frame. An append to a bounded ring on the receiving side
                    # and nothing more; it cannot raise into this loop.
                    on_media_frame(role, frame, placement, received_at)
                if role == "caller" and on_caller_frame is not None:
                    # The frame, not a measurement of it: the receiver decides
                    # whether anything will read it before doing the
                    # arithmetic, so a ringing call and an agent with
                    # collection off spend nothing here. It cannot raise into
                    # this loop either -- evidence about the media must not
                    # deform the media, and must not stop it.
                    on_caller_frame(frame)
        finally:
            if sampler is not None:
                # Cancel and do not await: the teardown's gather reaps it, and
                # awaiting here would let an outer cancellation be mistaken for
                # this one's.
                sampler.cancel()
                # The frame loop is over, so this last read cannot deform any
                # timestamp, and it is bounded by the same read budget.
                await sample_receive_stats(track, stats_source)
            await stream.aclose()

    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        # LiveKit's BackgroundAudioPlayer publishes an independently delivered
        # local `background_audio` track.  This sidecar records semantic agent
        # response audio; classifying that bed as an agent response would
        # contaminate turn/evidence timing with non-semantic ambience.
        if is_noncaller_ambience_track(
            publication_name=getattr(publication, "name", ""),
            participant_identity=participant.identity,
            caller_identity=caller_identity,
            agent_identity=agent_identity,
            ambience_track_name=BACKGROUND_TRACK_NAME,
        ):
            return
        role = "caller" if participant.identity == caller_identity else "agent"
        task = asyncio.create_task(capture(track, role))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    room.on("track_subscribed", on_track_subscribed)
    await room.connect(os.environ["LIVEKIT_URL"], token)
    return room, tasks, identity
