# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Agent job-process pool configuration: how many, and how long to let go.

A production call showed invite->accept taking ~7s because LiveKit cold-starts
a fresh job process per call.  Keeping warm idle job processes removes that
cold start; this module resolves how many to keep and best-effort pre-imports
the provider plugin modules so a warm process does not pay the import cost on
its first job.

It also resolves the other end of a job process's life -- the drain deadline a
worker gets when it is asked to stop -- because that is the same pool, decided
from the same environment, and a stop budget that disagrees with the container
runtime's own kill deadline is how a drain becomes a kill.

It also owns the pinned Silero VAD checkpoint, because prewarm is where every
entrypoint loads the VAD and a checkpoint is exactly a warm-process input.

Pure module: no livekit import, no network, no side effects.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any, Callable, List, Mapping, Sequence, Tuple
from warnings import warn

# The Silero VAD checkpoint every worker in this image runs against.
#
# The checkpoint is pinned by digest rather than inherited from whichever
# plugin version is installed.  livekit-plugins-silero swapped its bundled
# file once (1.6.10 replaced v6.1 with a newer checkpoint that scores the
# same audio differently), and a swap that reaches a live call moves every
# barge-in and endpointing threshold with no signal at all.  So the image
# copies the pinned file out of the installed plugin and verifies this digest
# at BUILD time (see the Dockerfiles); a changed upstream artifact fails the
# build, and at runtime the plugin raises ``FileNotFoundError`` rather than
# silently loading whatever it bundles.
#
# Which bytes are pinned is a measured decision.  From 2026-08-29 the pin held
# the v6.1 checkpoint (sha256 597d30b3...), because every interruption
# threshold had been measured on it.  Decided 2026-09-15: the pin now names
# the checkpoint bundled with livekit-plugins-silero 1.8.1, the installed
# plugin.  Offline replay of 37 recorded room-side caller tracks (a Sinhala
# cascade batch, a GPT-Live batch and the English claims demo) showed v6.1
# under-detecting quiet narrowband PSTN callers: over the 16 signal-bearing
# calls it detected 50% of the loud caller audio where the bundled checkpoint
# detected 68%, and on three calls its probability never crossed the 0.5
# activation threshold (peak 0.33 on a call the engine transcribed three
# times), so the no-caller-speech rule hung up on customers who were talking
# and barge-in could not fire.  The activation threshold and
# ``min_silence_duration`` are deliberately unchanged by that decision: it
# swaps the model, and the live thresholds are re-measured separately.
try:
    from .capabilities.workflow_listener import SILERO_VAD_PINNED_SHA256
except ImportError:  # flat worker image
    from capabilities.workflow_listener import SILERO_VAD_PINNED_SHA256

#: Resolved relative to this module so the one constant serves both layouts:
#: ``/app/prewarm.py`` in the built image, and the source tree in a checkout.
#: The name carries no model version on purpose: the digest above is the
#: identity, and the checkpoint's own ONNX metadata names no version.
SILERO_VAD_PINNED_PATH = Path(__file__).resolve().parent / "models" / "silero_vad_pinned.onnx"


def pinned_silero_vad_loader(load: Callable[..., Any]) -> Callable[..., Any]:
    """Bind the pinned checkpoint to a ``silero.VAD.load`` callable.

    Every VAD load site in this image goes through here or passes
    ``onnx_file_path`` itself, so no code path can pick up whatever checkpoint
    the installed plugin happens to bundle.  Returning a bound callable rather
    than loading here keeps this module free of any livekit import.
    """

    return functools.partial(load, onnx_file_path=SILERO_VAD_PINNED_PATH)


def resolve_idle_process_count(
    environ: Mapping[str, str],
    *,
    default: int = 2,
    minimum: int = 1,
    maximum: int = 16,
    var: str = "SCH236_NUM_IDLE_PROCESSES",
) -> int:
    """Resolve the warm idle process count from ``environ[var]``.

    Absent, blank, or unparseable values fall back to ``default``.  The result
    is clamped into ``[minimum, maximum]`` so the server always keeps at least
    one warm process and a typo / hostile env value (e.g. ``500``) cannot spawn
    an unbounded number of resident job processes and OOM the voice host.
    """
    raw = (environ.get(var) or "").strip()
    if not raw:
        count = default
    else:
        try:
            count = int(raw)
        except ValueError:
            count = default
    return max(minimum, min(count, maximum))


#: How many calls this worker is willing to run at once.
#:
#: This is the denominator of the worker's load, and with it the admission
#: gate becomes a job count rather than a CPU reading. The library's default
#: ``load_fnc`` is a 2.5 s moving average of the container's CPU fraction, and
#: on a two-core box a single job-process start is roughly half of that budget
#: for as long as it runs: the production host logged
#: ``full capacity, load 0.7258, threshold 0.7`` at 16:20:54Z during boot, with
#: no call anywhere on the box, purely because two job processes were
#: initialising. A gate that refuses a call because the worker is getting
#: READY for calls is measuring the wrong thing. (The phone
#: agents admit by an exact count in ``phone_admission`` instead, and give the
#: library's gate no load; this budget still sizes their job accounting.)
#:
#: ``5`` is provisional and deliberately not derived from the CPU measurement
#: that does not exist yet: the batch-window counters were lost to a container
#: recreate, so no per-call steady-state coefficient has been fitted. What is
#: known is that four concurrent calls ran on this host without the gate
#: tripping. Raise it only against a measured steady-state number.
#:
#: What the number means at the gate is NOT "five concurrent calls". The
#: library marks itself full at ``effective_load >= load_threshold`` (0.7 in
#: production), and reserves a slot per accepted-not-yet-launched job. WHILE AT
#: LEAST ONE JOB IS ACTIVE it estimates a reserved job at ``load/active_jobs``,
#: which over a job count is exactly ``1/N``, so with ``N = 5`` the worker
#: accepts while fewer than 3.5 jobs are counted -- it admits a fourth call and
#: refuses a fifth. That is the ceiling to compare against the dialer's
#: concurrency, and it is why raising this to match a campaign's concurrency
#: exactly would leave no headroom.
#:
#: With NO job active the library estimates a reservation differently --
#: ``load_threshold / max(num_idle_processes, 1)`` -- so the reservation's
#: weight there depends on the idle pool, not on this number. Read off the real
#: 1.8.1 gate, at zero active jobs with one reservation in flight: idle 4 gives
#: 0.175, idle 3 gives 0.233, idle 2 (this module's default) gives 0.35, and
#: idle 1 gives exactly 0.7 -- which is FULL, so a worker keeping a single warm
#: process refuses a second dispatch arriving in the same instant as its first.
#: That branch is the library's and predates this setting; it is written down
#: because the arithmetic above would otherwise read as covering it.
MAX_CONCURRENT_JOBS_VAR = "SKYSAY_WORKER_MAX_JOBS"
DEFAULT_MAX_CONCURRENT_JOBS = 5


def resolve_max_concurrent_jobs(
    environ: Mapping[str, str],
    *,
    default: int = DEFAULT_MAX_CONCURRENT_JOBS,
    minimum: int = 1,
    maximum: int = 64,
    var: str = MAX_CONCURRENT_JOBS_VAR,
) -> int:
    """Resolve the job-count denominator of the load gate from ``environ[var]``.

    Absent, blank, or unparseable values fall back to ``default``, and the
    result is clamped into ``[minimum, maximum]``, for the same reasons
    :func:`resolve_idle_process_count` does: a worker that refuses to start
    over a malformed number crash-loops the fleet on the next unattended
    restart, and a hostile value must not be able to uncap admission.

    ``minimum`` is 1 rather than 0 because a zero denominator is not a
    stricter setting, it is an undefined load: see :func:`active_job_load`.
    """
    raw = (environ.get(var) or "").strip()
    if not raw:
        count = default
    else:
        try:
            count = int(raw)
        except ValueError:
            count = default
    return max(minimum, min(count, maximum))


def active_job_load(active_jobs: int, max_concurrent_jobs: int) -> float:
    """The worker's load as the fraction of its job budget already in use.

    LiveKit's own documented alternative to the CPU-average default
    (``min(len(active_jobs) / 10, 1.0)``), which is deterministic: it counts
    what this worker was asked to do rather than what the box happens to be
    doing, so neither a job process warming up nor a neighbouring container
    can make this worker report itself full.

    The result is clamped into ``[0, 1]`` because that is the contract the
    library documents for ``load_fnc``, and because the value is also
    published to livekit-server in every status update -- an unclamped 1.4
    would be a load figure no reader of that field is expecting.

    A non-positive budget returns ``1.0`` (full). It cannot arise from
    :func:`resolve_max_concurrent_jobs`, which clamps to at least 1; it is
    answered here so that a caller passing one directly gets a refusal rather
    than a ``ZeroDivisionError`` inside the library's load task, which would
    leave the gate reading whatever the last successful sample was.
    """
    if max_concurrent_jobs <= 0:
        return 1.0
    return min(max(active_jobs, 0) / max_concurrent_jobs, 1.0)


#: THE PHONE HOST'S CALL CAPACITY: how many calls this host carries well at
#: once. The value is the release hook's PHONE_AGENT_CAPACITY constant, and its
#: comment says where the number comes from (6 on voice-lon1; opx-voice was 3).
#: One named per-host value that both phone roles read. The phone agents admit
#: exactly this many calls (``phone_admission``): a dispatch past it
#: is refused, and an inbound caller's ringing leg is ended with a busy. They
#: also keep at least this many warm processes, so an idle worker takes a
#: burst of that many dispatches. The outbound originator dials at most this
#: many calls at once. Absent, the agents admit what their old job gate did and
#: the originator dials one call at a time.
PHONE_AGENT_CAPACITY_VAR = "SKYSAY_PHONE_AGENT_CAPACITY"
PHONE_AGENT_CAPACITY_CEILING = 16


def resolve_phone_agent_capacity(environ: Mapping[str, str]) -> int | None:
    """``SKYSAY_PHONE_AGENT_CAPACITY``, or None when absent or unusable.

    Unusable is treated as absent, never as a refusal to start, for the same
    reason as the other resolvers here: each reader then falls back to its own
    safe default (the agents to their configured warm pool, the originator to
    one call at a time).
    """
    raw = (environ.get(PHONE_AGENT_CAPACITY_VAR) or "").strip()
    if not raw:
        return None
    try:
        capacity = int(raw)
    except ValueError:
        warn(f"{PHONE_AGENT_CAPACITY_VAR}={raw!r} is not a whole number; ignored", stacklevel=2)
        return None
    if capacity < 1 or capacity > PHONE_AGENT_CAPACITY_CEILING:
        warn(f"{PHONE_AGENT_CAPACITY_VAR}={capacity} is outside 1..{PHONE_AGENT_CAPACITY_CEILING}; ignored", stacklevel=2)
        return None
    return capacity


def phone_capacity(environ: Mapping[str, str]) -> int:
    """How many calls the phone agents admit: the ONE number both phone roles read.

    The declared ``SKYSAY_PHONE_AGENT_CAPACITY``; where none is declared, what
    the agents' old job gate admitted (``phone_agents_admit``), so an
    undeclared host keeps its ceiling. The agents admit exactly this many
    (``phone_admission``); the outbound originator dials at most the declared
    capacity (one at a time when none is declared).
    """
    return resolve_phone_agent_capacity(environ) or phone_agents_admit(environ)


def phone_agents_admit(environ: Mapping[str, str], threshold: float = 0.7) -> int:
    """What the phone agents' old job gate admitted once a call was active.

    livekit-agents 1.8.1 accepts while the jobs it counts, over the budget,
    stay under ``load_threshold`` (0.7 in production): the default budget of 5
    admits 4. Kept only as ``phone_capacity``'s default for a host that
    declares no capacity, so such a host keeps its old ceiling.
    """
    budget = resolve_max_concurrent_jobs(environ)
    admitted = 0
    while admitted < 64 and active_job_load(admitted, budget) < threshold:
        admitted += 1
    return admitted


def resolve_phone_agent_idle_processes(environ: Mapping[str, str]) -> int:
    """The phone agents' warm pool: never below the host's DECLARED capacity.

    With NO job active, livekit-agents 1.8.1 weighs each accepted-not-yet-
    launched dispatch at ``load_threshold / num_idle_processes`` (not at one
    job's share of the budget), so an idle worker takes at most
    ``num_idle_processes`` dispatches arriving together and refuses the next
    until one launches. An originator dialling ``capacity`` calls at once can
    deliver exactly that burst; with the default pool of 2 and a capacity of 3
    the third call would be answered into a room no agent took. Raising the
    pool to the capacity only lets the idle worker take MORE, never fewer, so
    the agents' gate is not lowered. Cost: one warm process per call of
    capacity above the configured pool (about 112-204 MB PSS each).
    """
    idle = resolve_idle_process_count(environ)
    capacity = resolve_phone_agent_capacity(environ)
    return idle if capacity is None else max(idle, capacity)


#: The memory a single job process may use before it is worth a log line, and
#: before it is killed.
#:
#: Both numbers are compared against the same figure livekit-agents samples:
#: ``_sample_memory_mb`` prefers **PSS**, falling back to USS and only then to
#: RSS, so on the Linux host these are PSS mebibytes. RSS for a forked job
#: process on this image is ~385 MB, nearly all of it pages shared with the
#: forkserver, so a threshold read as RSS would be off by that much.
#:
#: Measured in the worker image on Linux (forkserver, the production process
#: model): a warm job process that has imported the plugin set and loaded the
#: VAD is **204 MB PSS** on its own, and 112 MB PSS when three run together.
#:
#: A call adds its two captured 8 kHz tracks on top of that, about 0.9 MiB
#: per minute each: ~55 MiB for a 1800 s call (the default call ceiling).
#: Finalizing the evidence after hangup adds little on top of that.
#: ``SynchronizedEvidenceRecorder`` streams the listening mix in fixed-size
#: chunks over views of those tracks and copies neither of them. Measured on
#: the worker's Python (3.12, Linux, 3 GiB memory limit), an 1800 s call peaks
#: 27-32 MiB above its pre-finalize RSS, nearly all of it one transient
#: reallocation while the shorter track is padded to the longer one's length.
#: The mixer this replaced unpacked both tracks into Python ints and added
#: 1.3-1.7 GiB at that length, which is why these defaults were set with
#: finalization in mind.
#:
#: ``600`` warns. Finalization no longer comes near it, so a job that crosses
#: it is not explained by a long call ending; the library rate-limits the line.
#:
#: The kill limit still defaults to **disabled** (``0``, the library's own
#: default). It was disabled because finalization used to exceed any limit a
#: 3.8 GiB host could afford; choosing a non-zero default now is a separate
#: decision. An operator may still set one; it is then held above the warning
#: (see ``resolve_job_memory_thresholds``).
JOB_MEMORY_WARN_MB_VAR = "SKYSAY_WORKER_JOB_MEMORY_WARN_MB"
DEFAULT_JOB_MEMORY_WARN_MB = 600.0
JOB_MEMORY_LIMIT_MB_VAR = "SKYSAY_WORKER_JOB_MEMORY_LIMIT_MB"
DEFAULT_JOB_MEMORY_LIMIT_MB = 0.0


def resolve_job_memory_warn_mb(
    environ: Mapping[str, str],
    *,
    default: float = DEFAULT_JOB_MEMORY_WARN_MB,
    minimum: float = 1.0,
    var: str = JOB_MEMORY_WARN_MB_VAR,
) -> float:
    """Resolve the per-job memory warning threshold from ``environ[var]``.

    Absent, blank, unparseable, non-finite, or non-positive values fall back
    to ``default``; the result is floored at ``minimum`` so the threshold cannot
    be set to something every process trips on its first breath, which would
    bury the warning it exists to raise.
    """
    raw = (environ.get(var) or "").strip()
    if not raw:
        return default
    try:
        megabytes = float(raw)
    except ValueError:
        warn(f"{var}={raw!r} is not a number of megabytes; warning at {default} MB instead.")
        return default
    # ``float`` accepts "nan" and "inf", and NaN passes every comparison below.
    if not math.isfinite(megabytes) or megabytes <= 0:
        warn(f"{var}={megabytes} is not a usable memory threshold; warning at {default} MB instead.")
        return default
    return max(minimum, megabytes)


def resolve_job_memory_limit_mb(
    environ: Mapping[str, str],
    *,
    default: float = DEFAULT_JOB_MEMORY_LIMIT_MB,
    minimum: float = 1.0,
    var: str = JOB_MEMORY_LIMIT_MB_VAR,
) -> float:
    """Resolve the per-job memory kill threshold from ``environ[var]``.

    ``0`` is meaningful here and is NOT treated as a bad value: it is the
    library's own "no limit" setting, and an operator who has decided that no
    call should ever be killed for memory needs a way to say so without an
    image change. Everything else non-positive is a typo and falls back.

    Absent, blank, unparseable, or non-finite values fall back to ``default``,
    for the same reason every other resolver in this module does: a worker that
    refuses to start crash-loops the fleet on the next unattended restart.
    ``nan`` would otherwise floor to a 1 MB limit that kills every process, and
    ``inf`` would be a limit that reads as set and never fires.
    """
    raw = (environ.get(var) or "").strip()
    if not raw:
        return default
    try:
        megabytes = float(raw)
    except ValueError:
        warn(f"{var}={raw!r} is not a number of megabytes; {_limit_fallback(default)}")
        return default
    if megabytes == 0:
        return 0.0
    if not math.isfinite(megabytes) or megabytes < 0:
        warn(f"{var}={raw!r} is not a usable memory limit; {_limit_fallback(default)}")
        return default
    return max(minimum, megabytes)


def _limit_fallback(default: float) -> str:
    return "leaving the limit disabled." if default == 0 else f"limiting at {default} MB instead."


def resolve_job_memory_thresholds(environ: Mapping[str, str]) -> Tuple[float, float]:
    """Both memory thresholds together, with the relationship between them held.

    Resolved as a pair rather than by two independent calls because the pair
    has an invariant that neither half can check alone: the warning threshold
    must sit strictly below an enabled limit. The library checks the limit
    first on every sample, so a warning at or above it could never be logged
    at all -- the operator would be told nothing until a live call was gone.

    This orders the thresholds; it does not promise a warning ahead of every
    kill. Memory is sampled every few seconds, and a process that jumps from
    under the warning to over the limit between two samples is killed with no
    warning logged.

    So a warning at or above an enabled limit is pulled below it (to half the
    limit, which is a threshold rather than a cosmetic epsilon) and the
    substitution is warned about. A disabled limit (``0``, the default) imposes
    no relationship, and the warning stands wherever it was set.

    Returns ``(warn_mb, limit_mb)`` in the order ``AgentServer`` takes them.
    """
    warn_mb = resolve_job_memory_warn_mb(environ)
    limit_mb = resolve_job_memory_limit_mb(environ)
    if limit_mb > 0 and warn_mb >= limit_mb:
        # No floor here: the limit can be as low as 1 MB, and a floored
        # warning would land ON the limit and be unreachable again.
        adjusted = limit_mb / 2
        warn(
            f"{JOB_MEMORY_WARN_MB_VAR}={warn_mb} is not below "
            f"{JOB_MEMORY_LIMIT_MB_VAR}={limit_mb}, so the warning could never be "
            f"logged; warning at {adjusted} MB instead."
        )
        warn_mb = adjusted
    return warn_mb, limit_mb


#: The port livekit-agents binds its worker HTTP server to in production.
#:
#: The library's own production default, restated here rather than imported so
#: this module keeps its no-livekit-import promise. It is a health endpoint
#: only -- nothing on the voice host consumes it -- but it is a real bind, and
#: under ``network_mode: host`` every worker on the box shares one namespace:
#: whichever process starts second on a taken port dies on bind and silently
#: never registers. That is exactly why the browser demo (8082) and the
#: simulations-lab caller (8083) already carry ports of their own.
DEFAULT_WORKER_HTTP_PORT = 8081

#: The port variable the release hook injects into a second, transient copy of
#: a role during a blue/green roll.
WORKER_HTTP_PORT_VAR = "SKYSAY_VOICE_WORKER_PORT"

#: ``0`` asks the OS for a free port. It is the only value a second instance of
#: an already-running role can use without knowing what else the host holds.
WORKER_HTTP_PORT_OS_ASSIGNED = 0
_MAX_TCP_PORT = 65535


def resolve_worker_http_port(
    environ: Mapping[str, str],
    *,
    default: int = DEFAULT_WORKER_HTTP_PORT,
    var: str = WORKER_HTTP_PORT_VAR,
) -> int:
    """Resolve the worker's health-server port from ``environ[var]``.

    Absent (every deployment that exists today) returns ``default``, which is
    the library's own production default -- so a worker started without this
    variable binds exactly the port it always bound.

    ``0`` is the value that matters: it asks the OS for a free port, which is
    what lets a blue/green roll run a second copy of a role beside the one
    already holding 8081 on this host's shared network namespace. The release
    hook sets it only on the transient colour.

    Out-of-range or unparseable values warn and fall back rather than refusing
    to start, for the same reason :func:`resolve_drain_budget_seconds` does: a
    worker that will not start crash-loops the whole fleet on the next
    unattended restart, with no release running to notice and nothing to roll
    back.
    """
    raw = (environ.get(var) or "").strip()
    if not raw:
        return default
    try:
        port = int(raw)
    except ValueError:
        warn(f"{var}={raw!r} is not an integer port; binding {default} instead.")
        return default
    if not (WORKER_HTTP_PORT_OS_ASSIGNED <= port <= _MAX_TCP_PORT):
        warn(f"{var}={port} is not a usable TCP port; binding {default} instead.")
        return default
    return port


#: How long a worker may take to finish the calls it is already running.
#:
#: Set from the longest call a default deployment can place --
#: ``SKYSAY_OUTBOUND_MAX_CALL_DURATION_SECONDS`` defaults to 1800 -- and
#: NOT from the ten-minute wallet reserve. The reserve is a billing hold that
#: settles against real duration; a call is free to outlive it, and a drain
#: budget derived from it would cut every call between ten and thirty minutes
#: while looking like it had been reasoned about.
#:
#: The platform's own ceiling for that setting is 14400 (four hours), which
#: this deliberately does NOT default to: the same number is also how long a
#: promote can sit waiting for one stubborn session, and a release that may
#: hang for four hours is its own outage.
#:
#: This is only the FALLBACK. A worker the release hook started reads its own
#: stop deadline from the environment (see below) and derives the budget from
#: that, so the deployed number lives in exactly one place. This constant is
#: what a process started without the hook -- a developer's compose, a test --
#: drains for.
DEFAULT_DRAIN_TIMEOUT_SECONDS = 1800

#: The variable the release hook injects into every role that drains sessions,
#: carrying the exact ``stop_grace_period`` it rendered for that role.
STOP_GRACE_VAR = "SKYSAY_VOICE_STOP_GRACE_SECONDS"

#: How much of the container runtime's stop deadline the worker leaves itself
#: to finish shutting down after its drain returns. The drain is not the last
#: thing that happens: the process still has to close its sessions, flush and
#: exit, and a budget that ran right up to the deadline would be killed doing
#: exactly that.
DRAIN_GRACE_MARGIN_SECONDS = 60


def resolve_drain_budget_seconds(
    environ: Mapping[str, str],
    *,
    default: int = DEFAULT_DRAIN_TIMEOUT_SECONDS,
    minimum: int = 1,
    margin: int = DRAIN_GRACE_MARGIN_SECONDS,
    var: str = STOP_GRACE_VAR,
) -> int:
    """The longest drain the thing that sent SIGTERM will actually wait out.

    Derived from the deadline itself rather than agreed with it. The release
    hook renders one integer per role class and uses it twice -- as the
    service's ``stop_grace_period`` and as this variable -- so raising the
    grace raises the budget in the same edit, and no constant in this image
    has to be remembered alongside it.

    Absent (a developer's compose, a unit test) falls back to ``default``.
    Unparseable or non-positive warns and falls back: a worker that refuses to
    start over a malformed number would crash-loop the whole fleet on the next
    unattended restart, with no release running to notice and nothing to roll
    back -- the failure mode ``browser_demo_agent.py`` already cost this
    platform 13 releases.
    """
    raw = (environ.get(var) or "").strip()
    if not raw:
        return default
    try:
        grace = int(raw)
    except ValueError:
        warn(f"{var}={raw!r} is not an integer number of seconds; draining for {default}s.")
        return default
    if grace <= 0:
        warn(f"{var}={grace} is not a usable stop deadline; draining for {default}s.")
        return default
    return max(minimum, grace - margin)


#: The call-duration ceiling a deployment configures for itself, and the
#: platform's own maximum for it. Read here ONLY to compare against the stop
#: deadline: the two have different owners (this is the host's `carrier.env`,
#: the deadline is the release hook's constant), which is exactly why the
#: relationship is reported rather than derived or enforced.
MAX_CALL_DURATION_VAR = "SKYSAY_OUTBOUND_MAX_CALL_DURATION_SECONDS"
DEFAULT_MAX_CALL_DURATION_SECONDS = 1800


def _warn_if_calls_outlive_the_budget(environ: Mapping[str, str], budget: int) -> None:
    """Say so when this deployment allows calls longer than it can drain.

    The validator in ``outbound_worker`` accepts a ceiling up to 14400, and
    the release hook's stop grace is a fixed constant, so a deployment can
    legitimately configure four-hour calls and still have its workers replaced
    31 minutes into one. Nothing in the image can fix that -- lifting the
    deadline is the hook's decision and the host's -- so the honest move is to
    name both numbers at startup, where an operator sees them.

    Warned, never refused, for the same reason the malformed-deadline path is:
    a worker that will not start crash-loops the fleet on the next unattended
    restart, with no release running to notice and nothing to roll back.

    Compare against what the worker will ACTUALLY allow, which for an unset
    variable is not zero: ``OutboundWorkerSettings.from_environ`` reads
    ``int(environ.get(MAX_CALL_DURATION_VAR) or "1800")``, so the deployment
    that configures nothing is the one that permits half-hour calls -- and it
    is the common one. Reading a blank value as "no ceiling to compare" would
    silence this warning exactly where it is most needed: shorten the stop
    grace under the default and every ordinary call becomes cuttable with
    nothing said.

    An UNPARSEABLE or out-of-range value is different and stays silent, and
    the asymmetry is deliberate. ``from_environ`` raises on it, so the
    outbound ORIGINATOR refuses to start -- and the ceiling this compares
    against is the one the originator stamps onto the call it places, so with
    it down there is no call carrying that ceiling and no number to compare.
    The session roles reading this function do start, because they never parse
    the variable; what they lose is the originator's calls, not a warning they
    could act on. The startup refusal is the message there.
    """
    raw = (environ.get(MAX_CALL_DURATION_VAR) or "").strip()
    if not raw:
        ceiling, configured = DEFAULT_MAX_CALL_DURATION_SECONDS, False
    else:
        try:
            ceiling, configured = int(raw), True
        except ValueError:
            return
    if ceiling <= budget:
        return
    source = (
        f"{MAX_CALL_DURATION_VAR}={ceiling}"
        if configured
        else f"{MAX_CALL_DURATION_VAR} is unset, so its {ceiling}s default"
    )
    warn(
        f"{source} allows calls longer than the {budget}s this worker can "
        f"drain for, so a promote landing on one still cuts it. Raise SESSION_ROLE_STOP_GRACE_SECONDS "
        f"in infra/voice-host/skysay-voice-release to at least {ceiling + DRAIN_GRACE_MARGIN_SECONDS}, "
        f"or accept that calls past {budget}s are cut by a release."
    )


def resolve_drain_timeout_seconds(
    environ: Mapping[str, str],
    *,
    default: int | None = None,
    minimum: int = 1,
    # Never above what the container runtime will wait: a LONGER drain cannot
    # run to completion -- it restores exactly the mid-drain SIGKILL this
    # exists to remove, while looking like a more generous setting. Shortening
    # it is a real choice this side can honour alone. Computed from the
    # deadline the hook injected, so raising the grace raises this with it.
    maximum: int | None = None,
    var: str = "SKYSAY_VOICE_DRAIN_TIMEOUT_SECONDS",
) -> int:
    """How long a worker may spend finishing its sessions after SIGTERM.

    livekit-agents already does the right thing on SIGTERM: it reports itself
    full so the server stops dispatching to it, then waits for the jobs it is
    already running.  What it waits for is ``drain_timeout``, whose library
    default is an hour -- far past any deadline the container runtime gives
    it, so the process is killed mid-drain and the call is cut anyway.  The
    number that matters is therefore not "how long could a session possibly
    last" but "how long may this process take before the thing that sent the
    signal gives up on it", and it has to be SHORTER than that deadline. So it
    is derived from it: :func:`resolve_drain_budget_seconds` reads the grace
    the release hook rendered for this role and leaves a margin inside it, and
    an explicit setting here may only shorten what that allows.

    A session that outlives this budget is still cut. That is a real limit,
    not an oversight: the quiet gate exists so a promote does not land on one,
    and the alternative -- a stop deadline as long as the platform's four-hour
    call ceiling -- makes a single stuck session able to hang a release for an
    afternoon.

    Absent, blank, or unparseable values fall back to ``default``; the result
    is clamped so neither a typo nor a hostile value can make a stop
    instantaneous (cutting a live call) nor push it past the deadline the
    container runtime will actually wait out.
    """
    budget = resolve_drain_budget_seconds(environ) if maximum is None else maximum
    if default is None:
        default = budget
    _warn_if_calls_outlive_the_budget(environ, budget)
    raw = (environ.get(var) or "").strip()
    if not raw:
        seconds = default
    else:
        try:
            seconds = int(raw)
        except ValueError:
            seconds = default
    resolved = max(minimum, min(seconds, budget))
    if seconds > budget:
        # Say so, and name the deadline rather than a constant to edit. This
        # is now the only way the two can disagree: someone asked for a drain
        # longer than the stop grace their own release hook renders. Warned
        # rather than refused, because a worker that will not start is worse
        # than one that drains for the budget the runtime will actually wait.
        warn(
            f"{var}={seconds} exceeds the {budget}s available inside this container's stop deadline "
            f"({STOP_GRACE_VAR}); draining for {resolved}s instead. Raise "
            f"SESSION_ROLE_STOP_GRACE_SECONDS in infra/voice-host/skysay-voice-release to lift both."
        )
    return resolved


#: The provider modules a job process imports at warm-up, before any call.
#:
#: Names only, so this module keeps its no-livekit-import promise; the
#: worker's ``setup_process`` hands the tuple to :func:`prewarm_plugin_modules`.
#:
#: The SUPERVISOR imports the same tuple before it starts the library, and
#: that second call site is the one that makes this list nearly free. Every
#: ``livekit.plugins.*`` package registers itself on import
#: (``Plugin.register_plugin``), and livekit-agents builds its forkserver
#: preload list from exactly that registry -- ``[p.package for p in
#: Plugin.registered_plugins]`` plus a fixed tail (``worker.py`` in 1.8.1). A
#: supervisor that has imported nothing therefore preloads no plugin at all,
#: and each forked child pays the whole import set itself. With the supervisor
#: importing first, the forkserver holds the modules and children inherit them
#: copy-on-write; the ``setup_process`` call then finds them already in
#: ``sys.modules`` and costs nothing. Measured on a laptop, one child's warm-up
#: went from 814 ms / 0.72 CPU-s / 147 MB RSS to 67 ms / 0.05 CPU-s / 63 MB,
#: reaching the identical 2831 loaded modules either way.
#:
#: The two OpenAI entries were added on 2026-09-15: every GPT-Live job that
#: day blocked its event loop for 387-855 ms at job start importing
#: ``openai.types.realtime.*``, because the worker imports the GPT-Live
#: plugin lazily, inside the job, and that plugin's realtime model module
#: imports the whole ``openai.types.realtime`` package (156 submodules) at
#: import time. Warming the plugin here moves that cost to process warm-up.
#: The package is listed as well because its own ``__init__`` imports all
#: 156 submodules (measured: 699 ms on its own, with no livekit module
#: loaded), so the types stay warm even if a future plugin version stops
#: importing them eagerly.
#: The Gemini Live entries were added on 2026-09-24 after a homepage load test
#: of six concurrent Gemini Live sessions: each job blocked its event loop for
#: ~5.3 s importing ``google.genai.types`` on first use of the realtime model,
#: plus 0.5-0.8 s for ``anyio``, which serialised session starts (first-audio
#: p95 9.6 s). The same imports sit on a Gemini Live phone call's first turn.
#: The browser demo worker warms this same list (``browser_demo_agent``).
#:
#: ``anyio._backends._asyncio`` was added after the next burst: top-level
#: ``anyio`` (4.15.1) loads its submodules lazily, so each job start still
#: blocked 0.66-0.97 s apiece on ``anyio.abc._sockets``, ``anyio._core.
#: _sockets``, ``anyio.abc._eventloop`` and ``anyio._backends._asyncio`` --
#: the last one is imported by anyio's backend loader on first use. Importing
#: that one module loads all four (checked in the worker's own requirements
#: environment; ``anyio`` and ``anyio.abc`` alone load none of them), so it is
#: the smallest entry that covers the stall. The image build asserts it.
#: The homepage-demo prewarm tests pin the four names.
PREWARM_PLUGIN_MODULES: Tuple[str, ...] = (
    "livekit.plugins.cartesia",
    "livekit.plugins.deepgram",
    "livekit.plugins.google",
    "livekit.plugins.elevenlabs",
    "livekit.plugins.soniox",
    "livekit.plugins.fishaudio",
    "livekit.plugins.silero",
    "livekit.plugins.openai.realtime",
    "openai.types.realtime",
    "livekit.plugins.google.realtime",
    "google.genai.types",
    "anyio",
    "anyio._backends._asyncio",
)


def prewarm_plugin_modules(
    module_names: Sequence[str], importer: Callable[[str], object] = __import__
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Best-effort import each provider plugin module.

    Returns ``(warmed, failed)`` where ``warmed`` lists the modules that
    imported cleanly and ``failed`` lists ``(module_name, error)`` pairs.  A
    failing import is recorded (never swallowed silently) but never raises, so a
    broken/absent optional plugin cannot crash the warm process — while the
    failure still surfaces to the caller for logging.
    """
    warmed: List[str] = []
    failed: List[Tuple[str, str]] = []
    for name in module_names:
        try:
            importer(name)
        except Exception as exc:  # noqa: BLE001 - best-effort warm: report, don't raise
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        warmed.append(name)
    return warmed, failed


JOB_MEMORY_WARN_VAR = "SKYSAY_VOICE_JOB_MEMORY_WARN_MB"
JOB_MEMORY_LIMIT_VAR = "SKYSAY_VOICE_JOB_MEMORY_LIMIT_MB"


def job_memory_options(environ: Mapping[str, str]) -> dict[str, float]:
    """The per-job memory watchdog, only when a host deliberately configures it.

    Pinned livekit-agents 1.8.1 samples each job process's PSS (falling back
    to USS, then RSS) in MiB, logs above ``job_memory_warn_mb`` and kills the
    JOB process above ``job_memory_limit_mb``; zero disables the kill. Unset
    here leaves the SDK default untouched, so a host that does not opt in
    behaves exactly as before. A malformed or inverted pair is refused at
    startup rather than silently disabling the limit a host asked for.
    """
    warn_raw = (environ.get(JOB_MEMORY_WARN_VAR) or "").strip()
    limit_raw = (environ.get(JOB_MEMORY_LIMIT_VAR) or "").strip()
    if not warn_raw and not limit_raw:
        return {}
    try:
        warn, limit = int(warn_raw), int(limit_raw)
    except ValueError as exc:
        raise ValueError("job memory watchdog needs integer MiB for both warn and limit") from exc
    if not 64 <= warn < limit <= 4096:
        raise ValueError("job memory watchdog needs 64 <= warn < limit <= 4096 MiB")
    return {"job_memory_warn_mb": float(warn), "job_memory_limit_mb": float(limit)}
