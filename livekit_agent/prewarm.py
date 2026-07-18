# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Agent prewarming configuration.

A production call showed invite->accept taking ~7s because LiveKit cold-starts
a fresh job process per call.  Keeping warm idle job processes removes that
cold start; this module resolves how many to keep and best-effort pre-imports
the provider plugin modules so a warm process does not pay the import cost on
its first job.

Pure module: no livekit import, no network, no side effects.
"""

from __future__ import annotations

from typing import Callable, List, Mapping, Sequence, Tuple


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
