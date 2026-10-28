# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import os
from pathlib import Path


DEFAULT_ENV_PATHS = (
    Path("/etc/skysay/didww.env"),
    Path("/etc/skysay/.env"),
)


def configured_env_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for name in ("SKYSAY_DIDWW_ENV_FILE", "SKYSAY_ENV_FILE"):
        raw = os.environ.get(name, "").strip()
        if raw:
            paths.append(Path(raw).expanduser())
    for raw in os.environ.get("SKYSAY_ENV_FILES", "").split(os.pathsep):
        if raw.strip():
            paths.append(Path(raw.strip()).expanduser())
    paths.extend(DEFAULT_ENV_PATHS)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def merged_env(path: Path | None = None) -> dict[str, str]:
    values: dict[str, str] = {}
    paths = [path] if path else list(configured_env_paths())
    for candidate in paths:
        if candidate:
            values.update(load_env_file(candidate))
    values.update({k: v for k, v in os.environ.items() if isinstance(v, str)})
    return values
