# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ENV_PATHS = (
    Path("/etc/openphonex/asterisk.env"),
    Path("/etc/openphonex/didww.env"),
    Path("/etc/openphonex/.env"),
)


def configured_env_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for name in ("OPENPHONEX_ASTERISK_ENV_FILE", "OPENPHONEX_DIDWW_ENV_FILE", "OPENPHONEX_ENV_FILE"):
        raw = os.environ.get(name, "").strip()
        if raw:
            paths.append(Path(raw).expanduser())
    for raw in os.environ.get("OPENPHONEX_ENV_FILES", "").split(os.pathsep):
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


@dataclass(frozen=True)
class AsteriskARIConfig:
    ari_url: str
    ari_user: str
    ari_password: str
    outbound_endpoint: str
    caller_id: str
    app: str = "voicebot"
    app_args: str = "outbound"
    timeout_seconds: int = 45

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "AsteriskARIConfig":
        env = merged_env(env_path)
        missing = [key for key in ("ASTERISK_ARI_USER", "ASTERISK_ARI_PASSWORD", "DIDWW_CALLER_ID") if not env.get(key)]
        if missing:
            raise ValueError(f"missing Asterisk/DIDWW config: {', '.join(missing)}")
        return cls(
            ari_url=env.get("ASTERISK_ARI_URL", "http://127.0.0.1:8088/ari"),
            ari_user=env["ASTERISK_ARI_USER"],
            ari_password=env["ASTERISK_ARI_PASSWORD"],
            outbound_endpoint=env.get("DIDWW_ASTERISK_OUTBOUND_ENDPOINT", "didww-outbound"),
            caller_id=env["DIDWW_CALLER_ID"],
            app=env.get("ASTERISK_ARI_APP", "voicebot"),
            app_args=env.get("ASTERISK_ARI_APP_ARGS", "outbound"),
            timeout_seconds=int(env.get("ASTERISK_ARI_TIMEOUT", "45")),
        )

    def public_status(self) -> dict[str, str | int]:
        return {
            "ari_url": self.ari_url,
            "outbound_endpoint": self.outbound_endpoint,
            "app": self.app,
            "app_args": self.app_args,
            "timeout_seconds": self.timeout_seconds,
        }
