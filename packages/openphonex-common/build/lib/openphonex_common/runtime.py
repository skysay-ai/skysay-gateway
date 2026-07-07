from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__


REDACTED = "[REDACTED]"
SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "credential",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "session",
    "token",
)
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def env_flag(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return text in TRUE_ENV_VALUES


def env_enabled(name: str, env: dict[str, str] | None = None, *, default: bool = False) -> bool:
    source = env if env is not None else os.environ
    return env_flag(source.get(name), default=default)


def deployment_environment() -> str:
    value = (
        os.environ.get("OPENPHONEX_ENV")
        or os.environ.get("OPENPHONEX_ENVIRONMENT")
        or os.environ.get("APP_ENV")
        or "local"
    )
    return value.strip().lower() or "local"


def service_version() -> str:
    return os.environ.get("OPENPHONEX_VERSION", __version__)


def is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in SECRET_KEY_PARTS)


def redact_secrets(value: Any, *, _depth: int = 0) -> Any:
    if _depth > 12:
        return "[MAX_DEPTH]"
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            redacted[key_str] = REDACTED if is_sensitive_key(key_str) else redact_secrets(item, _depth=_depth + 1)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item, _depth=_depth + 1) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item, _depth=_depth + 1) for item in value]
    if isinstance(value, str):
        lower = value.lower().strip()
        if lower.startswith(("bearer ", "basic ", "api-key:", "api-key ")):
            return REDACTED
        if value.startswith("tai_") and len(value) > 16:
            return value[:8] + "..." + REDACTED
    return value


def structured_log(event: str, *, service: str = "telephony-ai", level: str = "info", force: bool = False, **fields: Any) -> dict[str, Any]:
    payload = redact_secrets(
        {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "level": level,
            "service": service,
            "environment": deployment_environment(),
            "version": service_version(),
            "event": event,
            **fields,
        }
    )
    if force or env_enabled("OPENPHONEX_STRUCTURED_LOGS"):
        print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def path_readiness(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path)
    out: dict[str, Any] = {
        "ok": False,
        "label": label,
        "path": str(resolved),
        "exists": resolved.exists(),
        "writable": False,
    }
    try:
        resolved.mkdir(parents=True, exist_ok=True)
        probe = resolved / f".telephony-ai-health-{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        out["writable"] = probe.read_text(encoding="utf-8") == "ok"
        probe.unlink(missing_ok=True)
        out["exists"] = resolved.exists()
        out["ok"] = bool(out["exists"] and out["writable"])
    except Exception as exc:
        out["error"] = exc.__class__.__name__
        out["message"] = str(exc)
    return out


def local_temp_readiness(*, label: str = "temp") -> dict[str, Any]:
    return path_readiness(Path(tempfile.gettempdir()) / "telephony-ai-health", label=label)
