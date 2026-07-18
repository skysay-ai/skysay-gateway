# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Tiny stdlib HTTP helpers shared by the provider-edge adapters.

Just enough to POST a multipart/form-data body (a WAV file plus a couple of
text fields) with :mod:`urllib` — the OpenAI and ElevenLabs-Scribe batch STT
paths both upload buffered turn audio this way, and neither needs an SDK. The
``opener`` seam lets tests substitute a fake/`ThreadingHTTPServer` transport so
no live provider call is ever made.
"""

from __future__ import annotations

import urllib.request
import uuid
from typing import Callable

# An opener matches ``urllib.request.urlopen(request, timeout=...)``: it takes a
# Request and returns a context-manager response exposing ``.read()``.
Opener = Callable[..., object]


def default_opener(request: "urllib.request.Request", *, timeout: float | None = None):
    return urllib.request.urlopen(request, timeout=timeout)


def encode_multipart(
    fields: "dict[str, str]",
    *,
    file_field: str,
    file_name: str,
    file_bytes: bytes,
    file_content_type: str = "audio/wav",
) -> "tuple[bytes, str]":
    """Return ``(body, content_type_header)`` for a multipart/form-data POST."""

    boundary = f"----openphonex{uuid.uuid4().hex}"
    crlf = b"\r\n"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(b"--" + boundary.encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(str(value).encode("utf-8"))
    parts.append(b"--" + boundary.encode())
    parts.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{file_name}"'.encode()
    )
    parts.append(f"Content-Type: {file_content_type}".encode())
    parts.append(b"")
    parts.append(file_bytes)
    parts.append(b"--" + boundary.encode() + b"--")
    parts.append(b"")
    body = crlf.join(parts)
    return body, f"multipart/form-data; boundary={boundary}"
