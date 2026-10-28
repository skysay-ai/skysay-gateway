# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Join all Azure stream waiters in the pinned SDK at image build time.

The upstream cancellation finally joins input/reconnect tasks but leaves its
session-stop waiter alive when the vendor does not emit that event. Apply one
exact source change; an SDK upgrade must re-prove the lifecycle and update
these pins. There is no runtime monkey patch or alternate execution path.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import sys
from email.parser import Parser
from pathlib import Path


SOURCE_SHA256 = "047d4a43ee6badaf7139b382af076fde488485f187ab99c6f24f2c4db42ffa0d"
OLD = b"await utils.aio.gracefully_cancel(process_input_task, wait_reconnect_task)"
NEW = b"await utils.aio.gracefully_cancel(process_input_task, wait_reconnect_task, wait_stopped_task)"


def joined_cleanup_source(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Azure STT cleanup fix requires its exact reviewed SDK source")
    if source.count(OLD) != 1:
        raise RuntimeError("Azure STT cleanup fix no longer has one matching lifecycle boundary")
    corrected = source.replace(OLD, NEW)
    ast.parse(corrected)
    return corrected


def installed_source() -> Path:
    """Resolve one installed SDK without importing provider code."""
    try:
        roots = {Path(entry or ".").resolve() for entry in sys.path if Path(entry or ".").is_dir()}
        distributions = [(root, item) for root in roots
                         for item in root.glob("livekit_plugins_azure-*.dist-info")]
        relative = "livekit/plugins/azure/stt.py"
        sources = [(root, root / relative) for root in roots if (root / relative).exists()]
        if len(distributions) != 1 or len(sources) != 1 or distributions[0][0] != sources[0][0]:
            raise RuntimeError("Azure STT cleanup fix cannot identify its installed source")
        metadata = Parser().parsestr((distributions[0][1] / "METADATA").read_text(encoding="utf-8"))
        if (metadata.get_all("Name") != ["livekit-plugins-azure"]
                or metadata.get_all("Version") != ["1.8.8"]):
            raise RuntimeError("Azure STT cleanup fix requires the pinned plugin version")
        with (distributions[0][1] / "RECORD").open(encoding="utf-8", newline="") as record:
            entries = [row for row in csv.reader(record) if row and row[0] == relative]
        path = sources[0][1]
        if len(entries) != 1 or not path.is_file():
            raise RuntimeError("Azure STT cleanup fix cannot identify its installed source")
        return path
    except (OSError, UnicodeError, csv.Error):
        raise RuntimeError("Azure STT cleanup fix cannot read its installed metadata") from None


def main() -> None:
    path = installed_source()
    corrected = joined_cleanup_source(path.read_bytes())
    path.write_bytes(corrected)
    print("Azure STT joined-cleanup source sha256=" + hashlib.sha256(corrected).hexdigest())


if __name__ == "__main__":
    main()
