"""Tests for the standalone ``openphonex_gateway.originate`` CLI (SCH-202 M4)."""
from __future__ import annotations

import ast
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from openphonex_gateway import originate


class OriginateDryRunTests(unittest.TestCase):
    def _write_env(self, directory: Path) -> Path:
        env_path = directory / "asterisk.env"
        env_path.write_text(
            "\n".join(
                [
                    "ASTERISK_ARI_USER=user",
                    "ASTERISK_ARI_PASSWORD=super-secret",
                    "DIDWW_CALLER_ID=+15550000000",
                    "DIDWW_OUTBOUND_HOST=voip.example.net",
                ]
            ),
            encoding="utf-8",
        )
        return env_path

    def test_dry_run_emits_outbound_payload(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            env_path = self._write_env(Path(raw))
            buffer = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                rc = originate.main(["+1 (555) 987-6543", "--env-file", str(env_path), "--dry-run"])
            self.assertEqual(rc, 0)
            payload = json.loads(buffer.getvalue())
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["app_args"], "outbound")
            self.assertEqual(payload["caller_id"], "+15550000000")
            self.assertEqual(payload["endpoint_profile"], "didww-outbound")
            self.assertEqual(payload["target_digits"], "15559876543")
            self.assertEqual(payload["endpoint"], "PJSIP/didww-outbound/sip:15559876543@voip.example.net")

    def test_dry_run_never_leaks_secret_or_touches_network(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            env_path = self._write_env(Path(raw))
            buffer = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                with patch("urllib.request.urlopen", side_effect=AssertionError("network call in --dry-run")):
                    rc = originate.main(["+37253231038", "--env-file", str(env_path), "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertNotIn("super-secret", buffer.getvalue())


class OriginateImportBoundaryTests(unittest.TestCase):
    def test_originate_imports_only_openphonex_common_and_stdlib(self) -> None:
        source = Path(originate.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    roots.add(node.module.split(".")[0])
        self.assertNotIn("telephony_ai", roots)
        allowed = {"openphonex_common"} | set(sys.stdlib_module_names)
        offenders = {root for root in roots if root not in allowed}
        self.assertEqual(offenders, set(), f"unexpected imports outside stdlib/openphonex_common: {offenders}")


if __name__ == "__main__":
    unittest.main()
