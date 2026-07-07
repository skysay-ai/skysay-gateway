from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openphonex_common.config import AsteriskARIConfig, load_env_file


class ConfigTests(unittest.TestCase):
    def test_load_env_file_supports_exports_and_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".env"
            path.write_text("export A=1\nB='two'\nC=\"three\"\n# ignored\n", encoding="utf-8")
            self.assertEqual(load_env_file(path), {"A": "1", "B": "two", "C": "three"})

    def test_asterisk_config_env_override_wins(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "ASTERISK_ARI_USER=file-user",
                        "ASTERISK_ARI_PASSWORD=file-pass",
                        "DIDWW_CALLER_ID=+15550000000",
                        "DIDWW_ASTERISK_OUTBOUND_ENDPOINT=didww-outbound",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"ASTERISK_ARI_PASSWORD": "env-pass"}, clear=True):
                config = AsteriskARIConfig.from_env(path)
        self.assertEqual(config.ari_user, "file-user")
        self.assertEqual(config.ari_password, "env-pass")
        self.assertEqual(config.outbound_endpoint, "didww-outbound")

    def test_asterisk_config_requires_live_call_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".env"
            path.write_text("", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    AsteriskARIConfig.from_env(path)
