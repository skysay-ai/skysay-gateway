from __future__ import annotations

import unittest

from openphonex_common.runtime import env_enabled, env_flag


class RuntimeTests(unittest.TestCase):
    def test_env_flag_uses_one_truth_set(self) -> None:
        for value in ("1", "true", "TRUE", "yes", "on", " On "):
            with self.subTest(value=value):
                self.assertTrue(env_flag(value))
        for value in ("", "0", "false", "no", "required", None):
            with self.subTest(value=value):
                self.assertFalse(env_flag(value))

    def test_env_enabled_reads_supplied_env(self) -> None:
        env = {"OPENPHONEX_HTTP_LOGS": "yes", "OPENPHONEX_LIVE_SMS_ENABLED": "on"}
        self.assertTrue(env_enabled("OPENPHONEX_HTTP_LOGS", env))
        self.assertTrue(env_enabled("OPENPHONEX_LIVE_SMS_ENABLED", env))
        self.assertFalse(env_enabled("MISSING_FLAG", env))


if __name__ == "__main__":
    unittest.main()
