from __future__ import annotations

import unittest

from livekit_agent.prewarm import (
    prewarm_plugin_modules,
    resolve_idle_process_count,
)


class ResolveIdleProcessCountTest(unittest.TestCase):
    def test_absent_variable_returns_default(self) -> None:
        self.assertEqual(resolve_idle_process_count({}), 2)

    def test_parses_positive_integer(self) -> None:
        self.assertEqual(resolve_idle_process_count({"SCH236_NUM_IDLE_PROCESSES": "3"}), 3)

    def test_one_is_accepted(self) -> None:
        self.assertEqual(resolve_idle_process_count({"SCH236_NUM_IDLE_PROCESSES": "1"}), 1)

    def test_zero_clamps_to_one(self) -> None:
        self.assertEqual(resolve_idle_process_count({"SCH236_NUM_IDLE_PROCESSES": "0"}), 1)

    def test_negative_clamps_to_one(self) -> None:
        self.assertEqual(resolve_idle_process_count({"SCH236_NUM_IDLE_PROCESSES": "-2"}), 1)

    def test_surrounding_whitespace_is_accepted(self) -> None:
        self.assertEqual(resolve_idle_process_count({"SCH236_NUM_IDLE_PROCESSES": "  4 "}), 4)

    def test_unparseable_value_returns_default(self) -> None:
        self.assertEqual(resolve_idle_process_count({"SCH236_NUM_IDLE_PROCESSES": "abc"}), 2)

    def test_empty_value_returns_default(self) -> None:
        self.assertEqual(resolve_idle_process_count({"SCH236_NUM_IDLE_PROCESSES": ""}), 2)

    def test_custom_default_is_used(self) -> None:
        self.assertEqual(resolve_idle_process_count({}, default=5), 5)
        self.assertEqual(
            resolve_idle_process_count({"SCH236_NUM_IDLE_PROCESSES": "nope"}, default=7), 7
        )

    def test_large_value_clamps_to_maximum(self) -> None:
        # Guards against OOM from a typo / hostile env value.
        self.assertEqual(resolve_idle_process_count({"SCH236_NUM_IDLE_PROCESSES": "500"}), 16)
        self.assertEqual(
            resolve_idle_process_count({"SCH236_NUM_IDLE_PROCESSES": "1000000"}), 16
        )

    def test_custom_maximum_is_honored(self) -> None:
        self.assertEqual(
            resolve_idle_process_count({"SCH236_NUM_IDLE_PROCESSES": "50"}, maximum=8), 8
        )


class PrewarmPluginModulesTest(unittest.TestCase):
    def test_all_imports_succeed(self) -> None:
        seen: list[str] = []
        warmed, failed = prewarm_plugin_modules(("a", "b", "c"), importer=seen.append)
        self.assertEqual(warmed, ["a", "b", "c"])
        self.assertEqual(failed, [])
        self.assertEqual(seen, ["a", "b", "c"])

    def test_failing_import_is_reported_not_raised(self) -> None:
        def importer(name: str) -> None:
            if name == "b":
                raise ImportError("boom")

        warmed, failed = prewarm_plugin_modules(("a", "b", "c"), importer=importer)
        self.assertEqual(warmed, ["a", "c"])
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0][0], "b")
        self.assertIn("ImportError", failed[0][1])

    def test_all_failing_never_raises(self) -> None:
        def importer(name: str) -> None:
            raise RuntimeError("nope")

        warmed, failed = prewarm_plugin_modules(("a", "b"), importer=importer)
        self.assertEqual(warmed, [])
        self.assertEqual([name for name, _ in failed], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
