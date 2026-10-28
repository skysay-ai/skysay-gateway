from __future__ import annotations

import unittest
from pathlib import Path

from livekit_agent import prewarm
from livekit_agent.prewarm import (
    PREWARM_PLUGIN_MODULES,
    SILERO_VAD_PINNED_PATH,
    SILERO_VAD_PINNED_SHA256,
    pinned_silero_vad_loader,
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


class PrewarmModuleListTest(unittest.TestCase):
    """The warm-up list is a named constant and includes the GPT-Live imports.

    On 2026-09-15 every GPT-Live job blocked its event loop for 387-855 ms
    at job start importing ``openai.types.realtime.*``: the worker imports
    the GPT-Live plugin lazily inside the job, and that plugin imports the
    whole realtime types package at import time. Warming both at process
    setup moves that cost off the call's event loop.
    """

    def test_list_keeps_the_provider_plugins_and_adds_the_realtime_imports(self) -> None:
        self.assertIsInstance(PREWARM_PLUGIN_MODULES, tuple)
        for name in (
            "livekit.plugins.cartesia",
            "livekit.plugins.deepgram",
            "livekit.plugins.google",
            "livekit.plugins.elevenlabs",
            "livekit.plugins.soniox",
            "livekit.plugins.fishaudio",
            "livekit.plugins.silero",
        ):
            self.assertIn(name, PREWARM_PLUGIN_MODULES)
        self.assertIn("livekit.plugins.openai.realtime", PREWARM_PLUGIN_MODULES)
        self.assertIn("openai.types.realtime", PREWARM_PLUGIN_MODULES)
        self.assertEqual(len(PREWARM_PLUGIN_MODULES), len(set(PREWARM_PLUGIN_MODULES)))

    def test_warming_the_list_imports_each_module_once_and_is_repeatable(self) -> None:
        seen: list[str] = []
        first = prewarm_plugin_modules(PREWARM_PLUGIN_MODULES, importer=seen.append)
        second = prewarm_plugin_modules(PREWARM_PLUGIN_MODULES, importer=seen.append)
        self.assertEqual(first, (list(PREWARM_PLUGIN_MODULES), []))
        self.assertEqual(second, first)
        # Each warm-up pass asks the importer once per name, in list order;
        # the interpreter's module cache is what makes the second pass free.
        self.assertEqual(seen, list(PREWARM_PLUGIN_MODULES) * 2)

    def test_real_import_of_an_absent_plugin_is_reported_never_raised(self) -> None:
        # The suite deliberately installs neither livekit plugins nor the
        # openai SDK, so this runs the real importer against the real list
        # and proves a missing module surfaces in ``failed`` rather than
        # crashing the warm process.
        warmed, failed = prewarm_plugin_modules(PREWARM_PLUGIN_MODULES)
        names = {name for name, _ in failed}
        self.assertEqual(set(warmed) | names, set(PREWARM_PLUGIN_MODULES))
        for name, error in failed:
            self.assertTrue(error.startswith("ModuleNotFoundError"), (name, error))

    # The pin that the worker's setup hands this constant to the importer
    # lives in the image-packaging test, which already reads the worker
    # source: this file is exported to the public gateway suite, where that
    # source is not present under this path.


class PinnedSileroCheckpointTest(unittest.TestCase):
    """The pinned checkpoint is a measured decision, so its identity is a literal here."""

    #: The checkpoint bundled with livekit-plugins-silero 1.8.1, pinned on
    #: 2026-09-15 after offline replay of recorded room-side caller tracks
    #: showed the v6.1 file pinned before it under-detecting quiet narrowband
    #: PSTN callers.  Changing the model is a reviewed edit in three places:
    #: the digest each Dockerfile verifies while copying the file, the
    #: constant in prewarm.py that the final image layer re-verifies against
    #: that file, and this literal.
    BUNDLED_WITH_PLUGIN_1_8_1 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
    RETIRED_V6_1 = "597d30b3ec076608d059477bb14cfeffdf951bf5cae370d38f65d33bbfe82004"

    def test_pin_is_the_reviewed_literal_not_the_retired_digest(self) -> None:
        # This suite never imports the plugin, so it cannot hash the installed
        # wheel; whether these bytes ARE the bundled file is proven by the image
        # build, which copies them out of the installed distribution under this
        # digest.  What this pins is that changing the pin is a reviewed edit.
        self.assertEqual(SILERO_VAD_PINNED_SHA256, self.BUNDLED_WITH_PLUGIN_1_8_1)
        self.assertNotEqual(SILERO_VAD_PINNED_SHA256, self.RETIRED_V6_1)

    def test_pinned_path_is_the_models_file_the_image_builds(self) -> None:
        # Both Dockerfiles write /app/models/<name>; prewarm.py resolves the
        # same models/<name> next to itself, so one constant serves the image
        # (/app/prewarm.py) and a source checkout alike.  The name deliberately
        # carries no model version.
        self.assertEqual(SILERO_VAD_PINNED_PATH.name, "silero_vad_pinned.onnx")
        self.assertEqual(SILERO_VAD_PINNED_PATH.parent.name, "models")
        self.assertEqual(SILERO_VAD_PINNED_PATH.parent.parent, Path(prewarm.__file__).resolve().parent)

    def test_loader_binds_the_pinned_path_and_keeps_the_callers_arguments(self) -> None:
        calls: list[dict[str, object]] = []

        def load(**kwargs: object) -> str:
            calls.append(kwargs)
            return "vad"

        bound = pinned_silero_vad_loader(load)
        self.assertEqual(bound(min_silence_duration=0.25), "vad")
        self.assertEqual(
            calls,
            [{"min_silence_duration": 0.25, "onnx_file_path": SILERO_VAD_PINNED_PATH}],
        )


if __name__ == "__main__":
    unittest.main()
