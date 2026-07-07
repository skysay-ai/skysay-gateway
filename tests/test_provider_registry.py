"""M5 provider registry tests (SCH-202).

Covers built-in resolution, the legacy REALTIME_PROVIDER_MODE triple expansion
and STT coupling, three-independent env vars, dotted-path + entry-point
loading, the unsupported-provider error funnels, and the mock.patch seam that
keeps the built-in adapter classes swappable through the worker module.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from openphonex_gateway.realtime.registry import (
    ENTRY_POINT_GROUPS,
    ProviderRegistry,
    ProviderSpec,
)
from openphonex_gateway.realtime.frames import AudioFrame, TranscriptEvent
from openphonex_gateway.realtime.worker import (
    RealtimeMediaWorker,
    RealtimeWorkerConfig,
    _build_default_registry,
)
from openphonex_gateway.realtime.asterisk_external_media import AsteriskExternalMediaConfig
from openphonex_gateway.realtime.adapters import FakeStreamingSTT, TemplateStreamingLLM, ToneStreamingTTS


def _config_from_env() -> RealtimeWorkerConfig:
    return RealtimeWorkerConfig.from_env()


class BuiltinResolutionTests(unittest.TestCase):
    def test_default_registry_resolves_all_six_builtins(self) -> None:
        registry = _build_default_registry()
        expected = {
            "stt": ["fake", "deepgram"],
            "llm": ["template", "gemini"],
            "tts": ["tone", "elevenlabs"],
        }
        for component, names in expected.items():
            for name in names:
                spec = registry.get(component, name)
                self.assertIsNotNone(spec, f"{component}:{name} should resolve")
                self.assertEqual(spec.name, name)
                self.assertEqual(spec.component, component)

    def test_builtin_real_flags(self) -> None:
        registry = _build_default_registry()
        self.assertFalse(registry.get("stt", "fake").real)
        self.assertTrue(registry.get("stt", "deepgram").real)
        self.assertFalse(registry.get("llm", "template").real)
        self.assertTrue(registry.get("llm", "gemini").real)
        self.assertFalse(registry.get("tts", "tone").real)
        self.assertTrue(registry.get("tts", "elevenlabs").real)


class LegacyModeTripleTests(unittest.TestCase):
    def _names(self, env: dict[str, str]) -> tuple[str, str, str]:
        with patch.dict("os.environ", env, clear=True):
            config = _config_from_env()
            return config.active_stt_name(), config.active_llm_name(), config.active_tts_name()

    def test_fake_triple(self) -> None:
        self.assertEqual(self._names({}), ("fake", "template", "tone"))

    def test_elevenlabs_triple(self) -> None:
        self.assertEqual(
            self._names({"REALTIME_PROVIDER_MODE": "elevenlabs_v3_tts"}),
            ("fake", "template", "elevenlabs"),
        )

    def test_gemini_fake_tts_triple(self) -> None:
        self.assertEqual(
            self._names({"REALTIME_PROVIDER_MODE": "gemini_llm_fake_tts"}),
            ("fake", "gemini", "tone"),
        )

    def test_gemini_elevenlabs_triple(self) -> None:
        self.assertEqual(
            self._names({"REALTIME_PROVIDER_MODE": "gemini_llm_elevenlabs_v3_tts"}),
            ("fake", "gemini", "elevenlabs"),
        )

    def test_full_realtime_triple_and_stt_coupling(self) -> None:
        # full_realtime implies Deepgram STT with no REALTIME_STT_PROVIDER set.
        self.assertEqual(
            self._names({"REALTIME_PROVIDER_MODE": "full_realtime"}),
            ("deepgram", "gemini", "elevenlabs"),
        )

    def test_full_realtime_alias_variants(self) -> None:
        for alias in (
            "full",
            "full_realtime_estonian",
            "production_realtime",
            "production_estonian",
            "deepgram_gemini_elevenlabs",
            "deepgram_gemini_elevenlabs_v3_tts",
        ):
            self.assertEqual(
                self._names({"REALTIME_PROVIDER_MODE": alias}),
                ("deepgram", "gemini", "elevenlabs"),
                alias,
            )

    def test_stt_coupling_with_explicit_fake_stt_provider(self) -> None:
        # full mode + REALTIME_STT_PROVIDER=fake still forces deepgram.
        self.assertEqual(
            self._names({"REALTIME_PROVIDER_MODE": "full_realtime", "REALTIME_STT_PROVIDER": "fake"}),
            ("deepgram", "gemini", "elevenlabs"),
        )

    def test_deepgram_stt_override_without_mode(self) -> None:
        self.assertEqual(
            self._names({"REALTIME_STT_PROVIDER": "deepgram"}),
            ("deepgram", "template", "tone"),
        )


class ThreeIndependentEnvTests(unittest.TestCase):
    def test_llm_and_tts_env_override_the_legacy_triple(self) -> None:
        env = {
            "REALTIME_PROVIDER_MODE": "fake",
            "REALTIME_LLM_PROVIDER": "gemini",
            "REALTIME_TTS_PROVIDER": "elevenlabs",
        }
        with patch.dict("os.environ", env, clear=True):
            config = _config_from_env()
            self.assertEqual(config.active_stt_name(), "fake")
            self.assertEqual(config.active_llm_name(), "gemini")
            self.assertEqual(config.active_tts_name(), "elevenlabs")
            self.assertTrue(config.uses_gemini_llm())
            self.assertTrue(config.uses_elevenlabs_tts())
            self.assertFalse(config.uses_deepgram_stt())
            components = RealtimeMediaWorker(config).provider_components()
            self.assertEqual(components["llm"]["provider"], "gemini")
            self.assertEqual(components["tts"]["provider"], "elevenlabs")
            self.assertEqual(components["stt"]["provider"], "fake")
            missing = config.missing_provider_env()
            self.assertIn("GEMINI_API_KEY", missing)
            self.assertIn("ELEVENLABS_API_KEY", missing)

    def test_unset_new_env_vars_fall_back_to_triple(self) -> None:
        with patch.dict("os.environ", {"REALTIME_PROVIDER_MODE": "gemini_llm_fake_tts"}, clear=True):
            config = _config_from_env()
            self.assertEqual(config.active_llm_name(), "gemini")
            self.assertEqual(config.active_tts_name(), "tone")


class DottedPathLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        class _CustomSTT:
            def __init__(self, config):
                self.config = config

            def transcribe(self, turn):
                yield TranscriptEvent("custom", is_final=True, language="en", metadata={"provider": "custom_stt"})

        def _describe(config):
            return {"provider": "custom_stt", "adapter": "_CustomSTT", "real": True, "language": "en", "model_id": "custom-1"}

        spec = ProviderSpec("custom_stt", "stt", _CustomSTT, _describe, lambda config: [])
        module = types.ModuleType("op_test_custom_stt_mod")
        module.SPEC = spec
        module.CustomSTT = _CustomSTT
        sys.modules["op_test_custom_stt_mod"] = module
        self.addCleanup(lambda: sys.modules.pop("op_test_custom_stt_mod", None))

    def test_dotted_path_spec_surfaces_in_provider_components(self) -> None:
        with patch.dict("os.environ", {"REALTIME_STT_PROVIDER": "op_test_custom_stt_mod:SPEC"}, clear=True):
            config = _config_from_env()
            self.assertEqual(config.active_stt_name(), "op_test_custom_stt_mod:SPEC")
            components = RealtimeMediaWorker(config).provider_components()
            self.assertEqual(components["stt"]["provider"], "custom_stt")
            self.assertEqual(components["stt"]["model_id"], "custom-1")

    def test_dotted_path_raw_class_is_adapted(self) -> None:
        with patch.dict("os.environ", {"REALTIME_STT_PROVIDER": "op_test_custom_stt_mod:CustomSTT"}, clear=True):
            config = _config_from_env()
            components = RealtimeMediaWorker(config).provider_components()
            # A raw class is adapted with a default describe: provider == env value.
            self.assertEqual(components["stt"]["provider"], "op_test_custom_stt_mod:CustomSTT")
            self.assertEqual(components["stt"]["adapter"], "_CustomSTT")
            self.assertTrue(components["stt"]["real"])

    def test_malformed_dotted_path_degrades_to_unsupported(self) -> None:
        with patch.dict(
            "os.environ",
            {"REALTIME_STT_PROVIDER": "op_test_nonexistent.module:Nope"},
            clear=True,
        ):
            worker = RealtimeMediaWorker(_config_from_env())
            self.assertEqual(worker.config.stt_kind(), "unsupported")
            with self.assertRaisesRegex(RuntimeError, "Unsupported REALTIME_STT_PROVIDER"):
                worker.run_offline_synthetic_call()


class EntryPointLoadingTests(unittest.TestCase):
    def _fake_entry_points(self, good_spec: ProviderSpec):
        class _EP:
            def __init__(self, name, loader):
                self.name = name
                self.group = ENTRY_POINT_GROUPS["tts"]
                self._loader = loader

            def load(self):
                return self._loader()

        def _broken():
            raise RuntimeError("boom")

        entries = [_EP("myvoice", lambda: good_spec), _EP("brokenvoice", _broken)]

        def _entry_points(*args, **kwargs):
            group = kwargs.get("group")
            if group == ENTRY_POINT_GROUPS["tts"]:
                return entries
            return []

        return _entry_points

    def test_entry_point_spec_registers_and_broken_one_degrades(self) -> None:
        def _factory(config):
            class _EPTTS:
                def stream(self, text):
                    yield AudioFrame.silence(20, direction="agent")

            return _EPTTS()

        good_spec = ProviderSpec(
            "myvoice",
            "tts",
            _factory,
            lambda config: {"provider": "myvoice", "adapter": "_EPTTS", "real": True, "model_id": "ep-1"},
            lambda config: [],
        )
        registry = ProviderRegistry()
        with patch("importlib.metadata.entry_points", self._fake_entry_points(good_spec)):
            resolved = registry.get("tts", "myvoice")
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.name, "myvoice")
            # A broken entry point degrades to None instead of crashing.
            self.assertIsNone(registry.get("tts", "brokenvoice"))


class PatchSeamRegressionTests(unittest.TestCase):
    def test_builtin_factories_read_worker_module_globals(self) -> None:
        constructed: dict[str, object] = {}

        class FakeDeepgram:
            def __init__(self, config):
                constructed["stt"] = self

        class FakeGemini:
            def __init__(self, config):
                constructed["llm"] = self

        class FakeElevenLabs:
            def __init__(self, config):
                constructed["tts"] = self

        config = RealtimeWorkerConfig(
            calls_dir=__import__("pathlib").Path("/tmp"),
            recordings_dir=__import__("pathlib").Path("/tmp"),
            rtp_bind="127.0.0.1",
            rtp_port=12000,
            asterisk=AsteriskExternalMediaConfig(ari_user="", ari_password=""),
            provider_mode="full_realtime",
            deepgram_api_key="dg",
            gemini_api_key="gm",
            elevenlabs_api_key="el",
            elevenlabs_voice_id="voice-1",
        )
        worker = RealtimeMediaWorker(config)
        with (
            patch("openphonex_gateway.realtime.worker.DeepgramStreamingSTT", FakeDeepgram),
            patch("openphonex_gateway.realtime.worker.GeminiStreamingLLM", FakeGemini),
            patch("openphonex_gateway.realtime.worker.ElevenLabsV3StreamingTTS", FakeElevenLabs),
            patch("openphonex_gateway.realtime.worker.resolve_elevenlabs_voice_id", return_value="voice-1"),
        ):
            stt, llm, tts = worker._provider_adapters()
        self.assertIs(stt, constructed["stt"])
        self.assertIs(llm, constructed["llm"])
        self.assertIs(tts, constructed["tts"])
        self.assertIsInstance(stt, FakeDeepgram)
        self.assertIsInstance(llm, FakeGemini)
        self.assertIsInstance(tts, FakeElevenLabs)

    def test_fake_builtins_still_constructed_for_default_mode(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            worker = RealtimeMediaWorker(_config_from_env())
            stt, llm, tts = worker._provider_adapters()
        self.assertIsInstance(stt, FakeStreamingSTT)
        self.assertIsInstance(llm, TemplateStreamingLLM)
        self.assertIsInstance(tts, ToneStreamingTTS)


class GA5ProviderResolutionTests(unittest.TestCase):
    """GA-5 (SCH-216) hosted adapters resolve under the catalog registry_names."""

    def test_registry_resolves_all_ga5_names_and_aliases(self) -> None:
        registry = _build_default_registry()
        expected = {
            ("stt", "google"): ("google", "stt"),
            ("stt", "openai"): ("openai", "stt"),
            ("stt", "elevenlabs_scribe"): ("elevenlabs_scribe", "stt"),
            ("llm", "claude"): ("claude", "llm"),
            ("llm", "anthropic"): ("claude", "llm"),  # alias
            ("llm", "openai"): ("openai", "llm"),
            ("llm", "gpt"): ("openai", "llm"),  # alias
            ("tts", "deepgram_tts"): ("deepgram_tts", "tts"),
            ("tts", "aura"): ("deepgram_tts", "tts"),  # alias
            ("tts", "cartesia"): ("cartesia", "tts"),
        }
        for (component, name), (spec_name, spec_component) in expected.items():
            self.assertTrue(registry.can_resolve(component, name), f"{component}:{name}")
            spec = registry.get(component, name)
            self.assertIsNotNone(spec, f"{component}:{name}")
            self.assertEqual(spec.name, spec_name)
            self.assertEqual(spec.component, spec_component)

    def test_stt_and_llm_openai_are_distinct_slots(self) -> None:
        registry = _build_default_registry()
        stt = registry.get("stt", "openai")
        llm = registry.get("llm", "openai")
        self.assertEqual(stt.component, "stt")
        self.assertEqual(llm.component, "llm")
        # Same module, two specs -> different adapter classes.
        self.assertNotEqual(stt.factory, llm.factory)

    def test_build_acceptance_trio_claude_google_cartesia(self) -> None:
        registry = _build_default_registry()
        env = {"ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k", "CARTESIA_API_KEY": "k", "CARTESIA_VOICE_ID": "v"}
        with patch.dict("os.environ", env, clear=False):
            llm = registry.build("llm", "claude", _config_from_env())
            stt = registry.build("stt", "google", _config_from_env())
            tts = registry.build("tts", "cartesia", _config_from_env())
        self.assertEqual(type(llm).__name__, "AnthropicStreamingLLM")
        self.assertEqual(type(stt).__name__, "GoogleStreamingSTT")
        self.assertEqual(type(tts).__name__, "CartesiaTTS")


class UnsupportedProviderTests(unittest.TestCase):
    def test_unsupported_mode_raises_mode_message(self) -> None:
        with patch.dict("os.environ", {"REALTIME_PROVIDER_MODE": "totally_unknown"}, clear=True):
            worker = RealtimeMediaWorker(_config_from_env())
            self.assertEqual(worker.config.provider_kind(), "unsupported")
            with self.assertRaisesRegex(RuntimeError, "Unsupported REALTIME_PROVIDER_MODE"):
                worker.run_offline_synthetic_call()

    def test_unsupported_llm_env_raises_llm_message(self) -> None:
        with patch.dict("os.environ", {"REALTIME_LLM_PROVIDER": "no_such_llm"}, clear=True):
            worker = RealtimeMediaWorker(_config_from_env())
            self.assertEqual(worker.config.llm_kind(), "unsupported")
            with self.assertRaisesRegex(RuntimeError, "Unsupported REALTIME_LLM_PROVIDER"):
                worker.run_offline_synthetic_call()

    def test_unsupported_tts_env_raises_tts_message(self) -> None:
        with patch.dict("os.environ", {"REALTIME_TTS_PROVIDER": "no_such_tts"}, clear=True):
            worker = RealtimeMediaWorker(_config_from_env())
            self.assertEqual(worker.config.tts_kind(), "unsupported")
            with self.assertRaisesRegex(RuntimeError, "Unsupported REALTIME_TTS_PROVIDER"):
                worker.run_offline_synthetic_call()


if __name__ == "__main__":
    unittest.main()
