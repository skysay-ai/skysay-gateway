from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openphonex_gateway.asterisk_voice_gateway import (
    AsteriskVoiceGatewayConfig,
    AsteriskVoiceGatewayWorker,
    detect_turn_window,
    extract_segment,
    linear_to_mulaw,
    read_wav_samples,
    waveform_peaks,
    write_pcm16_wav,
)
from openphonex_gateway.rtp import mulaw_to_linear


def synthetic_turn_wav(path: Path) -> None:
    rate = 8000
    samples: list[int] = []
    samples.extend([0] * int(rate * 0.4))
    for idx in range(int(rate * 1.2)):
        samples.append(int(9000 * math.sin(2 * math.pi * 440 * idx / rate)))
    samples.extend([0] * int(rate * 0.8))
    write_pcm16_wav(path, samples, rate)


class AsteriskVoiceGatewayTests(unittest.TestCase):
    def test_config_is_env_driven_and_redacts_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_file = Path(td) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "ASTERISK_ARI_URL=http://asterisk.local:8088/ari",
                        "ASTERISK_ARI_USER=ari-user",
                        "ASTERISK_ARI_PASSWORD=super-secret",
                        "OPENPHONEX_CALLS_DIR=/tmp/calls",
                        "OPENPHONEX_VOICE_RTP_PORT=10042",
                        "OPENPHONEX_VOICE_LIVE_ENABLED=1",
                        "OPENPHONEX_LEGACY_VOICEBOT_SCRIPT=/opt/telephony/asterisk_voicebot.py",
                        "OPENPHONEX_LEGACY_PYTHON=/venv/bin/python",
                    ]
                )
            )
            with patch.dict(os.environ, {}, clear=True):
                config = AsteriskVoiceGatewayConfig.from_env(env_file)
        status = config.public_status()
        self.assertEqual(config.ari_url, "http://asterisk.local:8088/ari")
        self.assertEqual(config.rtp_port, 10042)
        self.assertTrue(config.live_enabled)
        self.assertEqual(status["missing_live_env"], [])
        self.assertNotIn("super-secret", json.dumps(status))
        self.assertNotIn("/Users/", json.dumps(status))
        self.assertIn("--rtp-port", config.legacy_voicebot_args())
        command = config.legacy_voicebot_command()
        self.assertEqual(command[0], "/venv/bin/python")
        self.assertEqual(command[1], "/opt/telephony/asterisk_voicebot.py")
        self.assertIn("--opening", command)

    def test_config_reports_missing_live_credentials_without_crashing(self) -> None:
        with patch.dict(os.environ, {"OPENPHONEX_VOICE_LIVE_ENABLED": "1"}, clear=True):
            config = AsteriskVoiceGatewayConfig.from_env()
        worker = AsteriskVoiceGatewayWorker(config)
        status = worker.status()
        self.assertFalse(status["ok"])
        self.assertEqual(status["version"], "0.1.0")
        self.assertEqual(status["environment"], "local")
        self.assertEqual(status["readiness"]["provider"]["status"], "missing_live_env")
        self.assertIn("storage", status["readiness"])
        self.assertEqual(status["config"]["missing_live_env"], ["ASTERISK_ARI_USER", "ASTERISK_ARI_PASSWORD"])
        self.assertEqual(status["realtime_pipeline"]["pipecat_dependency"], "none")
        self.assertEqual(status["realtime_pipeline"]["livekit_dependency"], "none")

    def test_offline_recorded_audio_processing_extracts_turn_and_waveform(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "recorded.wav"
            synthetic_turn_wav(source)
            start, end = detect_turn_window(source, speech_rms=650, silence_rms=220, silence_hold_sec=0.2, min_speech_sec=0.3)
            self.assertLess(start, 0.5)
            self.assertGreater(end, 1.3)

            segment = root / "segment.wav"
            self.assertTrue(extract_segment(source, start, end, segment))
            self.assertGreater(max(waveform_peaks(segment)), 0.05)

            config = AsteriskVoiceGatewayConfig.from_env()
            worker = AsteriskVoiceGatewayWorker(config)
            result = worker.process_recorded_audio(source, root / "out")
            self.assertTrue(result["ok"])
            self.assertTrue(Path(result["segment_wav"]).exists())
            self.assertGreater(result["segment_max_rms"], 1000)
            self.assertTrue((root / "out" / "offline-result.json").exists())

    def test_mulaw_round_trip_is_close_enough_for_rtp_audio(self) -> None:
        for sample in (-12000, -3000, 0, 3000, 12000):
            encoded = linear_to_mulaw(sample)
            decoded = mulaw_to_linear(encoded)
            self.assertLess(abs(decoded - sample), 1200)

    def test_read_wav_samples_handles_standard_pcm16(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "audio.wav"
            write_pcm16_wav(wav, [0, 1000, -1000, 0], 8000)
            samples, rate = read_wav_samples(wav)
        self.assertEqual(rate, 8000)
        self.assertEqual(samples[:4], (0, 1000, -1000, 0))


if __name__ == "__main__":
    unittest.main()
