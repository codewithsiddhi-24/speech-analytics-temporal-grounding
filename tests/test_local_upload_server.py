import base64
import http.client
import io
import json
import tempfile
import threading
import unittest
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import local_upload_server as upload_server


class FakeWhisperModel:
    def transcribe(self, *_args, **_kwargs):
        words = [
            SimpleNamespace(word="hello", start=0.10, end=0.32, probability=0.91),
            SimpleNamespace(word="unclear", start=0.70, end=0.92, probability=0.31),
        ]
        segment = SimpleNamespace(
            text="hello unclear",
            words=words,
        )
        return iter([segment]), SimpleNamespace(duration=1.0)


class BrokenWhisperModel:
    def transcribe(self) -> None:
        pass


class LocalUploadServerTests(unittest.TestCase):
    def test_dashboard_starts_with_upload_only_and_no_recording_selector(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), upload_server.DashboardHandler
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=10
            )
            connection.request("GET", "/")
            response = connection.getresponse()
            html = response.read().decode("utf-8")
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(response.status, 200)
        self.assertIn('id="analyzerOutput" hidden', html)
        self.assertIn('id="uploadStatus" role="status" hidden', html)
        self.assertNotIn('id="recordingSelect"', html)

    def test_local_http_upload_returns_processed_recording_json(self):
        times = np.arange(16000) / 16000
        samples = (10000 * np.sin(2 * np.pi * 180 * times)).astype("<i2")
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(samples.tobytes())

        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), upload_server.DashboardHandler
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(
                upload_server, "_load_model", return_value=FakeWhisperModel()
            ):
                connection = http.client.HTTPConnection(
                    "127.0.0.1", server.server_address[1], timeout=10
                )
                connection.request(
                    "POST",
                    "/api/analyze",
                    body=audio_buffer.getvalue(),
                    headers={
                        "Content-Type": "audio/wav",
                        "X-Filename": "http-test.wav",
                    },
                )
                response = connection.getresponse()
                payload = json.loads(response.read())
                connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(response.status, 200)
        self.assertEqual(payload["display_name"], "http-test.wav")
        self.assertEqual(payload["recording_id"][:7], "upload_")
        self.assertIn("causal_findings", payload["analysis"])
        self.assertTrue(payload["analysis"]["feature_series"]["f0_hz"])

    def test_http_upload_returns_json_when_transcription_dependency_fails(self):
        samples = np.zeros(16000, dtype="<i2")
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(samples.tobytes())

        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), upload_server.DashboardHandler
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(
                upload_server, "_load_model", return_value=BrokenWhisperModel()
            ):
                connection = http.client.HTTPConnection(
                    "127.0.0.1", server.server_address[1], timeout=10
                )
                connection.request(
                    "POST",
                    "/api/analyze",
                    body=audio_buffer.getvalue(),
                    headers={
                        "Content-Type": "audio/wav",
                        "X-Filename": "dependency-error.wav",
                    },
                )
                response = connection.getresponse()
                payload = json.loads(response.read())
                connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(response.status, 500)
        self.assertIn("dependency is incompatible", payload["error"])

    def test_static_server_does_not_expose_project_files(self):
        blocked_path = upload_server.DashboardHandler.translate_path(
            None, "/training_frames.csv"
        )
        self.assertEqual(Path(blocked_path).name, "__not_served__")

    def test_gap_reason_distinguishes_quiet_from_active_gaps_cautiously(self):
        times = np.arange(10, dtype=np.float64) * 0.01
        quiet = np.full(10, -60.0)
        active = np.full(10, -30.0)

        quiet_reason, quiet_fraction = upload_server._gap_reason(
            times, quiet, 0.01, 0.08, -40.0
        )
        active_reason, active_fraction = upload_server._gap_reason(
            times, active, 0.01, 0.08, -40.0
        )

        self.assertIn("natural pause or silence", quiet_reason)
        self.assertEqual(quiet_fraction, 1.0)
        self.assertIn("possible breath", active_reason)
        self.assertEqual(active_fraction, 0.0)

    def test_upload_report_has_word_times_gap_reasons_and_visual_data(self):
        sample_rate = 16000
        samples = np.zeros(sample_rate, dtype="<i2")
        samples[1600:5600] = (
            np.sin(2 * np.pi * 220 * np.arange(4000) / sample_rate) * 12000
        ).astype("<i2")
        samples[11200:15200] = (
            np.sin(2 * np.pi * 220 * np.arange(4000) / sample_rate) * 12000
        ).astype("<i2")

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.wav"
            with wave.open(str(audio_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(sample_rate)
                audio.writeframes(samples.tobytes())

            result = upload_server.analyze_upload(
                audio_path,
                "sample.wav",
                FakeWhisperModel(),
                upload_server._load_reference_model(),
            )

        self.assertEqual(result["display_name"], "sample.wav")
        self.assertTrue(result["reference_comparison"]["available"])
        self.assertEqual(
            result["reference_comparison"]["recording_count"],
            len(upload_server._load_reference_model()["baseline_recording_ids"]),
        )
        self.assertEqual(len(result["words"]), 2)
        self.assertEqual(result["words"][1]["start"], 0.7)
        self.assertTrue(
            all(word["baseline_anomaly_score"] is not None for word in result["words"])
        )
        self.assertTrue(result["words"][1]["asr_unclear_candidate"])
        self.assertEqual(result["analysis"]["scores"]["recognition_confidence"], 61)
        self.assertEqual(result["analysis"]["scores"]["high_confidence_words"], 50)
        self.assertIsNone(result["analysis"]["scores"]["phrase_pace_in_range"])
        self.assertEqual(len(result["analysis"]["gaps"]), 1)
        self.assertIn("natural pause or silence", result["analysis"]["gaps"][0]["reason"])
        findings = result["analysis"]["causal_findings"]
        self.assertFalse(
            any(finding["category"] == "LOW-ENERGY PAUSE" for finding in findings)
        )
        self.assertEqual(
            sum(
                finding["category"]
                in {"POSSIBLE NATURAL PAUSE / BREATH", "INTER-WORD GAP"}
                for finding in findings
            ),
            len(result["analysis"]["gaps"]),
        )
        for pause in result["analysis"]["pauses"]:
            for word in result["words"]:
                self.assertLessEqual(
                    min(pause["end"], word["end"])
                    - max(pause["start"], word["start"]),
                    0.0,
                )
        unclear = next(
            finding
            for finding in findings
            if finding["category"] == "POSSIBLE MUMBLE / UNCLEAR SPEECH"
        )
        self.assertEqual(unclear["transcript"], "unclear")
        self.assertIn("ASR confidence 0.31", unclear["evidence"])
        self.assertTrue(result["analysis"]["feature_series"]["f0_hz"])
        self.assertTrue(
            result["analysis"]["feature_series"]["spectral_centroid_hz"]
        )
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["audio_visual"]["spectrogram"]["height"], 128)
        self.assertEqual(
            len(
                base64.b64decode(
                    result["audio_visual"]["spectrogram"]["png_data"]
                )
            ),
            128 * result["audio_visual"]["spectrogram"]["width"],
        )

    def test_phrase_pace_score_uses_the_documented_range(self):
        words = [
            {
                "start": float(index) * 0.6,
                "end": float(index) * 0.6 + 0.5,
                "asr_confidence": 0.9,
            }
            for index in range(5)
        ]

        scores = upload_server._upload_scores(words)

        self.assertEqual(scores["phrase_pace_in_range"], 100)
        self.assertEqual(scores["phrase_pace_window_count"], 1)

    def test_adjacent_reference_candidates_are_grouped_with_context(self):
        words = [
            {
                "word": text,
                "start": index * 0.3,
                "end": index * 0.3 + 0.24,
                "model_review_candidate": True,
                "baseline_anomaly_score": -0.7,
                "baseline_score_threshold": -0.5,
            }
            for index, text in enumerate(("one", "two", "three"))
        ]

        findings = upload_server._causal_findings(
            words,
            [],
            [],
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            2.0,
        )
        candidates = [
            item
            for item in findings
            if item["category"] == "REFERENCE-DEVIATION CANDIDATE"
        ]

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["transcript"], "one two three")
        self.assertEqual(candidates[0]["start"], 0.0)
        self.assertEqual(candidates[0]["end"], 0.84)
        self.assertEqual(candidates[0]["listen_start"], 0.0)
        self.assertGreater(candidates[0]["listen_end"], candidates[0]["end"])

    def test_relative_delivery_cues_use_voiced_pitch_and_energy(self):
        words = [
            {"word": f"w{index}", "start": float(index), "end": index + 0.8}
            for index in range(10)
        ]
        times = np.arange(0.0, 10.0, 0.05)
        pitch = np.select(
            [times < 4.4, times < 5.2],
            [240.0, 200.0],
            default=150.0,
        )
        levels = np.select(
            [times < 4.4, times < 5.2],
            [-20.0, -27.5],
            default=-35.0,
        )

        findings = upload_server._causal_findings(
            words,
            [],
            [],
            times,
            pitch,
            times,
            levels,
            10.0,
        )
        categories = {item["category"] for item in findings}

        self.assertIn("HIGHER PITCH THAN RECORDING TYPICAL", categories)
        self.assertIn("LOWER PITCH THAN RECORDING TYPICAL", categories)
        self.assertIn("HIGHER VOICE LEVEL", categories)
        self.assertIn("QUIETER VOICE LEVEL", categories)
        for finding in findings:
            self.assertLessEqual(finding["listen_start"], finding["start"])
            self.assertGreaterEqual(finding["listen_end"], finding["end"])

    def test_rejects_non_16_bit_pcm_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.wav"
            with wave.open(str(audio_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(1)
                audio.setframerate(16000)
                audio.writeframes(b"\x80" * 100)

            with self.assertRaisesRegex(ValueError, "16-bit PCM"):
                upload_server._read_pcm_wav(audio_path)


if __name__ == "__main__":
    unittest.main()
