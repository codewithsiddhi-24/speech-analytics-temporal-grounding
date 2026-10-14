import base64
import json
import wave
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import train_timestamp_review_model as review_model


class TimestampReviewModelTests(unittest.TestCase):
    def test_reference_model_uses_words_from_every_available_recording(self):
        recording_ids = ["good_01", "good_02", "good_15"]
        calls = []

        def fake_extract(recording_id):
            calls.append(recording_id)
            records = [
                {
                    "recording_id": recording_id,
                    "word_index": index,
                    "word": f"word{index}",
                    "start": index * 0.2,
                    "end": index * 0.2 + 0.1,
                    "duration_seconds": 0.1,
                    "alignment_score": 0.8,
                    "gap_before_seconds": 0.1,
                    "gap_after_seconds": 0.1,
                    "word_rms_db_median": -25.0,
                    "word_voiced_fraction": 0.8,
                    "word_zero_crossing_rate_median": 0.1,
                    "start_boundary_rms_change_db": 1.0,
                    "end_boundary_rms_change_db": -1.0,
                    "start_boundary_voiced_change": 0.1,
                    "end_boundary_voiced_change": -0.1,
                    "start_boundary_centroid_change_hz": 100.0,
                    "end_boundary_centroid_change_hz": -100.0,
                }
                for index in range(20)
            ]
            matrix = np.full(
                (len(records), len(review_model.FEATURE_NAMES)),
                recording_ids.index(recording_id),
                dtype=np.float64,
            )
            return records, matrix

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_dir, audio_dir, transcript_dir = (
                root / "features",
                root / "audio",
                root / "transcripts",
            )
            feature_dir.mkdir()
            audio_dir.mkdir()
            transcript_dir.mkdir()
            times = np.arange(0, 1, 0.01)
            for recording_id in recording_ids:
                np.savez(
                    feature_dir / f"{recording_id}_frames.npz",
                    time=times,
                    rms=np.full(times.shape, 0.1),
                    rms_db=np.full(times.shape, -25.0),
                    f0_hz=np.full(times.shape, 180.0),
                    spectral_centroid_hz=np.full(times.shape, 1200.0),
                    voiced_probability=np.full(times.shape, 0.8),
                )
                audio_path = audio_dir / f"{recording_id}.wav"
                with wave.open(str(audio_path), "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(16000)
                    audio.writeframes(np.zeros(16000, dtype="<i2").tobytes())

            with (
                patch.object(review_model, "_recording_ids", return_value=recording_ids),
                patch.object(review_model, "extract_recording_features", side_effect=fake_extract),
                patch.object(review_model, "FEATURE_DIR", feature_dir),
                patch.object(review_model, "AUDIO_DIR", audio_dir),
                patch.object(review_model, "TRANSCRIPT_DIR", transcript_dir),
            ):
                report, _ = review_model.train_and_score()

        self.assertEqual(calls, recording_ids)
        self.assertEqual(report["baseline_training_recording_ids"], recording_ids)
        self.assertEqual(report["baseline_training_words"], 60)
        self.assertIn("combined reference-corpus distribution", report["model_scope"])

    def test_features_include_alignment_context_and_local_acoustic_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_dir = root / "features"
            alignment_dir = root / "alignments"
            feature_dir.mkdir()
            alignment_dir.mkdir()
            (alignment_dir / "good_15.json").write_text(
                json.dumps(
                    {
                        "words": [
                            {
                                "index": 0,
                                "word": "one",
                                "start": 0.2,
                                "end": 0.4,
                                "score": 0.9,
                            },
                            {
                                "index": 1,
                                "word": "two",
                                "start": 0.6,
                                "end": 0.9,
                                "score": 0.4,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            times = np.arange(0, 1.01, 0.01)
            rms_db = np.where(times < 0.6, -30.0, -12.0)
            voiced = np.where(times < 0.6, 0.2, 0.9)
            centroid = np.where(times < 0.6, 1200.0, 2000.0)
            np.savez(
                feature_dir / "good_15_frames.npz",
                time=times,
                rms_db=rms_db,
                voiced_probability=voiced,
                spectral_centroid_hz=centroid,
                zero_crossing_rate=np.full(times.shape, 0.1),
            )

            with (
                patch.object(review_model, "ALIGNMENT_DIR", alignment_dir),
                patch.object(review_model, "FEATURE_DIR", feature_dir),
            ):
                words, matrix = review_model.extract_recording_features("good_15")

        self.assertEqual(len(words), 2)
        self.assertEqual(matrix.shape, (2, len(review_model.FEATURE_NAMES)))
        self.assertAlmostEqual(words[1]["gap_before_seconds"], 0.2)
        self.assertEqual(
            matrix[1, review_model.FEATURE_NAMES.index("alignment_score")],
            0.4,
        )
        self.assertGreater(
            matrix[1, review_model.FEATURE_NAMES.index("start_boundary_rms_change_db")],
            0,
        )
        self.assertGreater(
            matrix[1, review_model.FEATURE_NAMES.index("start_boundary_voiced_change")],
            0,
        )

    def test_audio_visual_data_contains_waveform_and_spectrogram(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "recording.wav"
            samples = (
                np.sin(2 * np.pi * 440 * np.arange(1600) / 16000) * 16000
            ).astype("<i2")
            with wave.open(str(audio_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(samples.tobytes())
            visual = review_model._audio_visual_data(audio_path)

        self.assertAlmostEqual(visual["duration_seconds"], 0.1)
        self.assertEqual(len(visual["waveform_minmax"]), 900)
        self.assertEqual(visual["spectrogram"]["height"], 64)
        self.assertEqual(visual["spectrogram"]["width"], 1)
        self.assertEqual(
            len(base64.b64decode(visual["spectrogram"]["png_data"])), 64
        )


if __name__ == "__main__":
    unittest.main()
