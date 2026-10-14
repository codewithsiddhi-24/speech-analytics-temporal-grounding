import csv
import json
import math
import struct
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from train_compare_models import (
    evaluate_models,
    load_group_balanced_sample,
)
from verify_original_audio import (
    analyze_wav,
    generate_report,
    update_review,
)


def write_pcm(path: Path, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    for index in range(sample_rate * 2):
        if sample_rate // 2 <= index < sample_rate * 3 // 2:
            amplitude = int(
                0.25 * 32767 * math.sin(2 * math.pi * 220 * index / sample_rate)
            )
        else:
            amplitude = 0
        samples.append(amplitude)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(struct.pack(f"<{len(samples)}h", *samples))


class OriginalAudioReviewTests(unittest.TestCase):
    def test_wav_metrics_detect_silence_and_activity_without_calling_it_a_flaw(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            write_pcm(path)

            metrics = analyze_wav(path)

        self.assertEqual(metrics["readable"], "yes")
        self.assertEqual(metrics["sample_rate"], 16000)
        self.assertEqual(metrics["channels"], 1)
        self.assertAlmostEqual(metrics["duration_seconds"], 2.0)
        self.assertAlmostEqual(metrics["speech_activity_ratio"], 0.5, delta=0.03)
        self.assertAlmostEqual(metrics["silence_ratio"], 0.5, delta=0.03)
        self.assertAlmostEqual(metrics["leading_silence_seconds"], 0.5, delta=0.03)
        self.assertAlmostEqual(metrics["trailing_silence_seconds"], 0.5, delta=0.04)
        self.assertEqual(metrics["clipping_ratio"], 0.0)

    def test_report_defaults_pending_and_preserves_human_review_on_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "clean_reference" / "sample.wav"
            standardized = root / "standardized_reference" / "sample.wav"
            write_pcm(original)
            write_pcm(standardized)

            feature_directory = root / "features"
            feature_directory.mkdir()
            (feature_directory / "sample_summary.json").write_text(
                json.dumps(
                    {
                        "sample_rate": 16000,
                        "duration_sec": 2.0,
                    }
                ),
                encoding="utf-8",
            )
            with zipfile.ZipFile(
                feature_directory / "sample_frames.npz", "w"
            ) as archive:
                archive.writestr("placeholder.npy", "valid zip archive")

            (root / "transcripts").mkdir()
            (root / "transcripts" / "sample.txt").write_text(
                "Hello, world.", encoding="utf-8"
            )
            (root / "alignments").mkdir()
            (root / "alignments" / "sample.json").write_text(
                json.dumps(
                    {
                        "duration": 2.0,
                        "words": [
                            {"word": "Hello", "start": 0.5, "end": 0.8},
                            {"word": "world.", "start": 0.9, "end": 1.2},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            labels = root / "flaw_labels.csv"
            with labels.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "audio_id",
                        "source_audio",
                        "flaw_type",
                        "synthetic",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "audio_id": "sample_pause_1s",
                        "source_audio": "sample.wav",
                        "flaw_type": "synthetic_inserted_pause",
                        "synthetic": "True",
                    }
                )
            report = root / "original_audio_review.csv"

            rows = generate_report(root, report)
            self.assertEqual(rows[0]["review_status"], "PENDING_REVIEW")
            self.assertEqual(rows[0]["verified_by_human"], "no")
            self.assertEqual(rows[0]["transcript_alignment_match"], "yes")
            self.assertEqual(rows[0]["feature_duration_match"], "yes")
            self.assertEqual(rows[0]["training_review_needed"], "yes")

            update_review(
                "sample",
                "VERIFIED_FLAWED",
                "Audible issue",
                "Tester",
                "clipping",
                "1.0",
                "1.5",
                report,
            )
            rows = generate_report(root, report)

        self.assertEqual(rows[0]["review_status"], "VERIFIED_FLAWED")
        self.assertEqual(rows[0]["verified_by_human"], "yes")
        self.assertEqual(rows[0]["review_notes"], "Audible issue")
        self.assertEqual(rows[0]["flaw_type"], "clipping")
        self.assertEqual(rows[0]["flaw_start_seconds"], "1.0")
        self.assertEqual(rows[0]["flaw_end_seconds"], "1.5")

    def test_bad_existing_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pcm(root / "clean_reference" / "sample.wav")
            report = root / "original_audio_review.csv"
            report.write_text("audio_id,old_column\nsample,keep me\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                generate_report(root, report)

            self.assertEqual(
                report.read_text(encoding="utf-8"),
                "audio_id,old_column\nsample,keep me\n",
            )

    def test_training_sample_is_capped_per_source_class_and_excludes_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "time_sec",
                        "feature",
                        "target",
                        "audio_id",
                        "source_audio",
                        "flaw_type",
                    ],
                )
                writer.writeheader()
                for source_index in range(4):
                    for label in ("normal", "flaw"):
                        for frame in range(20):
                            writer.writerow(
                                {
                                    "time_sec": frame / 100,
                                    "feature": source_index + frame,
                                    "target": label,
                                    "audio_id": f"{source_index}_{label}",
                                    "source_audio": f"source_{source_index}.wav",
                                    "flaw_type": label,
                                }
                            )

            x, y, groups, feature_names, audit = load_group_balanced_sample(
                path, max_per_source_class=3, seed=1
            )

        self.assertEqual(x.shape, (24, 1))
        self.assertEqual(feature_names, ["feature"])
        self.assertNotIn("time_sec", feature_names)
        self.assertEqual(set(y), {"normal", "flaw"})
        self.assertEqual(len(set(groups)), 4)
        self.assertEqual(audit["sampled_rows"], 24)
        for count in audit["sampled_rows_per_source_class"].values():
            self.assertEqual(count, 3)

    def test_grouped_model_evaluation_has_no_source_overlap(self):
        rng = np.random.default_rng(4)
        groups = np.repeat([f"source_{i}" for i in range(4)], 20)
        y = np.tile(np.repeat(["normal", "flaw"], 10), 4)
        x = rng.normal(size=(len(y), 3))
        x[:, 0] += (y == "flaw") * 2.0

        rows, ranking, audit = evaluate_models(x, y, groups, seed=4)

        self.assertEqual(len(rows), 20)
        self.assertEqual(len(ranking), 5)
        self.assertIn("rbf_svm", ranking)
        self.assertIn("random_forest", ranking)
        for fold in audit["splits"]:
            train, test = set(fold["train_groups"]), set(fold["test_groups"])
            self.assertFalse(train & test)
            self.assertEqual(fold["overlap"], [])
        self.assertTrue(
            all(0 <= metrics["accuracy"] <= 1 for metrics in ranking.values())
        )


if __name__ == "__main__":
    unittest.main()
