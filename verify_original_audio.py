"""Create and maintain a human-review report for original recordings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import struct
import wave
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_DIR = BASE_DIR / "clean_reference"
STANDARDIZED_DIR = BASE_DIR / "standardized_reference"
FEATURE_DIR = BASE_DIR / "features"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
ALIGNMENT_DIR = BASE_DIR / "alignments"
LABEL_FILE = BASE_DIR / "flaw_labels.csv"
TRAINING_FILE = BASE_DIR / "training_frames.csv"
REPORT_FILE = BASE_DIR / "original_audio_review.csv"

REPORT_FIELDS = [
    "audio_id",
    "file_path",
    "standardized_file_path",
    "audio_format",
    "readable",
    "duration_seconds",
    "sample_rate",
    "channels",
    "sample_width_bits",
    "peak_dbfs",
    "rms_dbfs",
    "clipping_ratio",
    "silence_ratio",
    "leading_silence_seconds",
    "trailing_silence_seconds",
    "speech_activity_ratio",
    "speech_activity_method",
    "noise_indicator_dbfs",
    "noise_indicator_method",
    "long_internal_pause_count",
    "longest_internal_pause_seconds",
    "standardized_sample_rate",
    "standardized_duration_seconds",
    "feature_summary_available",
    "feature_archive_available",
    "feature_archive_valid",
    "feature_sample_rate",
    "feature_duration_seconds",
    "feature_duration_match",
    "transcript_available",
    "alignment_available",
    "transcript_alignment_match",
    "transcript_word_count",
    "alignment_word_count",
    "automated_flags",
    "file_errors",
    "normal_training_variant_ids",
    "training_review_needed",
    "review_status",
    "review_notes",
    "verified_by_human",
    "reviewed_by",
    "reviewed_at_utc",
    "flaw_type",
    "flaw_start_seconds",
    "flaw_end_seconds",
]

REVIEW_STATUSES = (
    "PENDING_REVIEW",
    "VERIFIED_GOOD",
    "VERIFIED_FLAWED",
    "UNCERTAIN",
)
HUMAN_FIELDS = (
    "review_status",
    "review_notes",
    "verified_by_human",
    "reviewed_by",
    "reviewed_at_utc",
    "flaw_type",
    "flaw_start_seconds",
    "flaw_end_seconds",
)

# These inherit the existing check_quality.py / inspect_long_pauses.py
# cutoffs. They are review cues for this corpus, not universal quality limits.
CLIPPING_SAMPLE_THRESHOLD = 0.99
CLIPPING_RATIO_FLAG = 0.01
ACTIVITY_RELATIVE_FLOOR_DB = 35.0
LOW_ACTIVITY_RATIO_FLAG = 0.20
EDGE_SILENCE_FLAG_SECONDS = 3.0
LONG_PAUSE_FLAG_SECONDS = 3.0
LOW_RMS_FLAG_DBFS = -35.0
MIN_DURATION_FLAG_SECONDS = 2.0
MAX_DURATION_FLAG_SECONDS = 600.0
EXPECTED_STANDARDIZED_SAMPLE_RATE = 16000
FEATURE_DURATION_TOLERANCE_SECONDS = 0.05
FRAME_SECONDS = 0.025
REPORT_ENCODING = "utf-8-sig"


def _relative_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(path.resolve())


def _dbfs(amplitude: float) -> float:
    if not math.isfinite(amplitude) or amplitude <= 1e-12:
        return -120.0
    return 20.0 * math.log10(amplitude)


def _percentile(values: list[float], percentage: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _decode_samples(data: bytes, sample_width: int) -> list[float]:
    if sample_width == 1:
        return [(sample - 128) / 128.0 for sample in data]
    if sample_width == 2:
        return [sample[0] / 32768.0 for sample in struct.iter_unpack("<h", data)]
    if sample_width == 3:
        samples: list[float] = []
        for offset in range(0, len(data), 3):
            value = int.from_bytes(data[offset : offset + 3], "little", signed=False)
            if value & 0x800000:
                value -= 1 << 24
            samples.append(value / 8388608.0)
        return samples
    if sample_width == 4:
        return [
            sample[0] / 2147483648.0
            for sample in struct.iter_unpack("<i", data)
        ]
    raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")


def analyze_wav(path: Path) -> dict[str, Any]:
    """Measure PCM WAV properties without loading the whole recording."""
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        frame_count = audio.getnframes()
        compression = audio.getcomptype()

        if compression != "NONE":
            raise ValueError(f"Unsupported WAV compression: {compression}")
        if channels < 1 or sample_rate < 1:
            raise ValueError("Invalid channel count or sample rate")
        if frame_count < 1:
            raise ValueError("Audio file contains no frames")

        duration = frame_count / sample_rate
        frames_per_window = max(1, round(sample_rate * FRAME_SECONDS))
        chunk_rms: list[float] = []
        total_samples = 0
        clipped_samples = 0
        sum_squares = 0.0
        peak = 0.0

        while True:
            frame_data = audio.readframes(frames_per_window)
            if not frame_data:
                break
            samples = _decode_samples(frame_data, sample_width)
            if len(samples) % channels:
                raise ValueError("Incomplete multi-channel PCM frame")

            total_samples += len(samples)
            clipped_samples += sum(
                abs(sample) >= CLIPPING_SAMPLE_THRESHOLD for sample in samples
            )
            sum_squares += sum(sample * sample for sample in samples)
            peak = max(peak, max(abs(sample) for sample in samples))

            chunk_rms.append(
                math.sqrt(
                    sum(sample * sample for sample in samples) / len(samples)
                )
            )

    if not chunk_rms or total_samples == 0:
        raise ValueError("Audio file contains no decodable PCM samples")

    frame_dbfs = [_dbfs(value) for value in chunk_rms]
    maximum_frame_dbfs = max(frame_dbfs)
    activity_floor_dbfs = maximum_frame_dbfs - ACTIVITY_RELATIVE_FLOOR_DB
    active_frames = [
        level > activity_floor_dbfs and level > -120.0
        for level in frame_dbfs
    ]
    active_count = sum(active_frames)

    first_active = next((i for i, active in enumerate(active_frames) if active), None)
    last_active = next(
        (i for i in range(len(active_frames) - 1, -1, -1) if active_frames[i]),
        None,
    )
    leading_silence = (
        duration
        if first_active is None
        else min(duration, first_active * FRAME_SECONDS)
    )
    trailing_silence = (
        duration
        if last_active is None
        else min(
            duration,
            max(0.0, duration - (last_active + 1) * FRAME_SECONDS),
        )
    )

    internal_runs: list[int] = []
    run_length = 0
    if first_active is not None and last_active is not None:
        for active in active_frames[first_active : last_active + 1]:
            if active:
                if run_length:
                    internal_runs.append(run_length)
                    run_length = 0
            else:
                run_length += 1
        if run_length:
            internal_runs.append(run_length)

    internal_pause_durations = [run * FRAME_SECONDS for run in internal_runs]
    rms = math.sqrt(sum_squares / total_samples)
    return {
        "audio_format": f"WAV PCM-{sample_width * 8}",
        "readable": "yes",
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bits": sample_width * 8,
        "peak_dbfs": _dbfs(peak),
        "rms_dbfs": _dbfs(rms),
        "clipping_ratio": clipped_samples / total_samples,
        "silence_ratio": 1.0 - active_count / len(active_frames),
        "leading_silence_seconds": leading_silence,
        "trailing_silence_seconds": trailing_silence,
        "speech_activity_ratio": active_count / len(active_frames),
        "speech_activity_method": (
            "25 ms non-overlapping RMS windows; active if within 35 dB "
            "of this file's loudest window; energy proxy, not a speech recognizer"
        ),
        "noise_indicator_dbfs": _percentile(frame_dbfs, 10),
        "noise_indicator_method": (
            "P10 of 25 ms RMS windows in dBFS; rough low-level energy proxy, "
            "not a measured SNR or proof of background noise"
        ),
        "long_internal_pause_count": sum(
            duration_seconds >= LONG_PAUSE_FLAG_SECONDS
            for duration_seconds in internal_pause_durations
        ),
        "longest_internal_pause_seconds": max(
            internal_pause_durations, default=0.0
        ),
    }


def _wav_inventory(root: Path) -> dict[str, Path]:
    inventory: dict[str, Path] = {}
    if root.is_dir():
        for path in sorted(root.glob("*.wav")):
            inventory[path.stem] = path
    return inventory


def inventory_originals(root: Path = BASE_DIR) -> dict[str, dict[str, Path | None]]:
    originals = _wav_inventory(root / "clean_reference")
    standardized = _wav_inventory(root / "standardized_reference")
    audio_ids = sorted(set(originals) | set(standardized))
    if not audio_ids:
        raise FileNotFoundError(
            "No original WAV files found in clean_reference/ or "
            "standardized_reference/."
        )
    return {
        audio_id: {
            "original": originals.get(audio_id) or standardized.get(audio_id),
            "standardized": standardized.get(audio_id),
        }
        for audio_id in audio_ids
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _transcript_tokens(text: str) -> list[str]:
    normalized = text.replace("\u2019", "'").replace("\u2018", "'").lower()
    return re.findall(r"[a-z0-9']+", normalized)


def _normal_training_variants(root: Path) -> dict[str, list[str]]:
    variants: dict[str, list[str]] = {}
    if not (root / LABEL_FILE.name).is_file():
        return variants
    with (root / LABEL_FILE.name).open(
        "r", newline="", encoding="utf-8-sig"
    ) as stream:
        reader = csv.DictReader(stream)
        required = {"audio_id", "source_audio", "flaw_type", "synthetic"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{LABEL_FILE.name} is missing columns: {sorted(missing)}"
            )
        for row in reader:
            if (
                row.get("synthetic", "").strip().lower() == "true"
                and row.get("flaw_type", "") == "synthetic_inserted_pause"
            ):
                source = Path(row.get("source_audio", "")).name
                if source:
                    variants.setdefault(source, []).append(row["audio_id"])
    return {source: sorted(ids) for source, ids in variants.items()}


def _flag(flags: list[str], condition: bool, code: str) -> None:
    if condition:
        flags.append(code)


def analyze_recording(
    audio_id: str,
    original_path: Path,
    standardized_path: Path | None,
    root: Path,
    training_variants: dict[str, list[str]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        field: "" for field in REPORT_FIELDS
    }
    row.update(
        {
            "audio_id": audio_id,
            "file_path": _relative_path(original_path),
            "standardized_file_path": _relative_path(standardized_path),
            "readable": "no",
            "feature_summary_available": "no",
            "feature_archive_available": "no",
            "feature_archive_valid": "",
            "feature_duration_match": "",
            "transcript_available": "no",
            "alignment_available": "no",
            "transcript_alignment_match": "",
            "normal_training_variant_ids": ";".join(
                training_variants.get(original_path.name, [])
            ),
            "training_review_needed": (
                "yes" if training_variants.get(original_path.name) else "no"
            ),
            "review_status": "PENDING_REVIEW",
            "verified_by_human": "no",
        }
    )

    flags: list[str] = []
    errors: list[str] = []

    try:
        row.update(analyze_wav(original_path))
    except (OSError, EOFError, wave.Error, ValueError, struct.error) as exc:
        errors.append(f"original_audio: {exc}")
        flags.append("original_audio_unreadable")

    standardized_metrics: dict[str, Any] = {}
    if standardized_path is None:
        flags.append("standardized_audio_missing")
    else:
        try:
            standardized_metrics = analyze_wav(standardized_path)
            row["standardized_sample_rate"] = standardized_metrics["sample_rate"]
            row["standardized_duration_seconds"] = standardized_metrics[
                "duration_seconds"
            ]
            if (
                standardized_metrics["sample_rate"]
                != EXPECTED_STANDARDIZED_SAMPLE_RATE
            ):
                flags.append("standardized_sample_rate_not_16000")
        except (OSError, EOFError, wave.Error, ValueError, struct.error) as exc:
            errors.append(f"standardized_audio: {exc}")
            flags.append("standardized_audio_unreadable")

    summary_path = root / "features" / f"{audio_id}_summary.json"
    archive_path = root / "features" / f"{audio_id}_frames.npz"
    if summary_path.is_file():
        row["feature_summary_available"] = "yes"
        try:
            summary = _load_json(summary_path)
            row["feature_sample_rate"] = summary.get("sample_rate", "")
            feature_duration = summary.get("duration_sec", "")
            row["feature_duration_seconds"] = feature_duration
            if standardized_metrics and feature_duration != "":
                duration_matches = (
                    abs(float(feature_duration) - standardized_metrics["duration_seconds"])
                    <= FEATURE_DURATION_TOLERANCE_SECONDS
                )
                row["feature_duration_match"] = "yes" if duration_matches else "no"
                if not duration_matches:
                    flags.append("feature_duration_mismatch")
            if (
                standardized_metrics
                and summary.get("sample_rate")
                != standardized_metrics["sample_rate"]
            ):
                flags.append("feature_sample_rate_mismatch")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(f"feature_summary: {exc}")
            flags.append("feature_summary_invalid")
    else:
        flags.append("feature_summary_missing")

    if archive_path.is_file():
        row["feature_archive_available"] = "yes"
        try:
            with zipfile.ZipFile(archive_path) as archive:
                corrupt_member = archive.testzip()
            row["feature_archive_valid"] = "yes" if corrupt_member is None else "no"
            if corrupt_member is not None:
                errors.append(f"feature_archive: corrupt member {corrupt_member}")
                flags.append("feature_archive_corrupt")
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            row["feature_archive_valid"] = "no"
            errors.append(f"feature_archive: {exc}")
            flags.append("feature_archive_corrupt")
    else:
        flags.append("feature_archive_missing")

    transcript_path = root / "transcripts" / f"{audio_id}.txt"
    alignment_path = root / "alignments" / f"{audio_id}.json"
    transcript_tokens: list[str] | None = None
    if transcript_path.is_file():
        row["transcript_available"] = "yes"
        try:
            transcript_tokens = _transcript_tokens(
                transcript_path.read_text(encoding="utf-8")
            )
            row["transcript_word_count"] = len(transcript_tokens)
        except (OSError, UnicodeError) as exc:
            errors.append(f"transcript: {exc}")
            flags.append("transcript_unreadable")
    else:
        flags.append("transcript_missing")

    if alignment_path.is_file():
        row["alignment_available"] = "yes"
        try:
            alignment = _load_json(alignment_path)
            words = alignment.get("words", [])
            if not isinstance(words, list):
                raise ValueError("alignment words must be a list")
            alignment_tokens = [
                token
                for word in words
                if isinstance(word, dict)
                for token in _transcript_tokens(str(word.get("word", "")))
            ]
            row["alignment_word_count"] = len(alignment_tokens)
            if transcript_tokens is not None:
                match = transcript_tokens == alignment_tokens
                row["transcript_alignment_match"] = "yes" if match else "no"
                if not match:
                    flags.append("transcript_alignment_mismatch")
            alignment_duration = alignment.get("duration")
            if alignment_duration not in (None, "") and standardized_metrics:
                if (
                    abs(
                        float(alignment_duration)
                        - standardized_metrics["duration_seconds"]
                    )
                    > FEATURE_DURATION_TOLERANCE_SECONDS
                ):
                    flags.append("alignment_duration_mismatch")
            if standardized_metrics:
                out_of_range = any(
                    not isinstance(word, dict)
                    or word.get("start") is None
                    or word.get("end") is None
                    or float(word["start"]) < 0
                    or float(word["end"]) < float(word["start"])
                    or float(word["end"])
                    > standardized_metrics["duration_seconds"]
                    + FEATURE_DURATION_TOLERANCE_SECONDS
                    for word in words
                )
                if out_of_range:
                    flags.append("alignment_timestamps_out_of_range")
        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            errors.append(f"alignment: {exc}")
            flags.append("alignment_invalid")
    else:
        flags.append("alignment_missing")

    if row["readable"] == "yes":
        duration = float(row["duration_seconds"])
        _flag(
            flags,
            duration < MIN_DURATION_FLAG_SECONDS
            or duration > MAX_DURATION_FLAG_SECONDS,
            "duration_outside_existing_2_to_600s_range",
        )
        _flag(
            flags,
            float(row["clipping_ratio"]) > CLIPPING_RATIO_FLAG,
            "clipping_sample_ratio_over_1pct",
        )
        _flag(
            flags,
            float(row["leading_silence_seconds"]) > EDGE_SILENCE_FLAG_SECONDS,
            "leading_silence_over_3s",
        )
        _flag(
            flags,
            float(row["trailing_silence_seconds"]) > EDGE_SILENCE_FLAG_SECONDS,
            "trailing_silence_over_3s",
        )
        _flag(
            flags,
            float(row["speech_activity_ratio"]) < LOW_ACTIVITY_RATIO_FLAG,
            "energy_activity_under_20pct",
        )
        _flag(
            flags,
            float(row["long_internal_pause_count"]) > 0,
            "internal_pause_at_least_3s_review_naturalness",
        )
        _flag(
            flags,
            float(row["rms_dbfs"]) < LOW_RMS_FLAG_DBFS,
            "overall_rms_below_minus_35_dbfs",
        )

    row["automated_flags"] = ";".join(dict.fromkeys(flags))
    row["file_errors"] = ";".join(errors)
    return row


def _read_existing_report(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding=REPORT_ENCODING) as stream:
        reader = csv.DictReader(stream)
        required = {"audio_id", *HUMAN_FIELDS}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Existing report {path} lacks review columns "
                f"{sorted(missing)}; refusing to overwrite it."
            )
        saved: dict[str, dict[str, str]] = {}
        for row in reader:
            audio_id = row.get("audio_id", "")
            if not audio_id or audio_id in saved:
                raise ValueError(
                    f"Existing report {path} has a blank or duplicate audio_id."
                )
            status = row.get("review_status", "")
            if status not in REVIEW_STATUSES:
                raise ValueError(
                    f"Existing report has unsupported review status {status!r} "
                    f"for {audio_id}; refusing to overwrite it."
                )
            saved[audio_id] = row
        return saved


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("w", newline="", encoding=REPORT_ENCODING) as stream:
            writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def generate_report(
    root: Path = BASE_DIR, report_path: Path = REPORT_FILE
) -> list[dict[str, Any]]:
    existing = _read_existing_report(report_path)
    recordings = inventory_originals(root)
    training_variants = _normal_training_variants(root)
    rows = []

    for audio_id, paths in recordings.items():
        original_path = paths["original"]
        if original_path is None:
            continue
        row = analyze_recording(
            audio_id,
            original_path,
            paths["standardized"],
            root,
            training_variants,
        )
        previous = existing.get(audio_id, {})
        if previous:
            for field in HUMAN_FIELDS:
                value = previous.get(field, "")
                if value != "":
                    row[field] = value
            row["verified_by_human"] = (
                "yes"
                if row["review_status"] in ("VERIFIED_GOOD", "VERIFIED_FLAWED")
                else "no"
            )
        rows.append(row)

    _write_report(report_path, rows)
    return rows


def update_review(
    audio_id: str,
    status: str,
    notes: str | None,
    reviewer: str | None,
    flaw_type: str | None,
    flaw_start: str | None,
    flaw_end: str | None,
    report_path: Path = REPORT_FILE,
) -> list[dict[str, str]]:
    if status not in REVIEW_STATUSES:
        raise ValueError(f"Unsupported review status: {status}")
    if status != "VERIFIED_FLAWED" and any(
        value is not None for value in (flaw_type, flaw_start, flaw_end)
    ):
        raise ValueError(
            "Flaw type and timestamps may only be set for VERIFIED_FLAWED."
        )
    if (flaw_start is None) != (flaw_end is None):
        raise ValueError("Provide both --flaw-start and --flaw-end, or neither.")
    if flaw_start is not None:
        start = float(flaw_start)
        end = float(flaw_end or "")
        if start < 0 or end < start:
            raise ValueError("Flaw timestamps must be nonnegative and end >= start.")

    with report_path.open("r", newline="", encoding=REPORT_ENCODING) as stream:
        reader = csv.DictReader(stream)
        if not set(REPORT_FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError(
                f"{report_path} does not have the expected current report schema. "
                "Regenerate it first, preserving this file before any manual edits."
            )
        rows = list(reader)

    matches = [row for row in rows if row.get("audio_id") == audio_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one report row for {audio_id!r}; found {len(matches)}."
        )
    row = matches[0]
    if flaw_end is not None and row.get("duration_seconds"):
        if float(flaw_end) > float(row["duration_seconds"]):
            raise ValueError(
                f"Flaw end {flaw_end}s exceeds {audio_id}'s duration "
                f"({row['duration_seconds']}s)."
            )
    row["review_status"] = status
    row["verified_by_human"] = (
        "yes" if status in ("VERIFIED_GOOD", "VERIFIED_FLAWED") else "no"
    )
    row["reviewed_by"] = reviewer if reviewer is not None else row["reviewed_by"]
    row["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if notes is not None:
        row["review_notes"] = notes
    if status == "VERIFIED_FLAWED":
        if flaw_type is not None:
            row["flaw_type"] = flaw_type
        if flaw_start is not None:
            row["flaw_start_seconds"] = flaw_start
            row["flaw_end_seconds"] = flaw_end or ""
    else:
        row["flaw_type"] = ""
        row["flaw_start_seconds"] = ""
        row["flaw_end_seconds"] = ""
    _write_report(report_path, rows)
    return rows


def launch_player(audio_id: str, root: Path = BASE_DIR) -> None:
    paths = inventory_originals(root)
    if audio_id not in paths:
        raise ValueError(
            f"Unknown audio_id {audio_id!r}. Available IDs: "
            + ", ".join(paths)
        )
    path = paths[audio_id]["original"]
    if path is None:
        raise FileNotFoundError(f"No original audio path exists for {audio_id}.")
    startfile = getattr(os, "startfile", None)
    if callable(startfile):
        startfile(path)
    else:
        player = (
            shutil.which("ffplay")
            or shutil.which("xdg-open")
            or shutil.which("open")
        )
        if player is None:
            raise RuntimeError(
                "No audio player found. Install ffplay or use a desktop file manager."
            )
        command = (
            [player, "-nodisp", "-autoexit", str(path)]
            if Path(player).name == "ffplay"
            else [player, str(path)]
        )
        os.spawnvp(os.P_NOWAIT, player, command)
    print(f"Opened {path}")


def _training_frame_audit(
    report_rows: list[dict[str, Any]], training_path: Path = TRAINING_FILE
) -> None:
    if not training_path.is_file():
        print(f"Training-frame audit unavailable: {training_path} not found.")
        return
    by_source: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    total_rows = 0
    with training_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"audio_id", "source_audio", "target"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{training_path.name} is missing columns: {sorted(missing)}"
            )
        for row in reader:
            total_rows += 1
            target = row.get("target", "")
            counts[target] = counts.get(target, 0) + 1
            if target == "normal":
                source = Path(row.get("source_audio", "")).name
                if source:
                    by_source.setdefault(source, set()).add(row.get("audio_id", ""))

    statuses = {row["audio_id"]: row["review_status"] for row in report_rows}
    unverified = sorted(
        source
        for source in by_source
        if statuses.get(Path(source).stem, "PENDING_REVIEW") != "VERIFIED_GOOD"
    )
    print("\nTRAINING-FRAME DATASET AUDIT")
    print(f"Frame rows scanned: {total_rows}")
    print(f"Target row counts: {counts}")
    print(
        "Sources with normal frames: "
        + (", ".join(sorted(by_source)) if by_source else "none")
    )
    print(
        "Unverified sources contributing normal frames: "
        + (", ".join(unverified) if unverified else "none")
    )
    print(
        "Normal-label audio variants: "
        + str(sum(len(ids) for ids in by_source.values()))
    )


def _print_summary(rows: list[dict[str, Any]], report_path: Path) -> None:
    status_counts = {status: 0 for status in REVIEW_STATUSES}
    flagged = 0
    invalid = 0
    for row in rows:
        status_counts[row["review_status"]] = (
            status_counts.get(row["review_status"], 0) + 1
        )
        flagged += bool(row["automated_flags"])
        invalid += row["readable"] != "yes"

    print(f"Original recordings found: {len(rows)}")
    print(f"Verified good by a human: {status_counts['VERIFIED_GOOD']}")
    print(f"Verified flawed by a human: {status_counts['VERIFIED_FLAWED']}")
    print(
        "Pending review: "
        f"{status_counts['PENDING_REVIEW']}; uncertain: "
        f"{status_counts['UNCERTAIN']}"
    )
    print(f"Recordings with one or more automated review flags: {flagged}")
    print(f"Missing/unreadable original files: {invalid}")
    safe = [
        row["audio_id"]
        for row in rows
        if row["review_status"] == "VERIFIED_GOOD"
        and row["verified_by_human"] == "yes"
    ]
    print(
        "Human-verified baseline candidates: "
        + (", ".join(safe) if safe else "none")
    )
    unverified_training_sources = [
        row["audio_id"]
        for row in rows
        if row["training_review_needed"] == "yes"
        and row["review_status"] != "VERIFIED_GOOD"
    ]
    print(
        "Sources whose synthetic-pause normal training labels need review: "
        + (
            ", ".join(unverified_training_sources)
            if unverified_training_sources
            else "none"
        )
    )
    print(f"Report: {_relative_path(report_path)}")
    if not safe:
        print(
            "No recording is established as genuinely clean until a human "
            "listens and records VERIFIED_GOOD."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure original PCM WAVs, preserve human review decisions, "
            "and write original_audio_review.csv."
        )
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--listen", metavar="AUDIO_ID")
    action.add_argument("--review", metavar="AUDIO_ID")
    parser.add_argument(
        "--status", choices=REVIEW_STATUSES, help="Required with --review."
    )
    parser.add_argument("--notes", help="Human listening notes.")
    parser.add_argument("--reviewed-by", help="Reviewer name or initials.")
    parser.add_argument("--flaw-type", help="Human-identified flaw category.")
    parser.add_argument("--flaw-start", help="Approximate flaw start in seconds.")
    parser.add_argument("--flaw-end", help="Approximate flaw end in seconds.")
    parser.add_argument(
        "--audit-training-frames",
        action="store_true",
        help="Stream the full training_frames.csv to audit source/target counts.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.listen:
            if any(
                value is not None
                for value in (
                    args.status,
                    args.notes,
                    args.reviewed_by,
                    args.flaw_type,
                    args.flaw_start,
                    args.flaw_end,
                )
            ):
                parser.error("Review options cannot be used together with --listen.")
            launch_player(args.listen)
            return 0

        if args.review:
            if args.status is None:
                parser.error("--review requires --status.")
            rows = update_review(
                args.review,
                args.status,
                args.notes,
                args.reviewed_by,
                args.flaw_type,
                args.flaw_start,
                args.flaw_end,
            )
            print(f"Updated human review for {args.review}: {args.status}")
            _print_summary(rows, REPORT_FILE)
            return 0

        if any(
            value is not None
            for value in (
                args.status,
                args.notes,
                args.reviewed_by,
                args.flaw_type,
                args.flaw_start,
                args.flaw_end,
            )
        ):
            parser.error("Review options require --review AUDIO_ID.")

        rows = generate_report()
        _print_summary(rows, REPORT_FILE)
        if args.audit_training_frames:
            _training_frame_audit(rows)
        return 0
    except (OSError, ValueError, RuntimeError, wave.Error) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
