"""Fit a multi-recording reference anomaly model and create review artifacts.

The model ranks deviations from the selected baseline. It does not predict
whether an alignment is correct because this project has no verified error
labels.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


BASE_DIR = Path(__file__).resolve().parent
BASELINE_ID = "good_15"
FEATURE_DIR = BASE_DIR / "features"
ALIGNMENT_DIR = BASE_DIR / "alignments"
AUDIO_DIR = BASE_DIR / "standardized_reference"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
OUTPUT_DIR = BASE_DIR / "reports"
MODEL_PATH = OUTPUT_DIR / "timestamp_review_baseline_model.joblib"
REPORT_PATH = OUTPUT_DIR / "timestamp_review_baseline.json"
DASHBOARD_PATH = OUTPUT_DIR / "timestamp_review_dashboard.html"
RANDOM_SEED = 20261013
BOUNDARY_WINDOW_SECONDS = 0.1
CONTEXT_WINDOW_SECONDS = 0.15
BASELINE_REVIEW_PERCENTILE = 5.0

FEATURE_NAMES = [
    "alignment_score",
    "log_word_duration",
    "gap_before_seconds",
    "gap_after_seconds",
    "previous_alignment_score",
    "next_alignment_score",
    "word_rms_db_median",
    "word_voiced_fraction",
    "start_boundary_rms_change_db",
    "end_boundary_rms_change_db",
    "start_boundary_voiced_change",
    "end_boundary_voiced_change",
    "start_boundary_centroid_change_hz",
    "end_boundary_centroid_change_hz",
    "word_zero_crossing_rate_median",
]


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _recording_ids() -> list[str]:
    return sorted(
        path.stem
        for path in ALIGNMENT_DIR.glob("*.json")
        if (FEATURE_DIR / f"{path.stem}_frames.npz").is_file()
        and (AUDIO_DIR / f"{path.stem}.wav").is_file()
    )


def _downsample_series(
    times: np.ndarray, values: np.ndarray, max_points: int = 900
) -> list[dict[str, float | None]]:
    indices = np.linspace(
        0, max(0, times.size - 1), min(times.size, max_points), dtype=int
    )
    return [
        {
            "time": float(times[index]),
            "value": (
                float(values[index])
                if np.isfinite(values[index])
                else None
            ),
        }
        for index in indices
    ]


def _audio_visual_data(
    audio_path: Path, max_waveform_points: int = 900
) -> dict[str, Any]:
    with wave.open(str(audio_path), "rb") as audio:
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        sample_rate = audio.getframerate()
        frame_count = audio.getnframes()
        if sample_width != 2:
            raise ValueError(
                f"Expected 16-bit PCM audio for visualization: {audio_path}"
            )
        samples = np.frombuffer(audio.readframes(frame_count), dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.float32).mean(axis=1)
    samples = samples.astype(np.float32) / 32768.0
    duration = samples.size / sample_rate
    peak = max(float(np.max(np.abs(samples))), 1e-8)
    waveform_points = min(max_waveform_points, samples.size)
    edges = np.linspace(0, samples.size, waveform_points + 1, dtype=int)
    waveform = [
        [
            round(float(np.min(samples[edges[index] : edges[index + 1]]) / peak), 4),
            round(float(np.max(samples[edges[index] : edges[index + 1]]) / peak), 4),
        ]
        for index in range(waveform_points)
        if edges[index + 1] > edges[index]
    ]

    window_size = 512
    hop_size = max(1, sample_rate // 4)
    starts = np.arange(0, max(samples.size, 1), hop_size, dtype=np.int64)
    window = np.hanning(window_size).astype(np.float32)
    spectrogram = np.zeros((64, starts.size), dtype=np.uint8)
    for column, start in enumerate(starts):
        frame = samples[start : start + window_size]
        if frame.size < window_size:
            frame = np.pad(frame, (0, window_size - frame.size))
        magnitude = np.abs(np.fft.rfft(frame * window))
        grouped = magnitude[:256].reshape(64, 4).max(axis=1)
        decibels = 20 * np.log10(np.maximum(grouped / (window.sum() / 2), 1e-7))
        spectrogram[:, column] = np.clip((decibels + 70) * (255 / 70), 0, 255)

    return {
        "duration_seconds": duration,
        "waveform_minmax": waveform,
        "spectrogram": {
            "height": 64,
            "width": int(starts.size),
            "sample_rate": sample_rate,
            "nyquist_hz": sample_rate / 2,
            "png_data": base64.b64encode(spectrogram.tobytes()).decode("ascii"),
        },
    }


def _recording_analysis(recording_id: str, words: list[dict[str, Any]]) -> dict[str, Any]:
    summary_path = FEATURE_DIR / f"{recording_id}_summary.json"
    archive_path = FEATURE_DIR / f"{recording_id}_frames.npz"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        times = np.asarray(archive["time"], dtype=np.float64)
        series = {
            key: _downsample_series(times, np.asarray(archive[key], dtype=np.float64))
            for key in (
                "rms_db",
                "f0_hz",
                "spectral_centroid_hz",
                "voiced_probability",
            )
            if key in archive.files
        }
    pauses = summary.get("pauses", {})
    speech_rate = summary.get("speech_rate", {})
    pitch = summary.get("pitch", {}).get("f0_hz", {})
    durations = [
        word["duration_seconds"]
        for word in words
        if word["duration_seconds"] is not None
    ]
    gaps = [
        word["gap_before_seconds"]
        for word in words
        if word["gap_before_seconds"] is not None
    ]
    return {
        "duration_seconds": float(summary.get("duration_sec", times[-1] if times.size else 0)),
        "word_count": len(words),
        "mean_word_duration_seconds": _mean(np.asarray(durations)) if durations else None,
        "median_gap_seconds": float(np.median(gaps)) if gaps else None,
        "alignment_score_mean": _mean(
            np.asarray(
                [
                    word["alignment_score"]
                    for word in words
                    if word["alignment_score"] is not None
                ]
            )
        )
        if any(word["alignment_score"] is not None for word in words)
        else None,
        "alignment_score_missing": sum(
            word["alignment_score"] is None for word in words
        ),
        "global_wpm": _finite_number(speech_rate.get("global", {}).get("words_per_minute")),
        "local_10sec": speech_rate.get("local_10sec", []),
        "pitch_mean_hz": _finite_number(pitch.get("mean")),
        "pitch_std_hz": _finite_number(pitch.get("std")),
        "pauses": pauses.get("acoustic_intervals", []),
        "pause_summary": pauses.get("acoustic_statistics", {}),
        "pause_threshold_db": _finite_number(pauses.get("pause_threshold_db")),
        "pause_method": pauses.get("method", "unavailable"),
        "feature_series": series,
        "acoustic_summary": {
            "pitch": pitch,
            "energy_db": summary.get("energy", {}).get("rms_db", {}),
            "spectral_centroid": summary.get("spectral", {}).get("centroid_hz", {}),
        },
    }


def _frame_window(
    times: np.ndarray, values: np.ndarray, start: float, end: float
) -> np.ndarray:
    mask = (times >= start) & (times < end)
    return values[mask]


def _boundary_change(
    times: np.ndarray,
    values: np.ndarray,
    boundary: float,
    window: float = BOUNDARY_WINDOW_SECONDS,
) -> float:
    before = _median(_frame_window(times, values, boundary - window, boundary))
    after = _median(_frame_window(times, values, boundary, boundary + window))
    if not math.isfinite(before) or not math.isfinite(after):
        return float("nan")
    return after - before


def extract_recording_features(
    recording_id: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    alignment_path = ALIGNMENT_DIR / f"{recording_id}.json"
    archive_path = FEATURE_DIR / f"{recording_id}_frames.npz"
    if not alignment_path.is_file():
        raise FileNotFoundError(f"Alignment not found: {alignment_path}")
    if not archive_path.is_file():
        raise FileNotFoundError(f"Frame feature archive not found: {archive_path}")

    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    words = alignment.get("words")
    if not isinstance(words, list) or not words:
        raise ValueError(f"Alignment has no word entries: {alignment_path}")

    with np.load(archive_path, allow_pickle=False) as archive:
        required = {
            "time",
            "rms_db",
            "voiced_probability",
            "spectral_centroid_hz",
            "zero_crossing_rate",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                f"{archive_path} is missing acoustic features: {sorted(missing)}"
            )
        times = np.asarray(archive["time"], dtype=np.float64)
        rms_db = np.asarray(archive["rms_db"], dtype=np.float64)
        voiced_probability = np.asarray(
            archive["voiced_probability"], dtype=np.float64
        )
        centroid = np.asarray(
            archive["spectral_centroid_hz"], dtype=np.float64
        )
        zcr = np.asarray(archive["zero_crossing_rate"], dtype=np.float64)
        lengths = {
            times.size,
            rms_db.size,
            voiced_probability.size,
            centroid.size,
            zcr.size,
        }
        if len(lengths) != 1 or times.size == 0:
            raise ValueError(f"Frame feature arrays are not aligned in {archive_path}")

    records: list[dict[str, Any]] = []
    feature_rows: list[list[float]] = []
    for index, word in enumerate(words):
        if not isinstance(word, dict):
            raise ValueError(
                f"Alignment entry {index} in {alignment_path} is not an object."
            )
        start = _finite_number(word.get("start"))
        end = _finite_number(word.get("end"))
        score = _finite_number(word.get("score"))
        duration = (
            end - start
            if start is not None and end is not None and end > start
            else float("nan")
        )
        previous = words[index - 1] if index else {}
        following = words[index + 1] if index + 1 < len(words) else {}
        previous_end = _finite_number(previous.get("end"))
        next_start = _finite_number(following.get("start"))
        previous_score = _finite_number(previous.get("score"))
        next_score = _finite_number(following.get("score"))
        gap_before = (
            start - previous_end
            if start is not None and previous_end is not None
            else float("nan")
        )
        gap_after = (
            next_start - end
            if next_start is not None and end is not None
            else float("nan")
        )

        if start is None or end is None or end <= start:
            word_rms = word_voiced = word_zcr = float("nan")
            start_rms_change = end_rms_change = float("nan")
            start_voiced_change = end_voiced_change = float("nan")
            start_centroid_change = end_centroid_change = float("nan")
        else:
            word_rms = _median(_frame_window(times, rms_db, start, end))
            word_voiced = _mean(
                _frame_window(times, voiced_probability, start, end)
            )
            word_zcr = _median(_frame_window(times, zcr, start, end))
            start_rms_change = _boundary_change(times, rms_db, start)
            end_rms_change = _boundary_change(times, rms_db, end)
            start_voiced_change = _boundary_change(
                times, voiced_probability, start
            )
            end_voiced_change = _boundary_change(times, voiced_probability, end)
            start_centroid_change = _boundary_change(times, centroid, start)
            end_centroid_change = _boundary_change(times, centroid, end)

        row = [
            score if score is not None else float("nan"),
            math.log(max(duration, 1e-4))
            if math.isfinite(duration)
            else float("nan"),
            gap_before,
            gap_after,
            previous_score if previous_score is not None else float("nan"),
            next_score if next_score is not None else float("nan"),
            word_rms,
            word_voiced,
            start_rms_change,
            end_rms_change,
            start_voiced_change,
            end_voiced_change,
            start_centroid_change,
            end_centroid_change,
            word_zcr,
        ]
        feature_rows.append(row)
        records.append(
            {
                "recording_id": recording_id,
                "word_index": word.get("index", index),
                "word": str(word.get("word", "")),
                "start": start,
                "end": end,
                "duration_seconds": duration
                if math.isfinite(duration)
                else None,
                "alignment_score": score,
                "gap_before_seconds": gap_before
                if math.isfinite(gap_before)
                else None,
                "gap_after_seconds": gap_after
                if math.isfinite(gap_after)
                else None,
                "word_rms_db_median": (
                    word_rms if math.isfinite(word_rms) else None
                ),
                "word_voiced_fraction": (
                    word_voiced if math.isfinite(word_voiced) else None
                ),
                "word_zero_crossing_rate_median": (
                    word_zcr if math.isfinite(word_zcr) else None
                ),
                "start_boundary_rms_change_db": (
                    start_rms_change if math.isfinite(start_rms_change) else None
                ),
                "end_boundary_rms_change_db": (
                    end_rms_change if math.isfinite(end_rms_change) else None
                ),
                "start_boundary_voiced_change": (
                    start_voiced_change
                    if math.isfinite(start_voiced_change)
                    else None
                ),
                "end_boundary_voiced_change": (
                    end_voiced_change
                    if math.isfinite(end_voiced_change)
                    else None
                ),
                "start_boundary_centroid_change_hz": (
                    start_centroid_change
                    if math.isfinite(start_centroid_change)
                    else None
                ),
                "end_boundary_centroid_change_hz": (
                    end_centroid_change
                    if math.isfinite(end_centroid_change)
                    else None
                ),
            }
        )

    matrix = np.asarray(feature_rows, dtype=np.float64)
    return records, matrix


def _resolve_threshold(model: Any, baseline_matrix: np.ndarray) -> float:
    transformed = model.named_steps["simpleimputer"].transform(baseline_matrix)
    scores = model.named_steps["isolationforest"].score_samples(transformed)
    return float(np.percentile(scores, BASELINE_REVIEW_PERCENTILE))


def train_and_score() -> dict[str, Any]:
    recording_ids = _recording_ids()
    if BASELINE_ID not in recording_ids:
        raise FileNotFoundError(
            f"Baseline {BASELINE_ID} requires matching alignment, features, and WAV."
        )
    if not recording_ids:
        raise ValueError("No recordings are available to fit the reference model.")
    extracted = {
        recording_id: extract_recording_features(recording_id)
        for recording_id in recording_ids
    }
    baseline_records = [
        record
        for recording_id in recording_ids
        for record in extracted[recording_id][0]
    ]
    baseline_matrix = np.vstack(
        [extracted[recording_id][1] for recording_id in recording_ids]
    )
    if len(baseline_records) < 20:
        raise ValueError(
            "At least 20 valid word examples are required to fit the reference model."
        )
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        IsolationForest(
            n_estimators=300,
            max_samples=min(256, len(baseline_records)),
            contamination="auto",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    )
    model.fit(baseline_matrix)
    baseline_threshold = _resolve_threshold(model, baseline_matrix)

    recordings = []
    for recording_id in recording_ids:
        records, matrix = extracted[recording_id]
        transformed = model.named_steps["simpleimputer"].transform(matrix)
        model_scores = model.named_steps["isolationforest"].score_samples(
            transformed
        )
        for record, score in zip(records, model_scores):
            record["baseline_anomaly_score"] = float(score)
            record["baseline_score_threshold"] = baseline_threshold
            record["model_review_candidate"] = bool(score <= baseline_threshold)
            record["review_cues"] = []
            if (
                record["alignment_score"] is not None
                and record["alignment_score"] < 0.5
            ):
                record["review_cues"].append("low_alignment_score_below_0.50")
            if (
                record["duration_seconds"] is not None
                and record["duration_seconds"] > 1.0
            ):
                record["review_cues"].append("word_duration_over_1_second")
            if (
                record["gap_before_seconds"] is not None
                and record["gap_before_seconds"] > 0.5
            ):
                record["review_cues"].append("gap_before_over_0.5_seconds")
            if record["model_review_candidate"]:
                record["review_cues"].append("unusual_vs_reference_corpus")
        analysis = _recording_analysis(recording_id, records)
        audio_data = _audio_visual_data(AUDIO_DIR / f"{recording_id}.wav")
        recordings.append(
            {
                "recording_id": recording_id,
                "is_baseline": recording_id == BASELINE_ID,
                "audio_path": f"../standardized_reference/{recording_id}.wav",
                "transcript_path": f"../transcripts/{recording_id}.txt",
                "transcript": (
                    (TRANSCRIPT_DIR / f"{recording_id}.txt").read_text(
                        encoding="utf-8"
                    )
                    if (TRANSCRIPT_DIR / f"{recording_id}.txt").is_file()
                    else ""
                ),
                "words": records,
                "analysis": analysis,
                "audio_visual": audio_data,
                "review_candidate_count": sum(
                    record["model_review_candidate"] for record in records
                ),
                "heuristic_cue_count": sum(
                    bool(record["review_cues"]) for record in records
                ),
            }
        )

    total_words = sum(recording["analysis"]["word_count"] for recording in recordings)
    total_duration = sum(
        recording["analysis"]["duration_seconds"] for recording in recordings
    )
    total_pauses = sum(
        len(recording["analysis"]["pauses"]) for recording in recordings
    )
    total_pause_duration = sum(
        _finite_number(interval.get("duration")) or 0.0
        for recording in recordings
        for interval in recording["analysis"]["pauses"]
    )
    missing_alignment_scores = sum(
        recording["analysis"]["alignment_score_missing"]
        for recording in recordings
    )
    total_candidates = sum(
        recording["review_candidate_count"] for recording in recordings
    )
    comparison_ids = [
        recording_id
        for recording_id in ("good_01", "good_04")
        if recording_id in recording_ids
    ]

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": "IsolationForest multi-recording reference deviation model",
        "model_version": "reference-corpus-word-boundary-v3",
        "model_scope": (
            "Fit on word-level examples from every recording with matching "
            "audio, alignment, and acoustic features; scores rank deviations "
            "from the combined reference-corpus distribution."
        ),
        "prediction_meaning": (
            "A candidate means unusual relative to the combined reference "
            "corpus, not an incorrect timestamp. Scores are not probabilities."
        ),
        "baseline_recording_id": BASELINE_ID,
        "baseline_training_recording_ids": recording_ids,
        "comparison_recording_ids": comparison_ids,
        "recording_ids": recording_ids,
        "baseline_training_words": len(baseline_records),
        "feature_names": FEATURE_NAMES,
        "baseline_review_percentile": BASELINE_REVIEW_PERCENTILE,
        "baseline_score_threshold": baseline_threshold,
        "random_seed": RANDOM_SEED,
        "supervised_validation": {
            "available": False,
            "reason": (
                "No manually verified correct/incorrect word-timestamp labels "
                "are present; accuracy and precision cannot be measured. The "
                "source recordings also remain pending human review."
            ),
        },
        "dataset": {
            "recording_count": len(recordings),
            "total_duration_seconds": total_duration,
            "total_word_count": total_words,
            "review_candidate_count": total_candidates,
            "acoustic_pause_count": total_pauses,
            "acoustic_pause_duration_seconds": total_pause_duration,
            "missing_alignment_score_count": missing_alignment_scores,
            "alignment_score_count": total_words - missing_alignment_scores,
        },
        "recordings": recordings,
    }, model


def _energy_envelope(
    archive_path: Path, max_points: int = 600
) -> list[float]:
    with np.load(archive_path, allow_pickle=False) as archive:
        rms = np.asarray(archive["rms"], dtype=np.float64)
    if rms.size <= max_points:
        sampled = rms
    else:
        edges = np.linspace(0, rms.size, max_points + 1, dtype=int)
        sampled = np.asarray(
            [
                np.max(rms[edges[index] : edges[index + 1]])
                for index in range(max_points)
            ],
            dtype=np.float64,
        )
    peak = float(np.max(sampled)) if sampled.size else 0.0
    if peak <= 0 or not math.isfinite(peak):
        return [0.0 for _ in sampled]
    return [
        float(value / peak) if math.isfinite(float(value)) else 0.0
        for value in sampled
    ]


def _render_dashboard_legacy(report: dict[str, Any]) -> str:
    cards = []
    for recording in report["recordings"]:
        recording_id = html.escape(recording["recording_id"])
        audio_url = html.escape(recording["audio_path"], quote=True)
        transcript_path = BASE_DIR / "transcripts" / f"{recording_id}.txt"
        transcript = (
            html.escape(transcript_path.read_text(encoding="utf-8"))
            if transcript_path.is_file()
            else "Transcript unavailable"
        )
        words = sorted(
            recording["words"],
            key=lambda item: (
                item["model_review_candidate"],
                -(item["baseline_anomaly_score"]),
            ),
            reverse=True,
        )
        rows = []
        for word in words:
            cues = ", ".join(word["review_cues"]) or "No configured review cues"
            start = word["start"]
            end = word["end"]
            rows.append(
                "<tr class='candidate' "
                f"data-start='{'' if start is None else f'{start:.3f}'}'>"
                f"<td>{html.escape(str(word['word_index']))}</td>"
                f"<td>{html.escape(word['word'])}</td>"
                f"<td>{'' if start is None else f'{start:.3f}'}</td>"
                f"<td>{'' if end is None else f'{end:.3f}'}</td>"
                f"<td>{'' if word['duration_seconds'] is None else f'{word['duration_seconds']:.3f}'}</td>"
                f"<td>{'' if word['alignment_score'] is None else f'{word['alignment_score']:.3f}'}</td>"
                f"<td>{word['baseline_anomaly_score']:.4f}</td>"
                f"<td>{'Review candidate' if word['model_review_candidate'] else 'Within reference range'}</td>"
                f"<td>{html.escape(cues)}</td></tr>"
            )
        words_for_timeline = [
            word
            for word in recording["words"]
            if word["start"] is not None and word["end"] is not None
        ]
        audio_duration = max(
            (word["end"] for word in words_for_timeline), default=1.0
        )
        markers = []
        for word in words_for_timeline:
            left = max(0.0, min(100.0, word["start"] / audio_duration * 100))
            width = max(
                0.08,
                min(100.0 - left, (word["end"] - word["start"]) / audio_duration * 100),
            )
            markers.append(
                f'<button class="timeline-word {"flagged" if word["model_review_candidate"] else ""}" '
                f'data-start="{word["start"]:.3f}" '
                f'title="{html.escape(word["word"], quote=True)} · '
                f'{word["start"]:.3f}–{word["end"]:.3f}s" '
                f'aria-label="Seek to {html.escape(word["word"], quote=True)} at '
                f'{word["start"]:.3f} seconds" '
                f'style="left:{left:.4f}%;width:{width:.4f}%"></button>'
            )
        envelope = recording["energy_envelope"]
        waveform_points = " ".join(
            f"{(index / max(1, len(envelope) - 1) * 1000):.2f},"
            f"{38 - value * 34:.2f} "
            f"{(index / max(1, len(envelope) - 1) * 1000):.2f},"
            f"{38 + value * 34:.2f}"
            for index, value in enumerate(envelope)
        )
        baseline_tag = (
            "Baseline · human selected"
            if recording["is_baseline"]
            else "Comparison recording"
        )
        cards.append(
            f"""
            <section class="recording {'baseline' if recording['is_baseline'] else ''}">
              <header><div><span class="eyebrow">{baseline_tag}</span>
                <h2>{recording_id}</h2></div>
                <div class="count"><strong>{recording['review_candidate_count']}</strong>
                  <span>model review candidates</span></div></header>
              <audio controls preload="metadata" src="{audio_url}"></audio>
              <h3>Acoustic energy envelope</h3>
              <svg class="waveform" viewBox="0 0 1000 76" preserveAspectRatio="none"
                   role="img" aria-label="Normalized short-time RMS energy envelope">
                <polyline points="{waveform_points}" fill="none" stroke="#3478a5"
                          stroke-width="0.8"/>
              </svg>
              <h3>Word timeline <small>· green marks are model review candidates</small></h3>
              <div class="timeline" role="group" aria-label="Word timestamps">
                {''.join(markers)}
              </div>
              <div class="table-wrap"><table>
                <thead><tr><th>Word #</th><th>Word</th><th>Start s</th><th>End s</th>
                <th>Duration s</th><th>Align score</th><th>Deviation score</th>
                <th>Model signal</th><th>Review cues</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
              </table></div>
              <details><summary>Show original transcript</summary><pre>{transcript}</pre></details>
            </section>"""
        )

    serialized = json.dumps(report, ensure_ascii=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Baseline timestamp review</title>
<style>
:root{{--ink:#162635;--muted:#627386;--line:#d9e2eb;--paper:#f3f6fa;--white:#fff;--blue:#276b9b;--mint:#dff4e8;--amber:#fff1ce}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,sans-serif}}
main{{max-width:1440px;margin:auto;padding:28px}}h1{{font-size:clamp(1.8rem,4vw,2.8rem);margin:.3rem 0}}h2{{margin:.2rem 0}}
.eyebrow{{color:var(--blue);font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700}}
.intro,.recording{{background:var(--white);border:1px solid var(--line);border-radius:14px;padding:20px;margin:16px 0}}
.intro{{border-left:5px solid var(--blue)}}.notice{{background:var(--amber);padding:12px;border-radius:8px}}
.recording.baseline{{border-top:4px solid #42845c}}header{{display:flex;justify-content:space-between;gap:16px;align-items:center}}
.count{{display:flex;gap:10px;align-items:center;color:var(--muted)}}.count strong{{font-size:1.7rem;color:var(--ink)}}
audio{{width:100%;margin:14px 0}}.table-wrap{{max-height:560px;overflow:auto;border:1px solid var(--line);border-radius:8px}}
.waveform{{width:100%;height:76px;background:#f3f7fa;border:1px solid var(--line);border-radius:6px}}
.timeline{{position:relative;height:44px;margin:8px 0 18px;background:#edf2f6;border:1px solid var(--line);border-radius:5px;overflow:hidden}}
.timeline-word{{position:absolute;top:6px;height:30px;min-width:1px;padding:0;border:0;background:#86a8c2;cursor:pointer}}
.timeline-word.flagged{{background:#4a9a69}}.timeline-word:hover,.timeline-word:focus{{z-index:2;outline:2px solid #16384e}}
table{{border-collapse:collapse;width:100%;font-size:.86rem;white-space:nowrap}}th{{position:sticky;top:0;background:#eaf0f6;text-align:left}}
th,td{{padding:8px 10px;border-bottom:1px solid #e6ebf0}}tr:hover{{background:#f0f6fb}}
tr[data-start]{{cursor:pointer}}details{{margin-top:14px}}pre{{white-space:pre-wrap;max-height:320px;overflow:auto;background:#f7f9fb;padding:14px;border-radius:8px}}
footer{{color:var(--muted);font-size:.9rem;padding:14px 0}}@media(max-width:700px){{main{{padding:12px}}header{{align-items:flex-start;flex-direction:column}}}}
</style></head><body><main>
<p class="eyebrow">Track C · temporal grounding</p><h1>Word timestamp review</h1>
<section class="intro"><h2>Reference model trained across source recordings</h2>
<p>This unsupervised Isolation Forest learns the feature distribution of all
available source recordings ({report['baseline_training_words']} word intervals
across {len(report['baseline_training_recording_ids'])} recordings).
It ranks words by deviation from that combined reference corpus.</p>
<div class="notice"><strong>Interpretation:</strong> a model candidate is unusual
relative to the reference corpus; it is not a prediction that the timestamp is
wrong. The baseline alignment itself is not ground truth. Scores are not
probabilities; no accuracy or precision is available without manually verified
timestamp labels.</div>
<p>Reference {report['baseline_review_percentile']:g}th-percentile score threshold:
{report['baseline_score_threshold']:.4f}. Features include aligner score,
word duration, neighboring gaps/scores, and local RMS, voicing, spectral-centroid,
and zero-crossing changes at word boundaries. Select a table row to seek the audio.</p>
</section>
{''.join(cards)}
<footer>Original audio, transcript, and alignment files are read-only inputs.
The model output is a review-prioritization signal, not a correction or a
good/bad recording classifier. Generated {html.escape(report['created_at_utc'])}.</footer>
<script id="model-report" type="application/json">{serialized}</script>
<script>
for(const section of document.querySelectorAll(".recording")){{
 const player=section.querySelector("audio");
 for(const row of section.querySelectorAll("tr[data-start]")){{
  row.addEventListener("click",()=>{{const t=Number(row.dataset.start);if(Number.isFinite(t)){{player.currentTime=t;player.play();}}}});
 }}
}}
</script></main></body></html>"""


def _render_dashboard(report: dict[str, Any]) -> str:
    template_path = BASE_DIR / "dashboard_template.html"
    template = template_path.read_text(encoding="utf-8")
    serialized = json.dumps(report, ensure_ascii=True).replace("</", "<\\/")
    if "__REPORT_DATA__" not in template:
        raise ValueError(f"Dashboard data placeholder not found in {template_path}")
    return template.replace("__REPORT_DATA__", serialized)


def run(
    model_path: Path = MODEL_PATH,
    report_path: Path = REPORT_PATH,
    dashboard_path: Path = DASHBOARD_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    targets = (model_path, report_path, dashboard_path)
    if len({path.resolve() for path in targets}) != len(targets):
        raise ValueError("Model, report, and dashboard output paths must differ.")
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
        )

    report, model = train_and_score()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "baseline_id": BASELINE_ID,
            "baseline_recording_ids": report["baseline_training_recording_ids"],
            "feature_names": FEATURE_NAMES,
            "review_percentile": BASELINE_REVIEW_PERCENTILE,
            "review_threshold": report["baseline_score_threshold"],
            "random_seed": RANDOM_SEED,
        },
        model_path,
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dashboard_path.write_text(_render_dashboard(report), encoding="utf-8")

    print(
        f"Fitted {report['model_type']} on "
        f"{report['baseline_training_words']} word intervals across "
        f"{len(report['baseline_training_recording_ids'])} reference recordings."
    )
    print(
        "Reference-corpus anomaly scores prioritize review; they are not "
        "accuracy or correctness estimates."
    )
    for recording in report["recordings"]:
        print(
            f"{recording['recording_id']}: "
            f"{recording['review_candidate_count']} model candidates / "
            f"{recording['heuristic_cue_count']} rows with review cues"
        )
    print(f"Model: {model_path}")
    print(f"Report: {report_path}")
    print(f"Dashboard: {dashboard_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a multi-recording timestamp-deviation model and dashboard."
        )
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--dashboard", type=Path, default=DASHBOARD_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the specified generated model, report, and dashboard files.",
    )
    args = parser.parse_args()
    try:
        run(args.model, args.report, args.dashboard, overwrite=args.overwrite)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
