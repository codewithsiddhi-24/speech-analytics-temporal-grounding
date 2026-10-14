"""Serve the local review dashboard and analyze uploaded English WAV files."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import sys
import tempfile
import threading
import uuid
import wave
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "dashboard_template.html"
REPORT_PATH = BASE_DIR / "reports" / "timestamp_review_baseline.json"
REFERENCE_MODEL_PATH = BASE_DIR / "reports" / "timestamp_review_baseline_model.joblib"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MIN_GAP_SECONDS = 0.05
MIN_PAUSE_SECONDS = 0.25
ASR_CONFIDENCE_REVIEW_THRESHOLD = 0.45
FINDING_CONTEXT_PADDING_SECONDS = 0.25
MAX_FINDING_SPAN_SECONDS = 8.0
MODEL_NAME = os.environ.get("SPEECHLENS_WHISPER_MODEL", "small.en")

_model: Any | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_reference_model: dict[str, Any] | None = None
_reference_model_lock = threading.Lock()


def _load_model() -> Any:
    global _model
    with _model_lock:
        if _model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "Upload transcription requires faster-whisper. Install "
                    "the optional dependencies with: "
                    ".\\.venv\\Scripts\\python.exe -m pip install -r "
                    ".\\requirements-upload.txt"
                ) from exc
            _model = WhisperModel(
                MODEL_NAME, device="cpu", compute_type="int8"
            )
        return _model


def _load_reference_model() -> dict[str, Any]:
    global _reference_model
    with _reference_model_lock:
        if _reference_model is None:
            if not REFERENCE_MODEL_PATH.is_file():
                raise RuntimeError(
                    "Reference comparison model is missing. Generate it with "
                    ".\\.venv\\Scripts\\python.exe .\\train_timestamp_review_model.py "
                    "--overwrite"
                )
            try:
                import joblib
            except ImportError as exc:
                raise RuntimeError(
                    "Reference comparison requires joblib and scikit-learn. "
                    "Install the project model dependencies first."
                ) from exc
            artifact = joblib.load(REFERENCE_MODEL_PATH)
            if not isinstance(artifact, dict) or not {
                "model",
                "feature_names",
                "review_threshold",
                "baseline_recording_ids",
            }.issubset(artifact):
                raise ValueError(
                    f"Invalid reference comparison model: {REFERENCE_MODEL_PATH}"
                )
            _reference_model = artifact
        return _reference_model


def _read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getcomptype() != "NONE":
                raise ValueError("Upload an uncompressed PCM WAV file.")
            if audio.getsampwidth() != 2:
                raise ValueError("Upload a 16-bit PCM WAV file.")
            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            if channels < 1 or sample_rate < 1 or frame_count < 1:
                raise ValueError("The uploaded WAV has no readable audio samples.")
            samples = np.frombuffer(
                audio.readframes(frame_count), dtype="<i2"
            ).astype(np.float32)
    except wave.Error as exc:
        raise ValueError("The uploaded file is not a readable WAV recording.") from exc

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    samples /= 32768.0
    return samples, sample_rate


def _frame_levels(
    samples: np.ndarray, sample_rate: int, frame_seconds: float = 0.025
) -> tuple[np.ndarray, np.ndarray]:
    frame_size = max(1, round(sample_rate * frame_seconds))
    hop_size = max(1, round(sample_rate * 0.01))
    frame_count = max(1, math.ceil(max(0, samples.size - frame_size) / hop_size) + 1)
    starts = np.arange(frame_count, dtype=np.int64) * hop_size
    levels = np.empty(frame_count, dtype=np.float32)
    offsets = np.arange(frame_size, dtype=np.int64)

    for batch_start in range(0, frame_count, 8192):
        batch_starts = starts[batch_start : batch_start + 8192]
        indices = batch_starts[:, None] + offsets[None, :]
        frames = np.zeros(indices.shape, dtype=np.float32)
        valid = indices < samples.size
        frames[valid] = samples[indices[valid]]
        levels[batch_start : batch_start + len(batch_starts)] = np.sqrt(
            np.mean(frames * frames, axis=1)
        )

    times = starts.astype(np.float64) / sample_rate + frame_seconds / 2
    levels_db = 20 * np.log10(np.maximum(levels, 1e-7))
    return times, levels_db


def _intervals_from_mask(
    times: np.ndarray,
    mask: np.ndarray,
    minimum_duration: float,
    frame_seconds: float = 0.025,
) -> list[dict[str, float]]:
    intervals: list[dict[str, float]] = []
    active_start: int | None = None
    for index, active in enumerate(mask):
        if active and active_start is None:
            active_start = index
        elif not active and active_start is not None:
            start = max(0.0, float(times[active_start] - frame_seconds / 2))
            end = float(times[index - 1] + frame_seconds / 2)
            if end - start >= minimum_duration:
                intervals.append({"start": start, "end": end})
            active_start = None
    if active_start is not None:
        start = max(0.0, float(times[active_start] - frame_seconds / 2))
        end = float(times[-1] + frame_seconds / 2)
        if end - start >= minimum_duration:
            intervals.append({"start": start, "end": end})
    return intervals


def _quiet_threshold(levels_db: np.ndarray) -> float:
    peak_db = float(np.max(levels_db))
    noise_floor_db = float(np.percentile(levels_db, 10))
    return max(-60.0, min(peak_db - 20.0, max(noise_floor_db + 8.0, peak_db - 40.0)))


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
                if math.isfinite(float(values[index]))
                else None
            ),
        }
        for index in indices
    ]


def _audio_visual_data(
    samples: np.ndarray, sample_rate: int, duration: float
) -> dict[str, Any]:
    point_count = min(900, max(1, samples.size))
    edges = np.linspace(0, samples.size, point_count + 1, dtype=int)
    peak = max(float(np.max(np.abs(samples))), 1e-8)
    waveform = [
        [
            round(float(np.min(samples[edges[i] : edges[i + 1]]) / peak), 4),
            round(float(np.max(samples[edges[i] : edges[i + 1]]) / peak), 4),
        ]
        for i in range(point_count)
        if edges[i + 1] > edges[i]
    ]

    window_size = min(1024, max(64, 2 ** int(math.floor(math.log2(sample_rate / 8)))))
    window_size = min(window_size, max(64, samples.size))
    hop_size = max(window_size, math.ceil(samples.size / 900))
    starts = np.arange(0, max(1, samples.size - window_size + 1), hop_size)
    if starts.size == 0:
        starts = np.array([0])
    window = np.hanning(window_size).astype(np.float32)
    spectrogram = np.zeros((128, starts.size), dtype=np.uint8)
    for column, start in enumerate(starts):
        frame = samples[start : start + window_size]
        if frame.size < window_size:
            frame = np.pad(frame, (0, window_size - frame.size))
        magnitude = np.abs(np.fft.rfft(frame * window))
        usable = magnitude[:512]
        if usable.size < 512:
            usable = np.pad(usable, (0, 512 - usable.size))
        grouped = usable.reshape(128, 4).max(axis=1)
        decibels = 20 * np.log10(np.maximum(grouped / (window.sum() / 2), 1e-7))
        spectrogram[:, column] = np.clip((decibels + 70) * (255 / 70), 0, 255)

    return {
        "duration_seconds": duration,
        "waveform_minmax": waveform,
        "spectrogram": {
            "height": 128,
            "width": int(starts.size),
            "sample_rate": sample_rate,
            "nyquist_hz": sample_rate / 2,
            "png_data": base64.b64encode(spectrogram.tobytes()).decode("ascii"),
        },
    }


def _pitch_analysis(
    samples: np.ndarray, sample_rate: int, frame_times: np.ndarray
) -> tuple[
    list[dict[str, float]],
    list[dict[str, float]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    stride = max(1, math.ceil(frame_times.size / 30000))
    selected_times = frame_times[::stride]
    decimation = max(1, round(sample_rate / 16000))
    effective_rate = sample_rate / decimation
    frame_size = max(256, round(effective_rate * 0.04))
    fft_size = 1 << (2 * frame_size - 1).bit_length()
    min_lag = max(1, math.floor(effective_rate / 350))
    max_lag = min(frame_size - 1, math.ceil(effective_rate / 60))
    window = np.hanning(frame_size).astype(np.float32)
    pitch_values = np.full(selected_times.size, np.nan, dtype=np.float64)
    voicing_values = np.zeros(selected_times.size, dtype=np.float64)

    for index, time_sec in enumerate(selected_times):
        center = round(float(time_sec) * sample_rate)
        start = max(0, center - round(frame_size * decimation / 2))
        frame = samples[start : start + frame_size * decimation : decimation]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        if float(np.mean(frame * frame)) < 1e-6:
            continue
        centered = (frame - np.mean(frame)) * window
        spectrum = np.fft.rfft(centered, n=fft_size)
        autocorrelation = np.fft.irfft(
            spectrum * np.conjugate(spectrum), n=fft_size
        )[:frame_size]
        peak = float(autocorrelation[0])
        if peak <= 1e-10 or max_lag <= min_lag:
            continue
        candidates = autocorrelation[min_lag : max_lag + 1]
        lag = min_lag + int(np.argmax(candidates))
        periodicity = float(autocorrelation[lag] / peak)
        voicing_values[index] = min(1.0, max(0.0, periodicity))
        if periodicity >= 0.3:
            pitch_values[index] = effective_rate / lag

    valid = np.isfinite(pitch_values)
    pitch_series = [
        {"time": float(time_sec), "value": float(value)}
        for time_sec, value in zip(selected_times[valid], pitch_values[valid])
    ]
    voicing_series = [
        {"time": float(time_sec), "value": float(value)}
        for time_sec, value in zip(selected_times, voicing_values)
    ]
    return pitch_series, voicing_series, selected_times, pitch_values, voicing_values


def _window_values(
    times: np.ndarray, values: np.ndarray, start: float, end: float
) -> np.ndarray:
    mask = (times >= start) & (times < end)
    return values[mask]


def _finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _json_number(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _boundary_difference(
    times: np.ndarray, values: np.ndarray, boundary: float
) -> float:
    before = _finite_median(_window_values(times, values, boundary - 0.1, boundary))
    after = _finite_median(_window_values(times, values, boundary, boundary + 0.1))
    if not math.isfinite(before) or not math.isfinite(after):
        return float("nan")
    return after - before


def _upload_spectral_features(
    samples: np.ndarray, sample_rate: int, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    frame_size = max(64, round(sample_rate * 0.025))
    fft_size = 1 << (frame_size - 1).bit_length()
    window = np.hanning(frame_size)
    frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    centroid = np.empty(times.size, dtype=np.float64)
    zero_crossing = np.empty(times.size, dtype=np.float64)

    for index, time_sec in enumerate(times):
        center = round(float(time_sec) * sample_rate)
        start = center - frame_size // 2
        end = start + frame_size
        frame = np.zeros(frame_size, dtype=np.float64)
        source_start = max(0, start)
        source_end = min(samples.size, end)
        if source_end > source_start:
            frame[source_start - start : source_end - start] = samples[
                source_start:source_end
            ]
        magnitude = np.abs(np.fft.rfft(frame * window, n=fft_size))
        magnitude_total = float(np.sum(magnitude))
        centroid[index] = (
            float(np.dot(frequencies, magnitude) / magnitude_total)
            if magnitude_total > 1e-12
            else float("nan")
        )
        zero_crossing[index] = float(
            np.count_nonzero(np.diff(np.signbit(frame))) / max(frame_size - 1, 1)
        )
    return centroid, zero_crossing


def _score_upload_words(
    words: list[dict[str, Any]],
    times: np.ndarray,
    levels_db: np.ndarray,
    pitch_times: np.ndarray,
    voiced_probability: np.ndarray,
    spectral_centroid: np.ndarray,
    zero_crossing_rate: np.ndarray,
    reference_artifact: dict[str, Any],
) -> int:
    feature_names = reference_artifact["feature_names"]
    rows: list[list[float]] = []
    for index, word in enumerate(words):
        start, end = word["start"], word["end"]
        start_change = _boundary_difference(times, levels_db, start)
        end_change = _boundary_difference(times, levels_db, end)
        start_voiced_change = _boundary_difference(
            pitch_times, voiced_probability, start
        )
        end_voiced_change = _boundary_difference(
            pitch_times, voiced_probability, end
        )
        start_centroid_change = _boundary_difference(
            pitch_times, spectral_centroid, start
        )
        end_centroid_change = _boundary_difference(
            pitch_times, spectral_centroid, end
        )
        word_rms = _finite_median(
            _window_values(times, levels_db, start, end)
        )
        word_voiced = _finite_mean(
            _window_values(pitch_times, voiced_probability, start, end)
        )
        word_zcr = _finite_median(
            _window_values(pitch_times, zero_crossing_rate, start, end)
        )
        word["word_rms_db_median"] = _json_number(word_rms)
        word["word_voiced_fraction"] = _json_number(word_voiced)
        word["word_zero_crossing_rate_median"] = _json_number(word_zcr)
        word["start_boundary_rms_change_db"] = _json_number(start_change)
        word["end_boundary_rms_change_db"] = _json_number(end_change)
        word["start_boundary_voiced_change"] = _json_number(start_voiced_change)
        word["end_boundary_voiced_change"] = _json_number(end_voiced_change)
        word["start_boundary_centroid_change_hz"] = _json_number(
            start_centroid_change
        )
        word["end_boundary_centroid_change_hz"] = _json_number(
            end_centroid_change
        )
        previous = words[index - 1] if index else {}
        following = words[index + 1] if index + 1 < len(words) else {}
        values = {
            "alignment_score": float("nan"),
            "log_word_duration": math.log(max(word["duration_seconds"], 1e-4)),
            "gap_before_seconds": word["gap_before_seconds"],
            "gap_after_seconds": word["gap_after_seconds"],
            "previous_alignment_score": float("nan"),
            "next_alignment_score": float("nan"),
            "word_rms_db_median": word_rms,
            "word_voiced_fraction": word_voiced,
            "start_boundary_rms_change_db": start_change,
            "end_boundary_rms_change_db": end_change,
            "start_boundary_voiced_change": start_voiced_change,
            "end_boundary_voiced_change": end_voiced_change,
            "start_boundary_centroid_change_hz": start_centroid_change,
            "end_boundary_centroid_change_hz": end_centroid_change,
            "word_zero_crossing_rate_median": word_zcr,
        }
        if previous.get("alignment_score") is not None:
            values["previous_alignment_score"] = previous["alignment_score"]
        if following.get("alignment_score") is not None:
            values["next_alignment_score"] = following["alignment_score"]
        rows.append([values[name] for name in feature_names])

    if not rows:
        return 0
    matrix = np.asarray(rows, dtype=np.float64)
    model = reference_artifact["model"]
    transformed = model.named_steps["simpleimputer"].transform(matrix)
    scores = model.named_steps["isolationforest"].score_samples(transformed)
    threshold = float(reference_artifact["review_threshold"])
    baseline_ids = reference_artifact["baseline_recording_ids"]
    for word, score in zip(words, scores):
        word["baseline_anomaly_score"] = float(score)
        word["baseline_score_threshold"] = threshold
        word["model_review_candidate"] = bool(score <= threshold)
        word["review_cues"] = list(word.get("review_cues", []))
        if word["model_review_candidate"]:
            word["review_cues"].append("unusual_vs_reference_corpus")
    return len(baseline_ids)


def _upload_scores(words: list[dict[str, Any]]) -> dict[str, Any]:
    confidences = np.asarray(
        [
            float(word["asr_confidence"])
            for word in words
            if word.get("asr_confidence") is not None
            and math.isfinite(float(word["asr_confidence"]))
        ],
        dtype=np.float64,
    )
    pace_windows: list[float] = []
    for start_index in range(0, len(words), 4):
        group = words[start_index : start_index + 6]
        if len(group) < 5:
            break
        while (
            len(group) > 5
            and group[-1]["end"] - group[0]["start"] > MAX_FINDING_SPAN_SECONDS
        ):
            group = group[:-1]
        span = group[-1]["end"] - group[0]["start"]
        if 1.0 <= span <= MAX_FINDING_SPAN_SECONDS:
            pace_windows.append(len(group) * 60.0 / span)

    def percentage(value: float) -> int:
        return round(min(100.0, max(0.0, value)))

    return {
        "recognition_confidence": (
            percentage(float(np.mean(confidences)) * 100)
            if confidences.size
            else None
        ),
        "high_confidence_words": (
            percentage(
                float(
                    np.count_nonzero(
                        confidences >= ASR_CONFIDENCE_REVIEW_THRESHOLD
                    )
                )
                / confidences.size
                * 100
            )
            if confidences.size
            else None
        ),
        "phrase_pace_in_range": (
            percentage(
                sum(90 <= rate <= 190 for rate in pace_windows)
                / len(pace_windows)
                * 100
            )
            if pace_windows
            else None
        ),
        "phrase_pace_window_count": len(pace_windows),
    }


def _causal_findings(
    words: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    pauses: list[dict[str, Any]],
    pitch_times: np.ndarray,
    pitch_values: np.ndarray,
    level_times: np.ndarray,
    levels_db: np.ndarray,
    duration: float,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    def add(
        category: str,
        start: float,
        end: float,
        transcript: str,
        reason: str,
        evidence: str,
        listen_start: float | None = None,
        listen_end: float | None = None,
    ) -> None:
        start = max(0.0, float(start))
        end = min(duration, float(end))
        if end <= start:
            return
        findings.append(
            {
                "category": category,
                "start": start,
                "end": end,
                "listen_start": max(
                    0.0,
                    start - FINDING_CONTEXT_PADDING_SECONDS
                    if listen_start is None
                    else listen_start,
                ),
                "listen_end": min(
                    duration,
                    end + FINDING_CONTEXT_PADDING_SECONDS
                    if listen_end is None
                    else listen_end,
                ),
                "transcript": transcript,
                "reason": reason,
                "evidence": evidence,
                "confidence": None,
                "label_source": "heuristic_review_cue",
            }
        )

    for start_index in range(0, len(words), 4):
        group = words[start_index : start_index + 6]
        if len(group) < 5:
            break
        while (
            len(group) > 5
            and group[-1]["end"] - group[0]["start"] > MAX_FINDING_SPAN_SECONDS
        ):
            group = group[:-1]
        start, end = group[0]["start"], group[-1]["end"]
        if end - start < 1.0 or end - start > MAX_FINDING_SPAN_SECONDS:
            continue
        transcript = " ".join(word["word"] for word in group)
        rate = len(group) * 60.0 / (end - start)
        if rate < 90:
            add(
                "SLOWER SPEECH PACE",
                start,
                end,
                transcript,
                "Recognized-word rate is below the 90 words/min review threshold; "
                "pauses and emphasis can also lower this rate.",
                f"{rate:.0f} words/min over {end - start:.2f} s.",
            )
        elif rate > 190:
            add(
                "FASTER SPEECH PACE",
                start,
                end,
                transcript,
                "Recognized words are arriving more quickly than the review "
                "range; listen to check whether the words remain clear.",
                f"{rate:.0f} words/min over {end - start:.2f} s.",
            )

        local_pitch = _window_values(pitch_times, pitch_values, start, end)
        local_pitch = local_pitch[np.isfinite(local_pitch)]
        if local_pitch.size >= 12:
            typical_pitch = float(np.median(local_pitch))
            pitch_median = float(np.median(local_pitch))
            pitch_spread = float(np.std(local_pitch))
            overall_pitch = pitch_values[np.isfinite(pitch_values)]
            reference_pitch = (
                float(np.median(overall_pitch)) if overall_pitch.size else 0.0
            )
            if (
                reference_pitch > 0
                and pitch_median >= reference_pitch * 1.15
                and pitch_median - reference_pitch >= 20
            ):
                add(
                    "HIGHER PITCH THAN RECORDING TYPICAL",
                    start,
                    end,
                    transcript,
                    "Voiced pitch is higher here than the median voiced pitch in "
                    "this recording; pitch alone does not indicate a flaw.",
                    f"Median F0 {typical_pitch:.0f} Hz; recording median "
                    f"{reference_pitch:.0f} Hz.",
                )
            elif (
                reference_pitch > 0
                and pitch_median <= reference_pitch * 0.85
                and reference_pitch - pitch_median >= 20
            ):
                add(
                    "LOWER PITCH THAN RECORDING TYPICAL",
                    start,
                    end,
                    transcript,
                    "Voiced pitch is lower here than the median voiced pitch in "
                    "this recording; pitch alone does not indicate a flaw.",
                    f"Median F0 {typical_pitch:.0f} Hz; recording median "
                    f"{reference_pitch:.0f} Hz.",
                )
            if typical_pitch > 0 and pitch_spread / typical_pitch < 0.045:
                add(
                    "LIMITED PITCH VARIATION",
                    start,
                    end,
                    transcript,
                    "Pitch varies little through this phrase, which can make the "
                    "delivery sound more even or monotone.",
                    f"F0 standard deviation {pitch_spread:.1f} Hz "
                    f"({pitch_spread / typical_pitch:.1%} of the median).",
                )

        voiced_mask = np.isfinite(pitch_values)
        local_voice_mask = (
            (pitch_times >= start) & (pitch_times < end) & voiced_mask
        )
        aligned_levels = np.interp(pitch_times, level_times, levels_db)
        voiced_levels = aligned_levels[local_voice_mask]
        overall_levels = aligned_levels[voiced_mask]
        overall_voice = pitch_values[voiced_mask]
        if voiced_levels.size and overall_voice.size:
            typical_level = float(np.median(voiced_levels))
            reference_level = float(np.median(overall_levels))
            level_delta = typical_level - reference_level
            if level_delta <= -5.0:
                add(
                    "QUIETER VOICE LEVEL",
                    start,
                    end,
                    transcript,
                    "Measured speech energy is lower here than the recording "
                    "average; microphone distance and emphasis affect level.",
                    f"Median energy {typical_level:.1f} dBFS; recording median "
                    f"{reference_level:.1f} dBFS ({level_delta:+.1f} dB).",
                )
            elif level_delta >= 5.0:
                add(
                    "HIGHER VOICE LEVEL",
                    start,
                    end,
                    transcript,
                    "Measured speech energy is higher here than the recording "
                    "average; emphasis and microphone distance affect level.",
                    f"Median energy {typical_level:.1f} dBFS; recording median "
                    f"{reference_level:.1f} dBFS ({level_delta:+.1f} dB).",
                )

    fillers = {"um", "uh", "erm", "er", "ah"}
    normalized_words = [
        re.sub(r"[^\w']", "", word["word"]).casefold() for word in words
    ]
    for index, word in enumerate(words):
        normalized = normalized_words[index]
        if normalized in fillers:
            add(
                "HESITATION",
                word["start"],
                word["end"],
                word["word"],
                "A common hesitation filler was recognized in this interval.",
                f"Recognized filler token: {word['word']!r}.",
            )
        if (
            index > 0
            and normalized
            and normalized == normalized_words[index - 1]
            and normalized not in fillers
        ):
            previous = words[index - 1]
            add(
                "HESITATION",
                previous["start"],
                word["end"],
                f"{previous['word']} {word['word']}",
                "The same word was recognized twice in succession; this may be "
                "a repetition or a recognition artifact.",
                "Adjacent transcript tokens match after punctuation is removed.",
            )
        if word.get("asr_unclear_candidate"):
            add(
                "POSSIBLE MUMBLE / UNCLEAR SPEECH",
                word["start"],
                word["end"],
                word["word"],
                "Low speech-recognition confidence suggests this word may be "
                "unclear; noise, accent, overlap, or recognition error can cause "
                "the same cue.",
                f"ASR confidence {word['asr_confidence']:.2f} is below the "
                f"{ASR_CONFIDENCE_REVIEW_THRESHOLD:.2f} review threshold.",
            )
        if word.get("model_review_candidate"):
            add(
                "REFERENCE-DEVIATION CANDIDATE",
                word["start"],
                word["end"],
                word["word"],
                "Timing and acoustic features differ from the reference "
                "recordings; review this phrase in context.",
                f"Reference deviation score {word['baseline_anomaly_score']:.3f}; "
                f"review threshold {word['baseline_score_threshold']:.3f}.",
            )

    for gap in gaps:
        category = (
            "POSSIBLE NATURAL PAUSE / BREATH"
            if gap["duration"] >= 0.25 and gap["quiet_fraction"] >= 0.8
            else "INTER-WORD GAP"
        )
        add(
            category,
            gap["start"],
            gap["end"],
            f"{gap['before_word']}  …  {gap['after_word']}",
            gap["reason"],
            f"Gap duration {gap['duration']:.2f} s; "
            f"low-energy fraction {gap['quiet_fraction']:.0%}; "
            f"voiced fraction {gap['voiced_fraction']:.0%}.",
            max(0.0, gap["start"] - FINDING_CONTEXT_PADDING_SECONDS),
            min(duration, gap["end"] + FINDING_CONTEXT_PADDING_SECONDS),
        )

    for pause in pauses:
        if any(
            min(pause["end"], gap["end"]) - max(pause["start"], gap["start"])
            >= min(pause["duration"], gap["duration"]) * 0.5
            for gap in gaps
        ):
            continue
        before = next(
            (
                word["word"]
                for word in reversed(words)
                if word["end"] <= pause["start"]
            ),
            "recording start",
        )
        after = next(
            (
                word["word"]
                for word in words
                if word["start"] >= pause["end"]
            ),
            "recording end",
        )
        add(
            "LOW-ENERGY PAUSE",
            pause["start"],
            pause["end"],
            f"{before}  …  {after}",
            pause["reason"],
            f"Low-energy interval {pause['duration']:.2f} s; "
            f"adaptive threshold {pause.get('threshold_db', float('nan')):.1f} dB.",
        )

    findings.sort(key=lambda finding: (finding["category"], finding["start"]))
    merged: list[dict[str, Any]] = []
    for finding in findings:
        if (
            merged
            and merged[-1]["category"] == finding["category"]
            and finding["start"] <= merged[-1]["end"] + 0.35
            and max(merged[-1]["end"], finding["end"]) - merged[-1]["start"]
            <= MAX_FINDING_SPAN_SECONDS
        ):
            previous = merged[-1]
            previous["end"] = max(previous["end"], finding["end"])
            previous["listen_start"] = min(
                previous["listen_start"], finding["listen_start"]
            )
            previous["listen_end"] = max(
                previous["listen_end"], finding["listen_end"]
            )
            previous_tokens = previous["transcript"].split()
            next_tokens = finding["transcript"].split()
            overlap = min(len(previous_tokens), len(next_tokens))
            while overlap and previous_tokens[-overlap:] != next_tokens[:overlap]:
                overlap -= 1
            previous["transcript"] = " ".join(
                previous_tokens + next_tokens[overlap:]
            )
            if finding["evidence"] not in previous["evidence"]:
                previous["evidence"] += f" {finding['evidence']}"
        else:
            merged.append(finding)
    return sorted(merged, key=lambda finding: (finding["start"], finding["end"]))


def _gap_reason(
    times: np.ndarray,
    levels_db: np.ndarray,
    start: float,
    end: float,
    threshold: float,
    pitch_times: np.ndarray | None = None,
    pitch_values: np.ndarray | None = None,
) -> tuple[str, float]:
    in_gap = (times >= start) & (times <= end)
    quiet_fraction = (
        float(np.mean(levels_db[in_gap] <= threshold))
        if np.any(in_gap)
        else 1.0
    )
    if quiet_fraction >= 0.8:
        return (
            "Low-energy gap; likely a natural pause or silence. Breathing cannot "
            "be confirmed from this signal-only check.",
            quiet_fraction,
        )
    if pitch_times is not None and pitch_values is not None:
        in_pitch_gap = (pitch_times >= start) & (pitch_times <= end)
        pitch_samples = pitch_values[in_pitch_gap]
        if pitch_samples.size:
            voiced_fraction = float(np.mean(np.isfinite(pitch_samples)))
            if voiced_fraction >= 0.5:
                return (
                    "Voiced energy continues but no word was timestamped; possible "
                    "unrecognized or mumbled speech. Review the audio.",
                    quiet_fraction,
                )
            return (
                "Non-quiet, mostly unvoiced energy; possible breath or another "
                "non-speech sound. Fricatives can also be unvoiced speech.",
                quiet_fraction,
            )
    return (
        "Audio energy continues but no word was timestamped; possible breath, "
        "mumbled/unclear speech, or another non-speech sound. Review this segment.",
        quiet_fraction,
    )


def _transcribe(path: Path, model: Any) -> tuple[list[dict[str, Any]], str]:
    segments, _ = model.transcribe(
        str(path),
        language="en",
        beam_size=5,
        temperature=0.0,
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    words: list[dict[str, Any]] = []
    transcript_parts: list[str] = []
    for segment in segments:
        transcript_parts.append(str(segment.text).strip())
        for item in segment.words or []:
            word = str(item.word).strip()
            start = float(item.start)
            end = float(item.end)
            if not word or not math.isfinite(start) or not math.isfinite(end):
                continue
            if end <= start:
                continue
            probability = float(item.probability)
            probability = min(1.0, max(0.0, probability))
            words.append(
                {
                    "word_index": len(words),
                    "word": word,
                    "start": start,
                    "end": end,
                    "duration_seconds": end - start,
                    "alignment_score": None,
                    "asr_confidence": probability,
                    "asr_unclear_candidate": (
                        probability < ASR_CONFIDENCE_REVIEW_THRESHOLD
                    ),
                    "gap_before_seconds": (
                        max(0.0, start - words[-1]["end"]) if words else 0.0
                    ),
                    "gap_after_seconds": None,
                    "baseline_anomaly_score": None,
                    "baseline_score_threshold": None,
                    "model_review_candidate": False,
                    "review_cues": (
                        ["low_asr_confidence_review_audio"]
                        if probability < ASR_CONFIDENCE_REVIEW_THRESHOLD
                        else []
                    ),
                }
            )
    for index, word in enumerate(words[:-1]):
        word["gap_after_seconds"] = max(0.0, words[index + 1]["start"] - word["end"])
    if words:
        words[-1]["gap_after_seconds"] = 0.0
    return words, " ".join(part for part in transcript_parts if part)


def analyze_upload(
    path: Path,
    filename: str,
    model: Any,
    reference_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    samples, sample_rate = _read_pcm_wav(path)
    duration = samples.size / sample_rate
    times, levels_db = _frame_levels(samples, sample_rate)
    threshold = _quiet_threshold(levels_db)
    with _inference_lock:
        words, transcript = _transcribe(path, model)
    quiet_mask = levels_db <= threshold
    recognized_speech_mask = np.zeros(times.size, dtype=bool)
    for word in words:
        recognized_speech_mask |= (
            (times >= max(0.0, word["start"] - 0.05))
            & (times <= min(duration, word["end"] + 0.05))
        )
    pauses = _intervals_from_mask(
        times,
        quiet_mask & ~recognized_speech_mask,
        MIN_PAUSE_SECONDS,
    )
    for pause in pauses:
        pause["start"] = min(pause["start"], duration)
        pause["end"] = min(pause["end"], duration)
        pause["duration"] = pause["end"] - pause["start"]
        pause["threshold_db"] = threshold
        pause["reason"] = (
            "Low-energy interval outside recognized word times; may be a natural "
            "pause, breathing, or silence."
        )
    (
        pitch_series,
        voicing_series,
        pitch_times,
        pitch_values,
        pitch_voicing,
    ) = _pitch_analysis(samples, sample_rate, times)
    spectral_centroid, zero_crossing_rate = _upload_spectral_features(
        samples, sample_rate, pitch_times
    )
    baseline_recording_count = 0
    if reference_artifact is not None:
        baseline_recording_count = _score_upload_words(
            words,
            times,
            levels_db,
            pitch_times,
            pitch_voicing,
            spectral_centroid,
            zero_crossing_rate,
            reference_artifact,
        )
    gaps: list[dict[str, Any]] = []
    for previous, following in zip(words, words[1:]):
        start, end = previous["end"], following["start"]
        if end - start < MIN_GAP_SECONDS:
            continue
        reason, quiet_fraction = _gap_reason(
            times,
            levels_db,
            start,
            end,
            threshold,
            pitch_times,
            pitch_values,
        )
        in_pitch_gap = (pitch_times >= start) & (pitch_times <= end)
        pitch_samples = pitch_values[in_pitch_gap]
        voiced_fraction = (
            float(np.mean(np.isfinite(pitch_samples))) if pitch_samples.size else 0.0
        )
        gaps.append(
            {
                "start": start,
                "end": end,
                "duration": end - start,
                "kind": "gap",
                "before_word": previous["word"],
                "after_word": following["word"],
                "reason": reason,
                "quiet_fraction": quiet_fraction,
                "voiced_fraction": voiced_fraction,
            }
        )
    pauses = [
        pause
        for pause in pauses
        if not any(
            min(pause["end"], gap["end"]) - max(pause["start"], gap["start"])
            >= min(pause["duration"], gap["duration"]) * 0.5
            for gap in gaps
        )
    ]

    local_rates = []
    for start in np.arange(0, duration, 10.0):
        end = min(duration, float(start + 10.0))
        count = sum(start <= word["start"] < end for word in words)
        local_rates.append(
            {"start": float(start), "end": end, "wpm": count * 60.0 / (end - start)}
        )

    causal_findings = _causal_findings(
        words,
        gaps,
        pauses,
        pitch_times,
        pitch_values,
        times,
        levels_db,
        duration,
    )
    valid_pitch = pitch_values[np.isfinite(pitch_values)]
    duration_values = np.asarray([word["duration_seconds"] for word in words])
    gaps_before = np.asarray([word["gap_before_seconds"] for word in words[1:]])
    recording_id = f"upload_{uuid.uuid4().hex[:12]}"
    summary_pause_duration = sum(item["duration"] for item in pauses)
    return {
        "recording_id": recording_id,
        "display_name": Path(filename).name,
        "is_baseline": False,
        "is_upload": True,
        "reference_comparison": (
            {
                "available": reference_artifact is not None,
                "recording_count": baseline_recording_count,
                "review_candidate_count": sum(
                    bool(word["model_review_candidate"]) for word in words
                ),
                "meaning": (
                    "Candidate intervals differ from reference recordings; "
                    "this is a review priority, not a confirmed speech flaw."
                ),
            }
            if reference_artifact is not None
            else {"available": False, "recording_count": 0}
        ),
        "audio_path": "",
        "transcript_path": "",
        "transcript": transcript,
        "words": words,
        "analysis": {
            "duration_seconds": duration,
            "word_count": len(words),
            "scores": _upload_scores(words),
            "mean_word_duration_seconds": (
                float(np.mean(duration_values)) if duration_values.size else None
            ),
            "median_gap_seconds": (
                float(np.median(gaps_before)) if gaps_before.size else None
            ),
            "alignment_score_mean": None,
            "alignment_score_missing": len(words),
            "global_wpm": len(words) * 60.0 / duration,
            "local_10sec": local_rates,
            "pitch_mean_hz": (
                float(np.mean(valid_pitch)) if valid_pitch.size else None
            ),
            "pitch_std_hz": (
                float(np.std(valid_pitch)) if valid_pitch.size else None
            ),
            "pauses": pauses,
            "gaps": gaps,
            "causal_findings": causal_findings,
            "pause_summary": {
                "count": len(pauses),
                "total_duration": summary_pause_duration,
            },
            "pause_threshold_db": threshold,
            "pause_method": "relative frame-energy heuristic",
            "feature_series": {
                "rms_db": _downsample_series(times, levels_db),
                "f0_hz": pitch_series,
                "voiced_probability": voicing_series,
                "spectral_centroid_hz": _downsample_series(
                    pitch_times, spectral_centroid
                ),
            },
            "acoustic_summary": {
                "pitch_f0_hz": {
                    "mean": float(np.mean(valid_pitch)) if valid_pitch.size else None,
                    "std": float(np.std(valid_pitch)) if valid_pitch.size else None,
                },
                "energy_rms_db": {
                    "mean": float(np.mean(levels_db)),
                    "std": float(np.std(levels_db)),
                },
            },
        },
        "audio_visual": _audio_visual_data(samples, sample_rate, duration),
        "review_candidate_count": sum(
            bool(word["model_review_candidate"]) for word in words
        ),
        "heuristic_cue_count": len(causal_findings),
    }


def _safe_report() -> dict[str, Any]:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if not isinstance(report.get("recordings"), list):
        raise ValueError(f"Invalid dashboard report: {REPORT_PATH}")
    return report


class DashboardHandler(SimpleHTTPRequestHandler):
    server_version = "SpeechLensLocal/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def translate_path(self, path: str) -> str:
        parts = Path(unquote(urlsplit(path).path).lstrip("/")).parts
        if (
            len(parts) != 2
            or parts[0] != "standardized_reference"
            or not parts[1].lower().endswith(".wav")
        ):
            return str(BASE_DIR / "__not_served__")
        audio_dir = (BASE_DIR / "standardized_reference").resolve()
        requested = (audio_dir / parts[1]).resolve()
        if requested.parent != audio_dir or not requested.is_file():
            return str(BASE_DIR / "__not_served__")
        return str(requested)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            try:
                template = TEMPLATE_PATH.read_text(encoding="utf-8")
                if "__REPORT_DATA__" not in template:
                    raise ValueError("Dashboard report placeholder is missing.")
                body = template.replace(
                    "__REPORT_DATA__",
                    json.dumps(_safe_report(), ensure_ascii=True).replace("</", "<\\/"),
                ).encode("utf-8")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.send_error(500, f"Unable to load the local dashboard: {exc}")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/analyze":
            self.send_error(404, "Unknown local analysis endpoint.")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length."})
            return
        if content_length < 1 or content_length > MAX_UPLOAD_BYTES:
            self._send_json(
                413,
                {
                    "error": (
                        "Upload must be between 1 byte and "
                        f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                    )
                },
            )
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0] not in {
            "audio/wav",
            "audio/x-wav",
            "audio/x-pn-wav",
            "application/octet-stream",
        }:
            self._send_json(415, {"error": "Upload a WAV audio file."})
            return

        filename = Path(unquote(self.headers.get("X-Filename", "upload.wav"))).name
        if Path(filename).suffix.lower() != ".wav":
            self._send_json(415, {"error": "Upload a file with a .wav extension."})
            return
        body = self.rfile.read(content_length)
        if len(body) != content_length:
            self._send_json(
                400, {"error": "Upload ended before all audio data was received."}
            )
            return

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="speechlens_", suffix=".wav", delete=False
            ) as upload:
                upload.write(body)
                temp_path = Path(upload.name)
            analyzed = analyze_upload(
                temp_path,
                filename,
                _load_model(),
                _load_reference_model(),
            )
            self._send_json(200, analyzed)
        except RuntimeError as exc:
            self.log_error("Local audio analysis is unavailable: %s", exc)
            self._send_json(500, {"error": str(exc)})
        except (OSError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})
        except (AttributeError, TypeError) as exc:
            self.log_error("Local audio analysis failed: %s", exc)
            self._send_json(
                500,
                {
                    "error": (
                        "Local transcription could not run because an audio "
                        "dependency is incompatible. Reinstall the optional "
                        "dependencies with: .\\.venv\\Scripts\\python.exe -m "
                        "pip install -r .\\requirements-upload.txt"
                    )
                },
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def log_message(self, format: str, *args: Any) -> None:
        if self.path.startswith("/api/"):
            safe_path = "/api/analyze"
            print(f"{safe_path} - {format % args}", file=sys.stderr)
            return
        super().log_message(format, *args)


def main() -> int:
    host = "127.0.0.1"
    port = int(os.environ.get("SPEECHLENS_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"SpeechLens local dashboard: http://{host}:{port}/", flush=True)
    print(
        f"Uploads are analyzed locally; max size is "
        f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Press Ctrl+C to stop.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SpeechLens.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
