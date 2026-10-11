import os
import json
import numpy as np
import librosa


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_DIR = os.path.join(BASE_DIR, "generated_flaws\\long_pause")
ALIGNMENT_DIR = os.path.join(BASE_DIR, "alignments")
FEATURE_DIR = os.path.join(BASE_DIR, "generated_flaw_features")

SAMPLE_RATE = 16000

# 25 ms frame
FRAME_LENGTH = 400

# 10 ms hop
HOP_LENGTH = 160

# FFT size
N_FFT = 512

# MFCC count
N_MFCC = 13

# Pause configuration
MIN_PAUSE_SEC = 0.30

# Adaptive threshold:
# noise floor = P10 of RMS dB
# threshold = noise floor + this margin
PAUSE_MARGIN_DB = 8.0

# Alignment word-gap threshold
MIN_WORD_GAP_SEC = 0.30


# ============================================================
# ALIGNMENT QUALITY
# ============================================================

ALIGNMENT_QUALITY = {
    "good_01": "REVIEW",
    "good_02": "REVIEW",
    "good_03": "REVIEW",
    "good_04": "GOOD",
    "good_05": "GOOD",
    "good_06": "REVIEW",
    "good_07": "REVIEW",
    "good_08": "REVIEW",
    "good_09": "GOOD",
    "good_13": "REVIEW",
    "good_14": "REVIEW",
    "good_15": "REVIEW",
    "good_16": "POOR",
    "good_17": "REVIEW",
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(FEATURE_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def robust_stats(values):
    """
    Calculate robust descriptive statistics.

    NaN and infinite values are ignored.
    """

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
            "q25": None,
            "q75": None,
            "mad": None,
        }

    median = np.median(values)

    mad = np.median(
        np.abs(values - median)
    )

    q25, q75 = np.percentile(
        values,
        [25, 75]
    )

    return {
        "mean": round(float(np.mean(values)), 4),
        "median": round(float(median), 4),
        "min": round(float(np.min(values)), 4),
        "max": round(float(np.max(values)), 4),
        "std": round(float(np.std(values)), 4),
        "q25": round(float(q25), 4),
        "q75": round(float(q75), 4),
        "mad": round(float(mad), 4),
    }


def robust_normalize(values):
    """
    Robust normalization using:

        (x - median) / (1.4826 * MAD)

    This is stored for later analysis.

    It is NOT yet used for participant/reference comparison.
    """

    values = np.asarray(values, dtype=float)

    result = np.full_like(
        values,
        np.nan,
        dtype=float
    )

    valid = np.isfinite(values)

    if not np.any(valid):
        return result

    valid_values = values[valid]

    median = np.median(valid_values)

    mad = np.median(
        np.abs(valid_values - median)
    )

    scale = 1.4826 * mad

    if scale < 1e-8:
        result[valid] = 0.0
    else:
        result[valid] = (
            valid_values - median
        ) / scale

    return result


def f0_to_semitones(f0):
    """
    Convert F0 Hz to logarithmic/semitone representation.

    Invalid/unvoiced frames remain NaN.
    """

    f0 = np.asarray(f0, dtype=float)

    result = np.full_like(
        f0,
        np.nan,
        dtype=float
    )

    valid = (
        np.isfinite(f0)
        & (f0 > 0)
    )

    result[valid] = (
        12.0 * np.log2(f0[valid])
    )

    return result


def load_alignment(stem):
    """
    Load forced alignment JSON if available.
    """

    path = os.path.join(
        ALIGNMENT_DIR,
        stem + ".json"
    )

    if not os.path.exists(path):
        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def calculate_word_gaps(alignment):
    """
    Calculate gaps between consecutive aligned words.
    """

    if alignment is None:
        return []

    words = alignment.get(
        "words",
        []
    )

    gaps = []

    for i in range(len(words) - 1):

        current = words[i]
        next_word = words[i + 1]

        gap = (
            next_word["start"]
            - current["end"]
        )

        if gap >= MIN_WORD_GAP_SEC:

            gaps.append({
                "after_word": current["word"],
                "before_word": next_word["word"],
                "start": round(
                    float(current["end"]),
                    3
                ),
                "end": round(
                    float(next_word["start"]),
                    3
                ),
                "duration": round(
                    float(gap),
                    3
                ),
                "word_index": current["index"],
            })

    return gaps


def calculate_speech_rate(alignment):
    """
    Calculate global WPM using aligned word span.
    """

    if alignment is None:
        return {
            "word_count": 0,
            "speaking_span_sec": None,
            "words_per_minute": None,
        }

    words = alignment.get(
        "words",
        []
    )

    if not words:
        return {
            "word_count": 0,
            "speaking_span_sec": None,
            "words_per_minute": None,
        }

    start = words[0]["start"]
    end = words[-1]["end"]

    span = end - start

    if span <= 0:
        wpm = None
    else:
        wpm = (
            len(words)
            / span
            * 60.0
        )

    return {
        "word_count": len(words),
        "speaking_span_sec": round(
            float(span),
            3
        ),
        "words_per_minute": (
            round(float(wpm), 2)
            if wpm is not None
            else None
        ),
    }


def calculate_local_speech_rate(
    alignment,
    window_sec=10.0
):
    """
    Calculate WPM in 10-second windows.
    """

    if alignment is None:
        return []

    words = alignment.get(
        "words",
        []
    )

    if not words:
        return []

    last_time = words[-1]["end"]

    results = []

    window_start = 0.0

    while window_start < last_time:

        window_end = (
            window_start
            + window_sec
        )

        window_words = [
            word
            for word in words
            if (
                word["start"] >= window_start
                and word["start"] < window_end
            )
        ]

        if window_words:

            wpm = (
                len(window_words)
                / window_sec
                * 60.0
            )

            results.append({
                "start": round(
                    window_start,
                    3
                ),
                "end": round(
                    window_end,
                    3
                ),
                "word_count": len(
                    window_words
                ),
                "wpm": round(
                    float(wpm),
                    2
                ),
            })

        window_start += window_sec

    return results


def detect_adaptive_pauses(
    rms_db,
    times
):
    """
    Detect pauses using a recording-specific
    noise floor.

    noise_floor = P10(rms_db)

    threshold = noise_floor + PAUSE_MARGIN_DB

    This avoids using one fixed -40 dB threshold
    for every recording.
    """

    finite_db = rms_db[
        np.isfinite(rms_db)
    ]

    if len(finite_db) == 0:
        return {
            "noise_floor_db": None,
            "pause_threshold_db": None,
            "pauses": [],
        }

    noise_floor = np.percentile(
        finite_db,
        10
    )

    threshold = (
        noise_floor
        + PAUSE_MARGIN_DB
    )

    silent = (
        rms_db <= threshold
    )

    pauses = []

    start_idx = None

    for i, is_silent in enumerate(
        silent
    ):

        if (
            is_silent
            and start_idx is None
        ):
            start_idx = i

        elif (
            not is_silent
            and start_idx is not None
        ):

            start = times[start_idx]

            end = (
                times[i - 1]
                + HOP_LENGTH / SAMPLE_RATE
            )

            duration = end - start

            if duration >= MIN_PAUSE_SEC:

                pauses.append({
                    "start": round(
                        float(start),
                        3
                    ),
                    "end": round(
                        float(end),
                        3
                    ),
                    "duration": round(
                        float(duration),
                        3
                    ),
                })

            start_idx = None

    # Handle pause extending to end
    if start_idx is not None:

        start = times[start_idx]

        end = (
            times[-1]
            + HOP_LENGTH / SAMPLE_RATE
        )

        duration = end - start

        if duration >= MIN_PAUSE_SEC:

            pauses.append({
                "start": round(
                    float(start),
                    3
                ),
                "end": round(
                    float(end),
                    3
                ),
                "duration": round(
                    float(duration),
                    3
                ),
            })

    return {
        "noise_floor_db": round(
            float(noise_floor),
            3
        ),
        "pause_threshold_db": round(
            float(threshold),
            3
        ),
        "pauses": pauses,
    }


def pause_statistics(pauses):
    """
    Statistics for detected pauses.
    """

    if not pauses:
        return {
            "count": 0,
            "mean_duration": None,
            "median_duration": None,
            "min_duration": None,
            "max_duration": None,
            "total_duration": 0.0,
        }

    durations = np.array(
        [
            p["duration"]
            for p in pauses
        ],
        dtype=float
    )

    return {
        "count": len(pauses),
        "mean_duration": round(
            float(np.mean(durations)),
            3
        ),
        "median_duration": round(
            float(np.median(durations)),
            3
        ),
        "min_duration": round(
            float(np.min(durations)),
            3
        ),
        "max_duration": round(
            float(np.max(durations)),
            3
        ),
        "total_duration": round(
            float(np.sum(durations)),
            3
        ),
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("TRACK C - FEATURE EXTRACTION v2")
print("=" * 70)

if not os.path.exists(AUDIO_DIR):
    raise FileNotFoundError(
        f"Audio folder not found:\n{AUDIO_DIR}"
    )

audio_files = sorted(
    [
        f
        for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(".wav")
    ]
)

print(
    f"Audio files found: {len(audio_files)}"
)

print(
    f"Audio folder: {AUDIO_DIR}"
)

print(
    f"Output folder: {FEATURE_DIR}"
)

print()


# ============================================================
# PROCESS EACH AUDIO FILE
# ============================================================

for file_index, audio_file in enumerate(
    audio_files,
    start=1
):

    stem = os.path.splitext(
        audio_file
    )[0]

    print(
        f"[{file_index}/{len(audio_files)}] "
        f"Processing {audio_file}"
    )

    audio_path = os.path.join(
        AUDIO_DIR,
        audio_file
    )

    summary_path = os.path.join(
        FEATURE_DIR,
        stem + "_summary.json"
    )

    frames_path = os.path.join(
        FEATURE_DIR,
        stem + "_frames.npz"
    )

    # --------------------------------------------------------
    # LOAD AUDIO
    # --------------------------------------------------------

    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    duration = (
        len(y) / sr
    )

    # --------------------------------------------------------
    # F0 / PITCH
    # --------------------------------------------------------

    print("  extracting F0...")

    f0, voiced_flag, voiced_prob = (
        librosa.pyin(
            y,
            fmin=70,
            fmax=400,
            sr=sr,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )
    )

    # --------------------------------------------------------
    # RMS ENERGY
    # --------------------------------------------------------

    print("  extracting RMS...")

    rms = librosa.feature.rms(
        y=y,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        center=True,
    )[0]

    rms_db = librosa.amplitude_to_db(
        rms,
        ref=np.max
    )

    # --------------------------------------------------------
    # FFT / SPECTRAL FEATURES
    # --------------------------------------------------------

    print("  extracting spectral features...")

    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]
    )

    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]
    )

    spectral_rolloff = (
        librosa.feature.spectral_rolloff(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            roll_percent=0.85,
        )[0]
    )

    spectral_flatness = (
        librosa.feature.spectral_flatness(
            y=y,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]
    )

    zero_crossing_rate = (
        librosa.feature.zero_crossing_rate(
            y,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )[0]
    )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    print("  extracting MFCC...")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    mfcc_delta = librosa.feature.delta(
        mfcc
    )

    mfcc_delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # --------------------------------------------------------
    # TIMESTAMPS
    # --------------------------------------------------------

    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=HOP_LENGTH
    )

    # --------------------------------------------------------
    # ALIGN FEATURE LENGTHS
    # --------------------------------------------------------

    lengths = [
        len(times),
        len(f0),
        len(rms),
        len(spectral_centroid),
        len(spectral_bandwidth),
        len(spectral_rolloff),
        len(spectral_flatness),
        len(zero_crossing_rate),
        mfcc.shape[1],
    ]

    n_frames = min(lengths)

    times = times[:n_frames]

    f0 = f0[:n_frames]

    voiced_flag = (
        voiced_flag[:n_frames]
    )

    voiced_prob = (
        voiced_prob[:n_frames]
    )

    rms = rms[:n_frames]

    rms_db = rms_db[:n_frames]

    spectral_centroid = (
        spectral_centroid[:n_frames]
    )

    spectral_bandwidth = (
        spectral_bandwidth[:n_frames]
    )

    spectral_rolloff = (
        spectral_rolloff[:n_frames]
    )

    spectral_flatness = (
        spectral_flatness[:n_frames]
    )

    zero_crossing_rate = (
        zero_crossing_rate[:n_frames]
    )

    mfcc = mfcc[:, :n_frames]

    mfcc_delta = (
        mfcc_delta[:, :n_frames]
    )

    mfcc_delta2 = (
        mfcc_delta2[:, :n_frames]
    )

    # --------------------------------------------------------
    # NORMALIZED REPRESENTATIONS
    # --------------------------------------------------------

    f0_semitone = f0_to_semitones(
        f0
    )

    f0_normalized = robust_normalize(
        f0_semitone
    )

    rms_normalized = robust_normalize(
        rms_db
    )

    centroid_normalized = robust_normalize(
        spectral_centroid
    )

    # --------------------------------------------------------
    # ADAPTIVE PAUSES
    # --------------------------------------------------------

    print("  detecting adaptive pauses...")

    pause_result = detect_adaptive_pauses(
        rms_db,
        times
    )

    acoustic_pauses = (
        pause_result["pauses"]
    )

    acoustic_pause_stats = (
        pause_statistics(
            acoustic_pauses
        )
    )

    # --------------------------------------------------------
    # FORCED ALIGNMENT
    # --------------------------------------------------------

    alignment = load_alignment(
        stem
    )

    alignment_quality = (
        ALIGNMENT_QUALITY.get(
            stem,
            "UNKNOWN"
        )
    )

    if alignment is not None:

        aligned_words = alignment.get(
            "words",
            []
        )

        word_gaps = (
            calculate_word_gaps(
                alignment
            )
        )

        speech_rate = (
            calculate_speech_rate(
                alignment
            )
        )

        local_speech_rate = (
            calculate_local_speech_rate(
                alignment
            )
        )

    else:

        aligned_words = []

        word_gaps = []

        speech_rate = {
            "word_count": 0,
            "speaking_span_sec": None,
            "words_per_minute": None,
        }

        local_speech_rate = []

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {

        "file": audio_file,

        "feature_version": "2.0",

        "sample_rate": sr,

        "duration_sec": round(
            float(duration),
            3
        ),

        "alignment": {

            "quality": alignment_quality,

            "file": (
                stem + ".json"
                if alignment is not None
                else None
            ),

            "word_count": len(
                aligned_words
            ),
        },

        "frame_configuration": {

            "frame_length_ms": 25,

            "hop_length_ms": 10,

            "n_fft": N_FFT,

            "n_mfcc": N_MFCC,
        },

        "pitch": {

            "f0_hz": robust_stats(
                f0
            ),

            "f0_semitone": robust_stats(
                f0_semitone
            ),

            "f0_normalized": robust_stats(
                f0_normalized
            ),

            "voiced_frame_ratio": round(
                float(
                    np.mean(
                        voiced_flag
                    )
                ),
                4
            ),

            "mean_voiced_probability": round(
                float(
                    np.nanmean(
                        voiced_prob
                    )
                ),
                4
            ),
        },

        "energy": {

            "rms": robust_stats(
                rms
            ),

            "rms_db": robust_stats(
                rms_db
            ),

            "rms_normalized": robust_stats(
                rms_normalized
            ),
        },

        "spectral": {

            "centroid_hz": robust_stats(
                spectral_centroid
            ),

            "bandwidth_hz": robust_stats(
                spectral_bandwidth
            ),

            "rolloff_hz": robust_stats(
                spectral_rolloff
            ),

            "flatness": robust_stats(
                spectral_flatness
            ),

            "zero_crossing_rate": robust_stats(
                zero_crossing_rate
            ),

            "centroid_normalized": robust_stats(
                centroid_normalized
            ),
        },

        "mfcc": {

            "coefficient_count": N_MFCC,

            "mean": [
                round(float(v), 4)
                for v in np.nanmean(
                    mfcc,
                    axis=1
                )
            ],

            "std": [
                round(float(v), 4)
                for v in np.nanstd(
                    mfcc,
                    axis=1
                )
            ],

            "mad": [
                round(
                    float(
                        np.median(
                            np.abs(
                                mfcc[i]
                                - np.median(
                                    mfcc[i]
                                )
                            )
                        )
                    ),
                    4
                )
                for i in range(
                    N_MFCC
                )
            ],

            "delta_mean": [
                round(float(v), 4)
                for v in np.nanmean(
                    mfcc_delta,
                    axis=1
                )
            ],

            "delta2_mean": [
                round(float(v), 4)
                for v in np.nanmean(
                    mfcc_delta2,
                    axis=1
                )
            ],
        },

        "speech_rate": {

            "global": speech_rate,

            "local_10sec": (
                local_speech_rate
            ),
        },

        "pauses": {

            "method": "adaptive_noise_floor",

            "noise_floor_db": (
                pause_result[
                    "noise_floor_db"
                ]
            ),

            "pause_threshold_db": (
                pause_result[
                    "pause_threshold_db"
                ]
            ),

            "acoustic_statistics": (
                acoustic_pause_stats
            ),

            "acoustic_intervals": (
                acoustic_pauses
            ),

            "alignment_word_gaps": (
                word_gaps
            ),

            "alignment_gap_statistics": (
                pause_statistics(
                    word_gaps
                )
            ),
        },

        "notes": {

            "pitch": (
                "F0 estimated using librosa.pyin."
            ),

            "spectral": (
                "FFT-derived spectral features."
            ),

            "mfcc": (
                "13 MFCC coefficients with "
                "first and second derivatives."
            ),

            "normalization": (
                "Robust median/MAD normalization "
                "is stored for later comparison."
            ),

            "pause_detection": (
                "Pause threshold is recording-specific: "
                "P10 RMS dB + margin."
            ),

            "comparison": (
                "No participant/reference frame subtraction "
                "is performed during extraction."
            ),

            "alignment": (
                "Alignment quality affects confidence in "
                "word-level timing, not acoustic features."
            ),
        },
    }

    # --------------------------------------------------------
    # SAVE SUMMARY JSON
    # --------------------------------------------------------

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # SAVE FRAME-LEVEL DATA
    # --------------------------------------------------------

    np.savez_compressed(

        frames_path,

        time=times.astype(
            np.float32
        ),

        f0_hz=f0.astype(
            np.float32
        ),

        f0_semitone=f0_semitone.astype(
            np.float32
        ),

        f0_normalized=f0_normalized.astype(
            np.float32
        ),

        rms=rms.astype(
            np.float32
        ),

        rms_db=rms_db.astype(
            np.float32
        ),

        rms_normalized=rms_normalized.astype(
            np.float32
        ),

        spectral_centroid_hz=(
            spectral_centroid.astype(
                np.float32
            )
        ),

        spectral_bandwidth_hz=(
            spectral_bandwidth.astype(
                np.float32
            )
        ),

        spectral_rolloff_hz=(
            spectral_rolloff.astype(
                np.float32
            )
        ),

        spectral_flatness=(
            spectral_flatness.astype(
                np.float32
            )
        ),

        zero_crossing_rate=(
            zero_crossing_rate.astype(
                np.float32
            )
        ),

        spectral_centroid_normalized=(
            centroid_normalized.astype(
                np.float32
            )
        ),

        voiced=voiced_flag.astype(
            np.bool_
        ),

        voiced_probability=(
            voiced_prob.astype(
                np.float32
            )
        ),

        mfcc=mfcc.astype(
            np.float32
        ),

        mfcc_delta=mfcc_delta.astype(
            np.float32
        ),

        mfcc_delta2=mfcc_delta2.astype(
            np.float32
        ),
    )

    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------

    print(
        f"  duration       : "
        f"{duration:.2f} sec"
    )

    print(
        f"  frames         : "
        f"{n_frames}"
    )

    print(
        f"  aligned words  : "
        f"{len(aligned_words)}"
    )

    print(
        f"  alignment      : "
        f"{alignment_quality}"
    )

    print(
        f"  noise floor    : "
        f"{pause_result['noise_floor_db']} dB"
    )

    print(
        f"  pause threshold: "
        f"{pause_result['pause_threshold_db']} dB"
    )

    print(
        f"  acoustic pauses: "
        f"{len(acoustic_pauses)}"
    )

    print(
        f"  word gaps      : "
        f"{len(word_gaps)}"
    )

    print(
        f"  speech rate    : "
        f"{speech_rate['words_per_minute']} WPM"
    )

    print()


# ============================================================
# FINISHED
# ============================================================

print("=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"Generated files are in:\n{FEATURE_DIR}"
)

