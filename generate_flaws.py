
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import soundfile as sf


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("standardized_reference")
OUTPUT_DIR = Path("generated_flaws")
LABELS_FILE = Path("flaw_labels.csv")

# Enable or disable individual flaw types.
ENABLE_LONG_PAUSE = True
ENABLE_FAST_SPEECH = True
ENABLE_SLOW_SPEECH = True
ENABLE_LOW_ENERGY = True
ENABLE_HIGH_ENERGY = True
ENABLE_BACKGROUND_NOISE = True
ENABLE_CLIPPING = True

# Long-pause durations in seconds.
PAUSE_DURATIONS = [1.0, 2.0, 3.0]

# Time-stretch rates. These preserve pitch when librosa works.
FAST_RATE = 1.25
SLOW_RATE = 0.80

# Energy changes.
LOW_ENERGY_GAIN = 0.35
HIGH_ENERGY_GAIN = 1.5

# Noise strength relative to the recording's RMS.
NOISE_STRENGTH = 0.15

# Random generator for reproducible noise.
RNG = np.random.default_rng(2026)


# ============================================================
# OPTIONAL TIME-STRETCH SUPPORT
# ============================================================

try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception as exc:
    librosa = None
    LIBROSA_AVAILABLE = False
    warnings.warn(
        f"librosa unavailable; fast/slow speech will be skipped. "
        f"Reason: {exc}"
    )


# ============================================================
# AUDIO UTILITIES
# ============================================================

def load_audio(path):
    """Load audio as float32 with shape (samples, channels)."""
    audio, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True
    )
    return audio, sr


def save_audio(path, audio, sr):
    """Save generated audio as 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0
    )

    # Protect the output file from invalid sample amplitudes.
    audio = np.clip(audio, -1.0, 1.0)

    sf.write(
        str(path),
        audio,
        sr,
        subtype="PCM_16"
    )


def audio_duration(audio, sr):
    return len(audio) / sr


def make_label(
    output_path,
    source_path,
    flaw_type,
    start_sec,
    end_sec,
    sr,
    method,
    severity="moderate"
):
    """Create a label for the affected interval."""

    return {
        "audio_id": output_path.stem,
        "source_audio": source_path.name,
        "flaw_type": flaw_type,
        "start_sec": round(float(start_sec), 4),
        "end_sec": round(float(end_sec), 4),
        "duration_sec": round(
            float(end_sec - start_sec), 4
        ),
        "sample_rate": sr,
        "insertion_method": method,
        "synthetic": True,
        "severity": severity,
        "audio_path": output_path.as_posix(),
        "audio_duration_sec": None,
    }


def find_low_energy_position(audio, sr):
    """
    Find a low-energy 100 ms window in the middle portion
    of the recording. Returns the insertion sample index.

    This is a heuristic, not a verified word boundary.
    """
    mono = audio.mean(axis=1)
    total_samples = len(mono)

    window_size = max(1, int(0.10 * sr))

    search_start = int(total_samples * 0.35)
    search_end = int(total_samples * 0.65)

    if search_end - search_start < window_size:
        return total_samples // 2

    best_position = search_start
    best_rms = float("inf")

    step = max(1, int(0.02 * sr))

    for position in range(
        search_start,
        search_end - window_size + 1,
        step
    ):
        window = mono[position:position + window_size]
        rms = float(np.sqrt(np.mean(window ** 2) + 1e-12))

        if rms < best_rms:
            best_rms = rms
            best_position = position

    return best_position + window_size // 2


# ============================================================
# FL AW GENERATORS
# ============================================================

def generate_long_pauses(audio, sr, source_path, labels):
    """Insert known-duration silence into a recording."""

    if not ENABLE_LONG_PAUSE:
        return

    insertion_point = find_low_energy_position(audio, sr)

    for pause_duration in PAUSE_DURATIONS:
        pause_samples = int(round(pause_duration * sr))

        silence = np.zeros(
            (pause_samples, audio.shape[1]),
            dtype=np.float32
        )

        modified = np.concatenate(
            [
                audio[:insertion_point],
                silence,
                audio[insertion_point:]
            ],
            axis=0
        )

        output_path = OUTPUT_DIR / (
            f"{source_path.stem}__long_pause_"
            f"{pause_duration:g}s.wav"
        )

        save_audio(output_path, modified, sr)

        label = make_label(
            output_path,
            source_path,
            "synthetic_inserted_pause",
            insertion_point / sr,
            insertion_point / sr + pause_duration,
            sr,
            "insert_silence_at_low_energy_position",
            "moderate" if pause_duration < 3 else "high"
        )

        label["audio_duration_sec"] = round(
            audio_duration(modified, sr), 4
        )
        labels.append(label)


def generate_time_stretched(
    audio, sr, source_path, labels, rate, flaw_type
):
    """
    Change speaking speed using pitch-preserving time stretching.
    The complete recording is altered, so its label covers the
    complete output recording.
    """

    if not LIBROSA_AVAILABLE:
        return

    stretched = librosa.effects.time_stretch(
        y=audio.T,
        rate=rate
    ).T

    stretched = np.asarray(stretched, dtype=np.float32)

    output_path = OUTPUT_DIR / (
        f"{source_path.stem}__{flaw_type}.wav"
    )

    save_audio(output_path, stretched, sr)

    duration = audio_duration(stretched, sr)

    label = make_label(
        output_path,
        source_path,
        f"synthetic_{flaw_type}",
        0.0,
        duration,
        sr,
        f"librosa_pitch_preserving_time_stretch_rate_{rate}",
        "moderate"
    )

    label["audio_duration_sec"] = round(duration, 4)
    labels.append(label)


def generate_energy_variant(
    audio, sr, source_path, labels, gain, flaw_type
):
    """Change loudness while retaining the original duration."""

    peak = float(np.max(np.abs(audio)))

    if flaw_type == "high_energy" and peak > 0:
        # Increase amplitude without intentionally clipping
        # this high-energy example.
        gain = min(gain, 0.95 / peak)

    modified = audio * gain

    output_path = OUTPUT_DIR / (
        f"{source_path.stem}__{flaw_type}.wav"
    )

    save_audio(output_path, modified, sr)

    duration = audio_duration(modified, sr)

    label = make_label(
        output_path,
        source_path,
        f"synthetic_{flaw_type}",
        0.0,
        duration,
        sr,
        f"amplitude_gain_{gain:.4f}",
        "moderate"
    )

    label["audio_duration_sec"] = round(duration, 4)
    labels.append(label)


def generate_background_noise(audio, sr, source_path, labels):
    """Add controlled white noise to the entire recording."""

    mono_rms = float(
        np.sqrt(np.mean(audio ** 2) + 1e-12)
    )

    noise_rms = max(mono_rms * NOISE_STRENGTH, 0.002)

    noise = RNG.normal(
        loc=0.0,
        scale=noise_rms,
        size=audio.shape
    ).astype(np.float32)

    modified = audio + noise

    output_path = OUTPUT_DIR / (
        f"{source_path.stem}__background_noise.wav"
    )

    save_audio(output_path, modified, sr)

    duration = audio_duration(modified, sr)

    label = make_label(
        output_path,
        source_path,
        "synthetic_background_noise",
        0.0,
        duration,
        sr,
        f"add_white_noise_rms_{noise_rms:.6f}",
        "moderate"
    )

    label["audio_duration_sec"] = round(duration, 4)
    labels.append(label)


def generate_clipping(audio, sr, source_path, labels):
    """Create a hard-clipped version of the recording."""

    peak = float(np.max(np.abs(audio)))

    if peak < 1e-8:
        print(f"  Skipping clipping for silent file: {source_path.name}")
        return

    threshold = max(peak * 0.45, 1e-5)
    modified = np.clip(audio, -threshold, threshold)

    output_path = OUTPUT_DIR / (
        f"{source_path.stem}__clipping.wav"
    )

    save_audio(output_path, modified, sr)

    duration = audio_duration(modified, sr)

    label = make_label(
        output_path,
        source_path,
        "synthetic_clipping_distortion",
        0.0,
        duration,
        sr,
        f"hard_clip_threshold_{threshold:.6f}",
        "high"
    )

    label["audio_duration_sec"] = round(duration, 4)
    labels.append(label)


# ============================================================
# LABEL MERGING
# ============================================================

def save_merged_labels(new_labels):
    """
    Preserve existing labels and merge newly generated rows.
    Re-running the script replaces rows with matching audio_id.
    """

    new_df = pd.DataFrame(new_labels)

    if LABELS_FILE.exists():
        old_df = pd.read_csv(LABELS_FILE)
        combined = pd.concat(
            [old_df, new_df],
            ignore_index=True,
            sort=False
        )
    else:
        combined = new_df

    combined = combined.drop_duplicates(
        subset=["audio_id"],
        keep="last"
    )

    combined.to_csv(LABELS_FILE, index=False)

    print(f"\nCombined labels saved: {LABELS_FILE}")
    print(f"Total unique labelled examples: {len(combined)}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input folder not found: {INPUT_DIR.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_files = sorted(INPUT_DIR.glob("*.wav"))

    if not source_files:
        raise FileNotFoundError(
            f"No WAV files found in {INPUT_DIR.resolve()}"
        )

    labels = []

    print(f"Source recordings found: {len(source_files)}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")

    for source_path in source_files:
        print(f"\nProcessing: {source_path.name}")

        audio, sr = load_audio(source_path)

        if len(audio) == 0:
            print("  Skipping empty audio file.")
            continue

        if ENABLE_LONG_PAUSE:
            generate_long_pauses(
                audio, sr, source_path, labels
            )

        if ENABLE_FAST_SPEECH:
            generate_time_stretched(
                audio, sr, source_path, labels,
                FAST_RATE, "fast_speech"
            )

        if ENABLE_SLOW_SPEECH:
            generate_time_stretched(
                audio, sr, source_path, labels,
                SLOW_RATE, "slow_speech"
            )

        if ENABLE_LOW_ENERGY:
            generate_energy_variant(
                audio, sr, source_path, labels,
                LOW_ENERGY_GAIN, "low_energy"
            )

        if ENABLE_HIGH_ENERGY:
            generate_energy_variant(
                audio, sr, source_path, labels,
                HIGH_ENERGY_GAIN, "high_energy"
            )

        if ENABLE_BACKGROUND_NOISE:
            generate_background_noise(
                audio, sr, source_path, labels
            )

        if ENABLE_CLIPPING:
            generate_clipping(
                audio, sr, source_path, labels
            )

    if labels:
        save_merged_labels(labels)
    else:
        print(
            "\nNo new variants generated. "
            "Check which categories are enabled."
        )

    print("\nGeneration complete.")
    print(f"New audio variants generated: {len(labels)}")
    print(f"Audio folder: {OUTPUT_DIR.resolve()}")
    print(f"Labels file: {LABELS_FILE.resolve()}")

    if not LIBROSA_AVAILABLE:
        print(
            "\nWARNING: Fast and slow speech variants were skipped "
            "because librosa could not be imported."
        )


if __name__ == "__main__":
    main()
