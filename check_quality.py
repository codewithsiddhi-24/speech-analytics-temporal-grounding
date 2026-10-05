import os
import glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_FOLDER = "audio_inp"
OUTPUT_CSV = "audio_quality_report.csv"

# Audio quality thresholds
MIN_DURATION_SEC = 2
MAX_DURATION_SEC = 600

LOW_VOLUME_DB = -35
SEVERE_LOW_VOLUME_DB = -45

CLIPPING_THRESHOLD = 0.99
CLIPPING_PERCENT_LIMIT = 0.01

SILENCE_DB = 35
MAX_BEGINNING_SILENCE_SEC = 3
MAX_ENDING_SILENCE_SEC = 3

MIN_SPEECH_RATIO = 0.20

MIN_SNR_DB = 10
MODERATE_SNR_DB = 15


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def amplitude_to_db(value):
    """
    Convert amplitude to decibels.
    """
    if value <= 1e-10:
        return -100.0

    return 20 * np.log10(value)


def analyze_audio(filepath):
    """
    Analyze one audio file and return quality information.
    """

    result = {
        "filename": os.path.basename(filepath),
        "filepath": filepath,
        "status": "UNKNOWN",
        "error": ""
    }

    try:

        # ----------------------------------------------------
        # LOAD AUDIO
        # ----------------------------------------------------

        audio, sample_rate = sf.read(filepath)

        # Convert stereo/multi-channel audio to mono
        if audio.ndim == 1:

            channels = 1
            channel_type = "Mono"

            mono = audio.astype(np.float32)

        else:

            channels = audio.shape[1]

            if channels == 2:
                channel_type = "Stereo"
            else:
                channel_type = f"{channels} channels"

            mono = np.mean(audio, axis=1).astype(np.float32)

        # Remove NaN / infinity
        mono = np.nan_to_num(mono)

        # Empty file check
        if len(mono) == 0:

            result["status"] = "REJECT"
            result["error"] = "Empty audio file"

            return result

        # ----------------------------------------------------
        # BASIC AUDIO INFORMATION
        # ----------------------------------------------------

        duration = len(mono) / sample_rate

        rms = np.sqrt(np.mean(mono ** 2))

        rms_db = amplitude_to_db(rms)

        peak = np.max(np.abs(mono))

        peak_db = amplitude_to_db(peak)

        # ----------------------------------------------------
        # CLIPPING DETECTION
        # ----------------------------------------------------

        clipping_samples = np.sum(
            np.abs(mono) >= CLIPPING_THRESHOLD
        )

        clipping_percentage = (
            clipping_samples / len(mono)
        ) * 100

        if clipping_percentage > CLIPPING_PERCENT_LIMIT:

            clipping_status = "CLIPPING"

        elif clipping_percentage > 0:

            clipping_status = "MINOR_CLIPPING"

        else:

            clipping_status = "OK"

        # ----------------------------------------------------
        # SPEECH / SILENCE DETECTION
        # ----------------------------------------------------

        intervals = librosa.effects.split(
            mono,
            top_db=SILENCE_DB
        )

        if len(intervals) > 0:

            speech_start_sample = intervals[0][0]
            speech_end_sample = intervals[-1][1]

            beginning_silence = (
                speech_start_sample / sample_rate
            )

            ending_silence = (
                len(mono) - speech_end_sample
            ) / sample_rate

            speech_samples = sum(
                end - start
                for start, end in intervals
            )

            speech_ratio = (
                speech_samples / len(mono)
            )

        else:

            beginning_silence = duration
            ending_silence = duration
            speech_ratio = 0.0

        silence_ratio = 1.0 - speech_ratio

        # ----------------------------------------------------
        # ESTIMATE BACKGROUND NOISE
        # ----------------------------------------------------

        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)

        if len(mono) > frame_length:

            rms_frames = librosa.feature.rms(
                y=mono,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]

            frame_db = (
                20
                * np.log10(
                    np.maximum(rms_frames, 1e-10)
                )
            )

            # Lowest 10% of energy frames
            # are treated as approximate noise
            noise_threshold = np.percentile(
                frame_db,
                10
            )

            noise_frames = rms_frames[
                frame_db <= noise_threshold
            ]

            if len(noise_frames) > 0:

                noise_rms = np.mean(noise_frames)

            else:

                noise_rms = np.mean(rms_frames)

            noise_db = amplitude_to_db(noise_rms)

            snr_db = rms_db - noise_db

        else:

            noise_db = -100.0
            snr_db = 0.0

        # ----------------------------------------------------
        # SAMPLE RATE QUALITY
        # ----------------------------------------------------

        if sample_rate < 16000:

            sample_rate_status = "LOW"

        elif sample_rate == 16000:

            sample_rate_status = "GOOD"

        else:

            sample_rate_status = "HIGH"

        # ----------------------------------------------------
        # FIND PROBLEMS
        # ----------------------------------------------------

        problems = []

        # Duration
        if duration < MIN_DURATION_SEC:

            problems.append("Very short")

        elif duration > MAX_DURATION_SEC:

            problems.append("Very long")

        # Volume
        if rms_db < LOW_VOLUME_DB:

            problems.append("Low volume")

        # Clipping
        if clipping_percentage > CLIPPING_PERCENT_LIMIT:

            problems.append("Clipping")

        # Beginning silence
        if beginning_silence > MAX_BEGINNING_SILENCE_SEC:

            problems.append("Long beginning silence")

        # Ending silence
        if ending_silence > MAX_ENDING_SILENCE_SEC:

            problems.append("Long ending silence")

        # Speech activity
        if speech_ratio < MIN_SPEECH_RATIO:

            problems.append("Low speech activity")

        # Background noise
        if snr_db < MIN_SNR_DB:

            problems.append("High background noise")

        elif snr_db < MODERATE_SNR_DB:

            problems.append("Moderate background noise")

        # Sample rate
        if sample_rate < 16000:

            problems.append("Low sample rate")

        # ----------------------------------------------------
        # SEVERE PROBLEM COUNT
        # ----------------------------------------------------

        severe_problems = 0

        if duration < 2:
            severe_problems += 1

        if clipping_percentage > 1:
            severe_problems += 1

        if speech_ratio < 0.10:
            severe_problems += 1

        if rms_db < SEVERE_LOW_VOLUME_DB:
            severe_problems += 1

        # ----------------------------------------------------
        # FINAL STATUS
        # ----------------------------------------------------

        if severe_problems > 0:

            status = "REJECT"

        elif len(problems) > 0:

            status = "CHECK"

        else:

            status = "PASS"

        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        result.update({

            "sample_rate": sample_rate,

            "channels": channels,

            "channel_type": channel_type,

            "sample_rate_status": sample_rate_status,

            "duration_sec": round(
                duration,
                2
            ),

            "rms_db": round(
                rms_db,
                2
            ),

            "peak_db": round(
                peak_db,
                2
            ),

            "clipping_percentage": round(
                clipping_percentage,
                4
            ),

            "clipping_status": clipping_status,

            "beginning_silence_sec": round(
                beginning_silence,
                2
            ),

            "ending_silence_sec": round(
                ending_silence,
                2
            ),

            "speech_ratio": round(
                speech_ratio,
                3
            ),

            "silence_ratio": round(
                silence_ratio,
                3
            ),

            "estimated_noise_db": round(
                noise_db,
                2
            ),

            "estimated_snr_db": round(
                snr_db,
                2
            ),

            "problems": "; ".join(problems),

            "status": status

        })

    except Exception as e:

        result["status"] = "ERROR"
        result["error"] = str(e)

    return result


# ============================================================
# FIND AUDIO FILES
# ============================================================

print("=" * 70)
print("TRACK C - AUDIO QUALITY CHECK")
print("=" * 70)

print()

if not os.path.exists(AUDIO_FOLDER):

    print(f"ERROR: Folder '{AUDIO_FOLDER}' does not exist.")

    print()
    print("Create the folder:")
    print(f"    mkdir {AUDIO_FOLDER}")

    print()
    print("Then put your audio files inside it.")

    raise SystemExit


extensions = [
    "*.wav",
    "*.mp3",
    "*.flac",
    "*.m4a",
    "*.ogg",
    "*.aac"
]


audio_files = []

for extension in extensions:

    audio_files.extend(
        glob.glob(
            os.path.join(
                AUDIO_FOLDER,
                extension
            )
        )
    )


# Remove duplicates
audio_files = sorted(
    list(set(audio_files))
)


print(
    f"Found {len(audio_files)} audio files."
)

print()


if len(audio_files) == 0:

    print("No audio files found.")

    print()
    print("Put files such as these inside:")
    print("    audio_inputs\\")
    print()
    print("Example:")
    print("    audio_inputs\\speech1.wav")
    print("    audio_inputs\\speech2.wav")

    raise SystemExit


# ============================================================
# ANALYZE ALL FILES
# ============================================================

results = []


for index, filepath in enumerate(audio_files, start=1):

    print(
        f"[{index}/{len(audio_files)}] "
        f"Analyzing: {os.path.basename(filepath)}"
    )

    result = analyze_audio(filepath)

    results.append(result)


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(results)


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print()

print(
    f"Report saved to: {OUTPUT_CSV}"
)

print()

print("STATUS SUMMARY")
print("-" * 30)

print(
    df["status"].value_counts()
)


print()
print("PROBLEM SUMMARY")
print("-" * 30)


problem_counts = {}


for problems in df["problems"].fillna(""):

    if problems.strip() == "":
        continue

    for problem in problems.split("; "):

        problem_counts[problem] = (
            problem_counts.get(problem, 0) + 1
        )


if len(problem_counts) == 0:

    print("No problems detected.")

else:

    for problem, count in sorted(
        problem_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{problem}: {count}"
        )


# ============================================================
# SHOW REJECTED / CHECK FILES
# ============================================================

print()
print("FILES REQUIRING ATTENTION")
print("-" * 30)


attention_files = df[
    df["status"].isin(
        ["REJECT", "CHECK"]
    )
]


if len(attention_files) == 0:

    print("All files passed.")

else:

    for _, row in attention_files.iterrows():

        print()
        print(
            f"{row['filename']} "
            f"--> {row['status']}"
        )

        if row["problems"]:

            print(
                f"Problems: {row['problems']}"
            )


print()
print("=" * 70)
print("DONE")
print("=" * 70)