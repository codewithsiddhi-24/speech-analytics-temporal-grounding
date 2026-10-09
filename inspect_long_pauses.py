import json
from pathlib import Path

import numpy as np
import soundfile as sf


# --------------------------------------------------
# Paths
# --------------------------------------------------

ALIGN_DIR = Path("alignments")
FEATURE_DIR = Path("features")
AUDIO_DIR = Path("standardized_reference")

LONG_PAUSE = 3.0


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_gap(frame_features, start, end):
    frames = [
        x for x in frame_features
        if start <= x["time"] < end
    ]

    if not frames:
        return {
            "frame_count": 0,
            "mean_rms": None,
            "median_rms": None,
            "voiced_fraction": None,
        }

    rms = np.array(
        [x["rms"] for x in frames],
        dtype=float
    )

    voiced = np.array(
        [x["voiced"] for x in frames],
        dtype=bool
    )

    return {
        "frame_count": len(frames),
        "mean_rms": float(np.mean(rms)),
        "median_rms": float(np.median(rms)),
        "voiced_fraction": float(np.mean(voiced)),
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

print("\nLONG PAUSE AUDIT\n")

for alignment_path in sorted(ALIGN_DIR.glob("*.json")):

    alignment = load_json(alignment_path)

    audio_file = alignment["file"]
    words = alignment.get("words", [])

    feature_path = FEATURE_DIR / audio_file.replace(
        ".wav",
        "_features.json"
    )

    if not feature_path.exists():
        print(f"{audio_file}: feature file missing")
        continue

    features = load_json(feature_path)

    frame_features = features.get("frame_features", [])

    print(f"\n{'=' * 80}")
    print(audio_file)

    found = 0

    for previous, current in zip(words, words[1:]):

        if previous.get("end") is None:
            continue

        if current.get("start") is None:
            continue

        gap_start = previous["end"]
        gap_end = current["start"]

        gap_duration = gap_end - gap_start

        if gap_duration < LONG_PAUSE:
            continue

        found += 1

        gap_stats = analyze_gap(
            frame_features,
            gap_start,
            gap_end
        )

        print(
            f"\nLONG GAP: {gap_duration:.2f}s"
        )

        print(
            f"  {previous['word']} "
            f"({previous['end']:.3f}s)"
            f"  ->  "
            f"{current['word']} "
            f"({current['start']:.3f}s)"
        )

        print(
            f"  Previous score: "
            f"{previous.get('score', 'N/A')}"
        )

        print(
            f"  Current score: "
            f"{current.get('score', 'N/A')}"
        )

        print(
            f"  Gap RMS mean: "
            f"{gap_stats['mean_rms']}"
        )

        print(
            f"  Gap RMS median: "
            f"{gap_stats['median_rms']}"
        )

        print(
            f"  Voiced fraction: "
            f"{gap_stats['voiced_fraction']}"
        )

        print(
            f"  Frames: "
            f"{gap_stats['frame_count']}"
        )

    if found == 0:
        print("  No gaps >= 3 seconds found.")