import os
import json
import glob
import statistics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIR = os.path.join(BASE_DIR, "features")

summary_files = sorted(
    glob.glob(os.path.join(FEATURE_DIR, "*_summary.json"))
)

if not summary_files:
    raise FileNotFoundError(
        f"No summary files found in: {FEATURE_DIR}"
    )


def safe_get(d, *keys, default=None):
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


rows = []

for path in summary_files:
    with open(path, "r", encoding="utf-8") as f:
        x = json.load(f)

    filename = os.path.basename(path)
    stem = filename.replace("_summary.json", "")

    rows.append({
        "file": stem,
        "quality": safe_get(x, "alignment", "quality"),
        "duration": safe_get(x, "duration_sec"),

        "f0_median": safe_get(
            x, "pitch", "f0_hz", "median"
        ),
        "f0_mad": safe_get(
            x, "pitch", "f0_hz", "mad"
        ),
        "voiced_pct": (
            safe_get(x, "pitch", "voiced_frame_ratio") * 100
            if safe_get(x, "pitch", "voiced_frame_ratio") is not None
            else None
        ),

        "rms_db_median": safe_get(
            x, "energy", "rms_db", "median"
        ),
        "rms_db_mad": safe_get(
            x, "energy", "rms_db", "mad"
        ),

        "wpm": safe_get(
            x, "speech_rate", "global", "words_per_minute"
        ),

        "pause_count": safe_get(
            x, "pauses", "acoustic_statistics", "count"
        ),
        "pause_total_sec": safe_get(
            x, "pauses", "acoustic_statistics", "total_duration"
        ),
        "pause_median_sec": safe_get(
            x, "pauses", "acoustic_statistics", "median_duration"
        ),
        "pause_max_sec": safe_get(
            x, "pauses", "acoustic_statistics", "max_duration"
        ),

        "noise_floor_db": safe_get(
            x, "pauses", "noise_floor_db"
        ),
        "pause_threshold_db": safe_get(
            x, "pauses", "pause_threshold_db"
        ),
    })


# Print header
headers = [
    "file",
    "quality",
    "duration",
    "f0_median",
    "voiced_pct",
    "rms_db_median",
    "wpm",
    "pause_count",
    "pause_total_sec",
    "pause_median_sec",
    "pause_max_sec",
    "noise_floor_db",
    "pause_threshold_db",
]

print()
print("=" * 150)
print("FEATURE VALIDATION REPORT")
print("=" * 150)

print(
    f"{'FILE':<10}"
    f"{'QUAL':<9}"
    f"{'DUR':>7}"
    f"{'F0':>9}"
    f"{'VOICED%':>10}"
    f"{'RMSdB':>10}"
    f"{'WPM':>8}"
    f"{'PAUSES':>9}"
    f"{'PAUSE_S':>11}"
    f"{'MED_S':>9}"
    f"{'MAX_S':>9}"
    f"{'NOISE':>9}"
    f"{'THRESH':>9}"
)

print("-" * 150)

for r in rows:
    print(
        f"{r['file']:<10}"
        f"{str(r['quality']):<9}"
        f"{r['duration']:>7.2f}"
        f"{r['f0_median']:>9.2f}"
        f"{r['voiced_pct']:>10.2f}"
        f"{r['rms_db_median']:>10.2f}"
        f"{r['wpm']:>8.2f}"
        f"{r['pause_count']:>9}"
        f"{r['pause_total_sec']:>11.2f}"
        f"{r['pause_median_sec']:>9.2f}"
        f"{r['pause_max_sec']:>9.2f}"
        f"{r['noise_floor_db']:>9.2f}"
        f"{r['pause_threshold_db']:>9.2f}"
    )

print("=" * 150)

print()
print(f"Files analyzed: {len(rows)}")

qualities = {}
for r in rows:
    qualities[r["quality"]] = qualities.get(r["quality"], 0) + 1

print("Alignment quality:")
for quality, count in qualities.items():
    print(f"  {quality}: {count}")

print()

# Simple duration check
bad_duration = [
    r for r in rows
    if r["duration"] is None or r["duration"] <= 0
]

if bad_duration:
    print("WARNING: Invalid duration:")
    for r in bad_duration:
        print(" ", r["file"])
else:
    print("Duration check: PASS")

# Feature sanity checks
bad_f0 = [
    r for r in rows
    if r["f0_median"] is None
    or r["f0_median"] < 50
    or r["f0_median"] > 500
]

if bad_f0:
    print("WARNING: Suspicious F0:")
    for r in bad_f0:
        print(
            f"  {r['file']}: "
            f"{r['f0_median']}"
        )
else:
    print("F0 median sanity check: PASS")

bad_wpm = [
    r for r in rows
    if r["wpm"] is None
    or r["wpm"] <= 0
    or r["wpm"] > 300
]

if bad_wpm:
    print("WARNING: Suspicious WPM:")
    for r in bad_wpm:
        print(
            f"  {r['file']}: "
            f"{r['wpm']}"
        )
else:
    print("Speech-rate sanity check: PASS")

print()
print("Validation complete.")