import os
import json
import csv
import statistics

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
ALIGNMENT_FOLDER = os.path.join(BASE, "alignments")
OUTPUT_FILE = os.path.join(BASE, "alignment_quality_report.csv")

LOW_SCORE = 0.50

# --------------------------------------------------
# FIND ALIGNMENT FILES
# --------------------------------------------------

files = sorted(
    f for f in os.listdir(ALIGNMENT_FOLDER)
    if f.lower().endswith(".json")
)

print(f"Found {len(files)} alignment files\n")

results = []

# --------------------------------------------------
# ANALYZE EACH FILE
# --------------------------------------------------

for filename in files:

    path = os.path.join(ALIGNMENT_FOLDER, filename)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data.get("words", [])

    scores = [
        float(w["score"])
        for w in words
        if "score" in w
    ]

    total_words = len(words)

    if scores:
        average_score = statistics.mean(scores)
        minimum_score = min(scores)
        low_score_count = sum(
            1 for score in scores if score < LOW_SCORE
        )
    else:
        average_score = 0
        minimum_score = 0
        low_score_count = 0

    if total_words > 0:
        low_score_percent = (
            low_score_count / total_words
        ) * 100
    else:
        low_score_percent = 0

    # --------------------------------------------------
    # QUALITY CLASSIFICATION
    # --------------------------------------------------

    if low_score_percent <= 2:
        quality = "GOOD"

    elif low_score_percent <= 8:
        quality = "REVIEW"

    else:
        quality = "POOR"

    results.append({
        "file": data.get("file", filename),
        "duration": data.get("duration", ""),
        "total_words": total_words,
        "low_score_words": low_score_count,
        "low_score_percent": round(low_score_percent, 2),
        "average_score": round(average_score, 3),
        "minimum_score": round(minimum_score, 3),
        "quality": quality
    })

# --------------------------------------------------
# PRINT SUMMARY
# --------------------------------------------------

print("=" * 80)
print("ALIGNMENT QUALITY SUMMARY")
print("=" * 80)

print(
    f"{'FILE':<15}"
    f"{'WORDS':>8}"
    f"{'LOW':>8}"
    f"{'LOW %':>10}"
    f"{'AVG':>10}"
    f"{'MIN':>10}"
    f"{'QUALITY':>12}"
)

print("-" * 80)

for r in results:

    print(
        f"{r['file']:<15}"
        f"{r['total_words']:>8}"
        f"{r['low_score_words']:>8}"
        f"{r['low_score_percent']:>10.2f}"
        f"{r['average_score']:>10.3f}"
        f"{r['minimum_score']:>10.3f}"
        f"{r['quality']:>12}"
    )

print("-" * 80)

# --------------------------------------------------
# OVERALL COUNTS
# --------------------------------------------------

good = sum(1 for r in results if r["quality"] == "GOOD")
review = sum(1 for r in results if r["quality"] == "REVIEW")
poor = sum(1 for r in results if r["quality"] == "POOR")

print()
print(f"GOOD   : {good}")
print(f"REVIEW : {review}")
print(f"POOR   : {poor}")

# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------

fieldnames = [
    "file",
    "duration",
    "total_words",
    "low_score_words",
    "low_score_percent",
    "average_score",
    "minimum_score",
    "quality"
]

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)

print()
print(f"Report saved to:")
print(OUTPUT_FILE)