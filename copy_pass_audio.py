
import os
import shutil
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing your original audio files
AUDIO_FOLDER = "audio_inp"

# Quality report created by audio_quality_check.py
QUALITY_REPORT = "audio_quality_report.csv"

# Destination for usable PASS recordings
OUTPUT_FOLDER = "clean_reference"


# ============================================================
# CHECK INPUTS
# ============================================================

print("=" * 70)
print("COPYING PASS AUDIO FILES")
print("=" * 70)
print()

if not os.path.exists(QUALITY_REPORT):
    print(f"ERROR: Cannot find {QUALITY_REPORT}")
    print()
    print("Make sure audio_quality_report.csv is in:")
    print(os.getcwd())
    raise SystemExit


if not os.path.exists(AUDIO_FOLDER):
    print(f"ERROR: Cannot find audio folder: {AUDIO_FOLDER}")
    raise SystemExit


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# READ QUALITY REPORT
# ============================================================

df = pd.read_csv(QUALITY_REPORT)

print(f"Quality report loaded: {QUALITY_REPORT}")
print(f"Total records in report: {len(df)}")
print()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["filename", "status"]

for column in required_columns:

    if column not in df.columns:

        print(
            f"ERROR: Required column '{column}' "
            f"is missing from the CSV."
        )

        raise SystemExit


# ============================================================
# SELECT ONLY PASS FILES
# ============================================================

pass_files = df[
    df["status"]
    .astype(str)
    .str.upper()
    .str.strip()
    == "PASS"
]


print(f"PASS files found: {len(pass_files)}")
print()


# ============================================================
# COPY PASS FILES
# ============================================================

copied = []
missing = []
failed = []


for _, row in pass_files.iterrows():

    filename = str(row["filename"]).strip()

    source = os.path.join(
        AUDIO_FOLDER,
        filename
    )

    destination = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    print(f"Processing: {filename}")

    # Check whether original audio exists
    if not os.path.exists(source):

        print("  -> NOT FOUND")
        missing.append(filename)
        continue

    try:

        shutil.copy2(
            source,
            destination
        )

        print("  -> COPIED")

        copied.append(filename)

    except Exception as e:

        print(f"  -> ERROR: {e}")

        failed.append(filename)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("COPY COMPLETE")
print("=" * 70)
print()

print(f"PASS files in report : {len(pass_files)}")
print(f"Successfully copied  : {len(copied)}")
print(f"Missing source files : {len(missing)}")
print(f"Copy failures        : {len(failed)}")

print()


# ============================================================
# SHOW COPIED FILES
# ============================================================

if copied:

    print("COPIED FILES")
    print("-" * 40)

    for filename in copied:

        print(f"  {filename}")

    print()


# ============================================================
# SHOW MISSING FILES
# ============================================================

if missing:

    print("MISSING FILES")
    print("-" * 40)

    for filename in missing:

        print(f"  {filename}")

    print()


# ============================================================
# SHOW FAILED FILES
# ============================================================

if failed:

    print("FAILED FILES")
    print("-" * 40)

    for filename in failed:

        print(f"  {filename}")

    print()


# ============================================================
# FINAL LOCATION
# ============================================================

print("Clean reference folder:")
print(
    os.path.abspath(OUTPUT_FOLDER)
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)