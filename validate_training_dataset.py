
import pandas as pd

path = "training_frames.csv"

# Read only the metadata columns to keep memory usage low.
columns = ["audio_id", "source_audio", "flaw_type", "target"]
df = pd.read_csv(path, usecols=columns)

print("=== DATASET VALIDATION ===")
print("Frame rows:", len(df))
print("Unique audio IDs:", df["audio_id"].nunique())
print("Unique source recordings:", df["source_audio"].nunique())

print("\n=== TARGETS ===")
print(df["target"].value_counts().to_string())

print("\n=== SOURCES PER TARGET ===")
for target, group in df.groupby("target"):
    print(
        f"{target}: "
        f"{group['source_audio'].nunique()} source recordings, "
        f"{group['audio_id'].nunique()} audio files"
    )

print("\n=== ORIGINAL AUDIO COVERAGE ===")
print("Original feature directory exists:", __import__("os").path.isdir("features"))
if __import__("os").path.isdir("features"):
    import os
    original_files = [
        f for f in os.listdir("features")
        if f.endswith("_frames.npz")
    ]
    print("Original frame feature files:", len(original_files))

print("\n=== SOURCE SPLIT CHECK ===")
print("Example source IDs:")
print(df["source_audio"].drop_duplicates().head(10).to_string(index=False))
