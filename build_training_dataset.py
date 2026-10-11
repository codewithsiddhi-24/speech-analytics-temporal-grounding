
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIR = os.path.join(BASE_DIR, "generated_flaw_features")
LABEL_FILE = os.path.join(BASE_DIR, "flaw_labels.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "training_frames.csv")

labels = pd.read_csv(LABEL_FILE)
labels["audio_id"] = labels["audio_id"].astype(str).str.strip()

# Map each feature filename to its label row.
label_map = {
    row["audio_id"]: row
    for _, row in labels.iterrows()
}

records = []
matched = set()
missing_labels = []
feature_files = sorted(
    f for f in os.listdir(FEATURE_DIR)
    if f.endswith("_frames.npz")
)

for filename in feature_files:
    audio_id = filename.removesuffix("_frames.npz")

    if audio_id not in label_map:
        missing_labels.append(audio_id)
        continue

    label = label_map[audio_id]
    feature_path = os.path.join(FEATURE_DIR, filename)

    with np.load(feature_path, allow_pickle=False) as data:
        if "time" not in data.files:
            print(f"Skipping {audio_id}: no time array")
            continue

        times = data["time"].astype(float)
        n_frames = len(times)

        # Scalar features: one value per frame.
        frame_features = {}

        for key in data.files:
            arr = data[key]

            if arr.ndim == 1 and len(arr) == n_frames:
                if key != "time":
                    frame_features[key] = arr.astype(float)

            elif arr.ndim == 2 and arr.shape[-1] == n_frames:
                # MFCC and delta features: one column per frame.
                for i in range(arr.shape[0]):
                    frame_features[f"{key}_{i+1}"] = (
                        arr[i].astype(float)
                    )

            elif arr.ndim == 2 and arr.shape[0] == n_frames:
                for i in range(arr.shape[1]):
                    frame_features[f"{key}_{i+1}"] = (
                        arr[:, i].astype(float)
                    )

        if not frame_features:
            print(f"Skipping {audio_id}: no frame features")
            continue

        df = pd.DataFrame(frame_features)
        df.insert(0, "time_sec", times)

        flaw_type = str(label["flaw_type"])
        start = pd.to_numeric(
            pd.Series([label.get("start_sec")]),
            errors="coerce"
        ).iloc[0]
        end = pd.to_numeric(
            pd.Series([label.get("end_sec")]),
            errors="coerce"
        ).iloc[0]

        if "synthetic_inserted_pause" in flaw_type:
            # Only the inserted interval is a pause target.
            df["target"] = "normal"
            if pd.notna(start) and pd.notna(end):
                pause_mask = (
                    (df["time_sec"] >= start) &
                    (df["time_sec"] <= end)
                )
                df.loc[pause_mask, "target"] = "long_pause"
            else:
                print(f"Warning: missing pause interval for {audio_id}")
                continue
        else:
            # These transformations affect the whole recording.
            df["target"] = flaw_type

        df["audio_id"] = audio_id
        df["source_audio"] = str(label.get("source_audio", ""))
        df["flaw_type"] = flaw_type
        records.append(df)
        matched.add(audio_id)

if not records:
    raise RuntimeError("No labeled feature files matched. Check IDs.")

dataset = pd.concat(records, ignore_index=True)
dataset = dataset.replace([np.inf, -np.inf], np.nan)

# Keep metadata separate from model input features.
dataset.to_csv(OUTPUT_FILE, index=False)

print("\nDataset creation complete")
print("Feature files found:", len(feature_files))
print("Feature files matched:", len(matched))
print("Missing label IDs:", len(missing_labels))
print("Frame rows:", len(dataset))
print("Output:", OUTPUT_FILE)

print("\nTarget distribution:")
print(dataset["target"].value_counts().to_string())

if missing_labels:
    print("\nUnmatched feature IDs (first 20):")
    print(missing_labels[:20])
