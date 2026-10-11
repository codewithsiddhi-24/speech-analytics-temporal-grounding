
from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_DIR = Path("features")
SUMMARY_OUT = Path("features_summary.csv")
FRAMES_OUT = Path("features_frames.csv")

summary_rows = []
frame_tables = []

for path in sorted(FEATURE_DIR.glob("*_frames.npz")):
    audio_id = path.stem.replace("_frames", "")

    with np.load(path, allow_pickle=False) as data:
        times = data["time"]
        n_frames = len(times)

        # Frame-level features: one row per 10 ms frame
        frame_data = {
            "audio_id": np.repeat(audio_id, n_frames),
            "time_sec": times,
        }

        # One-dimensional features
        for name in data.files:
            arr = data[name]

            if name == "time":
                continue

            if arr.ndim == 1 and len(arr) == n_frames:
                frame_data[name] = arr

            # Expand MFCCs into separate columns
            elif arr.ndim == 2 and arr.shape[1] == n_frames:
                for i in range(arr.shape[0]):
                    frame_data[f"{name}_{i + 1:02d}"] = arr[i]

        frame_df = pd.DataFrame(frame_data)
        frame_tables.append(frame_df)

        # Recording-level summary statistics
        row = {
            "audio_id": audio_id,
            "duration_sec": float(times[-1]) if n_frames else 0.0,
            "frame_count": n_frames,
        }

        for name in data.files:
            arr = data[name]

            if name == "time" or arr.ndim != 1:
                continue

            if arr.dtype == np.bool_:
                row[f"{name}_ratio"] = float(np.mean(arr))
                continue

            values = arr.astype(float)
            finite = values[np.isfinite(values)]

            row[f"{name}_valid_ratio"] = (
                float(len(finite) / len(values)) if len(values) else 0.0
            )

            if len(finite):
                row[f"{name}_mean"] = float(np.mean(finite))
                row[f"{name}_std"] = float(np.std(finite))
                row[f"{name}_median"] = float(np.median(finite))

        # MFCC summary statistics
        for name in ("mfcc", "mfcc_delta", "mfcc_delta2"):
            if name in data.files:
                arr = data[name]
                for i, coeff in enumerate(arr):
                    finite = coeff[np.isfinite(coeff)]
                    if len(finite):
                        row[f"{name}_{i + 1:02d}_mean"] = float(
                            np.mean(finite)
                        )
                        row[f"{name}_{i + 1:02d}_std"] = float(
                            np.std(finite)
                        )

        summary_rows.append(row)

if not frame_tables:
    raise SystemExit(
        "No *_frames.npz files found in the features folder."
    )

pd.concat(frame_tables, ignore_index=True).to_csv(
    FRAMES_OUT, index=False
)

pd.DataFrame(summary_rows).to_csv(
    SUMMARY_OUT, index=False
)

print(f"Processed recordings: {len(summary_rows)}")
print(f"Frame-level CSV: {FRAMES_OUT}")
print(f"Summary CSV: {SUMMARY_OUT}")
print(f"Total frames exported: {sum(len(x) for x in frame_tables)}")
