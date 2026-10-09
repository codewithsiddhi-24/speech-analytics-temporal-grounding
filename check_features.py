import numpy as np
from pathlib import Path

FEAT_DIR = Path("features")

KEYS = {
    "time": "time",
    "f0": "f0_hz",
    "rms_db": "rms_db",
    "mfcc": "mfcc",
}


def check(stem):
    file_path = FEAT_DIR / f"{stem}_frames.npz"

    z = np.load(file_path)

    issues = []

    time = z[KEYS["time"]]
    f0 = z[KEYS["f0"]]
    rms_db = z[KEYS["rms_db"]]
    mfcc = z[KEYS["mfcc"]]

    n = len(time)

    # --------------------------------------------------
    # 1. Check array lengths
    # --------------------------------------------------
    if not (
        len(f0) == len(rms_db) == n
        and mfcc.shape[-1] == n
    ):
        issues.append(
            f"length mismatch: "
            f"time={n}, "
            f"f0={len(f0)}, "
            f"rms={len(rms_db)}, "
            f"mfcc={mfcc.shape}"
        )

    # --------------------------------------------------
    # 2. Check time axis
    # --------------------------------------------------
    if n > 1:
        time_diff = np.diff(time)

        if not np.allclose(
            time_diff,
            0.01,
            atol=1e-3
        ):
            issues.append(
                "time axis is not a uniform 10 ms grid"
            )

    # --------------------------------------------------
    # 3. Check NaN / infinite values
    # --------------------------------------------------
    if np.isnan(rms_db).any():
        issues.append(
            "NaN found in energy"
        )

    if np.isnan(mfcc).any():
        issues.append(
            "NaN found in MFCC"
        )

    if not np.isfinite(rms_db).all():
        issues.append(
            "infinite value found in energy"
        )

    if not np.isfinite(mfcc).all():
        issues.append(
            "infinite value found in MFCC"
        )

    # --------------------------------------------------
    # 4. F0 / voiced-frame check
    # --------------------------------------------------
    voiced = np.isfinite(f0) & (f0 > 0)

    voiced_ratio = voiced.mean()

    if voiced_ratio < 0.20:
        issues.append(
            f"very low voiced ratio {voiced_ratio:.2f}"
        )

    # --------------------------------------------------
    # 5. F0 plausible range
    # --------------------------------------------------
    if voiced.any():

        voiced_f0 = f0[voiced]

        f0_min = voiced_f0.min()
        f0_max = voiced_f0.max()

        if f0_min < 60 or f0_max > 600:
            issues.append(
                f"F0 outside plausible range: "
                f"{f0_min:.0f} to {f0_max:.0f} Hz"
            )

        # --------------------------------------------------
        # 6. Large consecutive pitch jumps
        # --------------------------------------------------
        voiced_indices = np.where(voiced)[0]

        if len(voiced_indices) > 1:

            consecutive = (
                voiced_indices[1:]
                == voiced_indices[:-1] + 1
            )

            if consecutive.any():

                f0_a = f0[voiced_indices[:-1]][consecutive]
                f0_b = f0[voiced_indices[1:]][consecutive]

                semitone_change = (
                    12
                    * np.log2(f0_b / f0_a)
                )

                big_jumps = int(
                    (np.abs(semitone_change) > 7).sum()
                )

                if big_jumps > 0.01 * voiced.sum():
                    issues.append(
                        f"{big_jumps} pitch jumps > "
                        f"7 semitones "
                        f"(possible octave errors)"
                    )

    # --------------------------------------------------
    # Result
    # --------------------------------------------------
    return {
        "audio": stem,
        "frames": n,
        "voiced_ratio": round(
            float(voiced_ratio),
            2
        ),
        "issues": issues if issues else ["OK"],
    }


# ======================================================
# MAIN
# ======================================================

def main():

    if not FEAT_DIR.exists():
        print(
            f"ERROR: features folder not found: "
            f"{FEAT_DIR.resolve()}"
        )
        return

    files = sorted(
        FEAT_DIR.glob("*_frames.npz")
    )

    if not files:
        print(
            f"ERROR: No *_frames.npz files found "
            f"in {FEAT_DIR.resolve()}"
        )
        return

    print("=" * 75)
    print("FEATURE INTEGRITY CHECK")
    print("=" * 75)
    print()

    total = 0
    problems = 0

    for file in files:

        stem = file.name.replace(
            "_frames.npz",
            ""
        )

        try:
            result = check(stem)

            print(
                f"{result['audio']:10s} "
                f"{result['frames']:6d} frames | "
                f"voiced={result['voiced_ratio']:.2f} | "
                f"{'; '.join(result['issues'])}"
            )

            total += 1

            if result["issues"] != ["OK"]:
                problems += 1

        except Exception as e:

            print(
                f"{stem:10s} ERROR | {e}"
            )

            total += 1
            problems += 1

    print()
    print("=" * 75)
    print(f"Files checked : {total}")
    print(f"Files with issues : {problems}")
    print("=" * 75)


if __name__ == "__main__":
    main()