"""Compare frame classifiers using source-grouped cross-validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "training_frames.csv"
METRICS_PATH = BASE_DIR / "model_evaluation_grouped.csv"
REPORT_PATH = BASE_DIR / "model_evaluation_grouped.json"
MODEL_PATH = BASE_DIR / "best_model_grouped.joblib"
MAX_FRAMES_PER_SOURCE_CLASS = 150
N_SPLITS = 4
RANDOM_SEED = 20261011
EXCLUDED_FEATURES = {"time_sec"}


def load_group_balanced_sample(
    path: Path,
    max_per_source_class: int = MAX_FRAMES_PER_SOURCE_CLASS,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Stream the CSV and reservoir-sample a bounded count in each stratum."""
    if max_per_source_class < 1:
        raise ValueError("max_per_source_class must be at least 1")
    if not path.is_file():
        raise FileNotFoundError(f"Training frame CSV not found: {path}")

    rng = random.Random(seed)
    total_counts: Counter[tuple[str, str]] = Counter()
    reservoirs: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    feature_names: list[str] | None = None
    total_rows = 0

    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Training CSV has no header: {path}")
        required = {"target", "source_audio", "audio_id"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Training CSV is missing required columns: {sorted(missing)}"
            )

        feature_names = [
            name
            for name in reader.fieldnames
            if name not in EXCLUDED_FEATURES
            and name not in {"target", "source_audio", "audio_id", "flaw_type"}
        ]
        if not feature_names:
            raise ValueError("Training CSV contains no model features.")

        for row in reader:
            total_rows += 1
            target = row.get("target", "").strip()
            source = Path(row.get("source_audio", "").strip()).name
            if not target or not source:
                raise ValueError(
                    f"Training row {total_rows + 1} has a blank target/source_audio."
                )
            stratum = (source, target)
            seen = total_counts[stratum]
            total_counts[stratum] += 1

            reservoir = reservoirs[stratum]
            if len(reservoir) >= max_per_source_class:
                selected = rng.randrange(seen + 1)
                if selected >= max_per_source_class:
                    continue
                slot = selected
            else:
                slot = len(reservoir)

            values: list[float] = []
            for feature_name in feature_names:
                value = row.get(feature_name, "")
                if value is None or value.strip() == "":
                    values.append(float("nan"))
                    continue
                try:
                    numeric = float(value)
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric feature {feature_name!r} in CSV row "
                        f"{total_rows + 1}: {value!r}"
                    ) from exc
                values.append(numeric if math.isfinite(numeric) else float("nan"))

            if slot == len(reservoir):
                reservoir.append(values)
            else:
                reservoir[slot] = values

    if feature_names is None or not reservoirs:
        raise ValueError("Training CSV contains no sampleable data rows.")

    feature_rows: list[list[float]] = []
    labels: list[str] = []
    groups: list[str] = []
    sampled_counts: Counter[tuple[str, str]] = Counter()
    for (source, target), rows in sorted(reservoirs.items()):
        feature_rows.extend(rows)
        labels.extend([target] * len(rows))
        groups.extend([source] * len(rows))
        sampled_counts[(source, target)] = len(rows)

    x = np.asarray(feature_rows, dtype=np.float64)
    y = np.asarray(labels, dtype=str)
    group_array = np.asarray(groups, dtype=str)
    if not np.isfinite(x).any():
        raise ValueError("Every sampled feature is missing or non-finite.")

    source_names = sorted(set(groups))
    class_names = sorted(set(labels))
    if len(source_names) < N_SPLITS:
        raise ValueError(
            f"Need at least {N_SPLITS} distinct source recordings for "
            f"{N_SPLITS}-fold GroupKFold; found {len(source_names)}."
        )
    absent_strata = [
        (source, target)
        for source in source_names
        for target in class_names
        if sampled_counts[(source, target)] == 0
    ]
    if absent_strata:
        raise ValueError(
            "At least one source/class stratum has no sampled rows: "
            f"{absent_strata[:10]}"
        )

    audit = {
        "total_csv_rows": total_rows,
        "sampled_rows": len(y),
        "source_recordings": source_names,
        "class_names": class_names,
        "max_frames_per_source_class": max_per_source_class,
        "sampled_rows_per_class": dict(Counter(y)),
        "sampled_rows_per_source_class": {
            f"{source}|{target}": count
            for (source, target), count in sorted(sampled_counts.items())
        },
        "source_class_rows_in_full_csv": {
            f"{source}|{target}": count
            for (source, target), count in sorted(total_counts.items())
        },
        "features": feature_names,
        "excluded_features": sorted(EXCLUDED_FEATURES),
        "sampling": (
            "Seeded reservoir sample capped independently for each "
            "source_audio and target pair."
        ),
    }
    return x, y, group_array, feature_names, audit


def build_models(seed: int) -> dict[str, Any]:
    """Return a small comparison set; all imputation is fit inside each fold."""
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
            ),
        ),
        "rbf_svm": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(
                C=2.0,
                kernel="rbf",
                class_weight="balanced",
                cache_size=512,
            ),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=160,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=160,
                class_weight="balanced",
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_iter=120,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=seed,
            ),
        ),
    }


def evaluate_models(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    splitter = GroupKFold(n_splits=N_SPLITS)
    models = build_models(seed)
    class_names = sorted(set(y.tolist()))
    oof_predictions = {
        name: np.full(y.shape, "", dtype=object) for name in models
    }
    fold_rows: list[dict[str, Any]] = []
    split_audit: list[dict[str, Any]] = []

    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(x, y, groups), start=1
    ):
        train_groups = set(groups[train_indices])
        test_groups = set(groups[test_indices])
        if train_groups & test_groups:
            raise RuntimeError(f"Source leakage detected in fold {fold_number}.")
        missing_train_classes = set(class_names) - set(y[train_indices])
        missing_test_classes = set(class_names) - set(y[test_indices])
        if missing_train_classes or missing_test_classes:
            raise RuntimeError(
                f"Fold {fold_number} lacks classes: "
                f"train={sorted(missing_train_classes)}, "
                f"test={sorted(missing_test_classes)}"
            )
        split_audit.append(
            {
                "fold": fold_number,
                "train_groups": sorted(train_groups),
                "test_groups": sorted(test_groups),
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
                "overlap": [],
            }
        )

        for name, model in models.items():
            model.fit(x[train_indices], y[train_indices])
            predictions = model.predict(x[test_indices])
            oof_predictions[name][test_indices] = predictions
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold_number,
                    "train_groups": ";".join(sorted(train_groups)),
                    "test_groups": ";".join(sorted(test_groups)),
                    "train_rows": len(train_indices),
                    "test_rows": len(test_indices),
                    "accuracy": accuracy_score(y[test_indices], predictions),
                    "balanced_accuracy": balanced_accuracy_score(
                        y[test_indices], predictions
                    ),
                    "macro_f1": f1_score(
                        y[test_indices],
                        predictions,
                        average="macro",
                        zero_division=0,
                    ),
                    "weighted_f1": f1_score(
                        y[test_indices],
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                }
            )
            print(
                f"fold={fold_number}/{N_SPLITS} model={name} "
                f"accuracy={fold_rows[-1]['accuracy']:.4f} "
                f"balanced_accuracy={fold_rows[-1]['balanced_accuracy']:.4f} "
                f"macro_f1={fold_rows[-1]['macro_f1']:.4f}",
                flush=True,
            )

    for name, predictions in oof_predictions.items():
        if np.any(predictions == ""):
            raise RuntimeError(f"Missing out-of-fold predictions for {name}.")

    ranking: dict[str, Any] = {}
    for name, predictions in oof_predictions.items():
        ranking[name] = {
            "accuracy": float(accuracy_score(y, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
            "macro_f1": float(
                f1_score(y, predictions, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(y, predictions, average="weighted", zero_division=0)
            ),
            "per_class": classification_report(
                y,
                predictions,
                labels=class_names,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y, predictions, labels=class_names
            ).tolist(),
            "out_of_fold_predictions": predictions,
        }

    best_name = max(
        ranking,
        key=lambda name: (
            ranking[name]["balanced_accuracy"],
            ranking[name]["macro_f1"],
            ranking[name]["accuracy"],
        ),
    )
    return fold_rows, ranking, {
        "splits": split_audit,
        "best_model_name": best_name,
        "class_names": class_names,
    }


def train_best_model(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    model_path: Path,
) -> Any:
    model = build_models(seed)[model_name]
    model.fit(x, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "fold",
        "train_groups",
        "test_groups",
        "train_rows",
        "test_rows",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing evaluation metrics: {path}"
        )
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(
    data_path: Path = DATA_PATH,
    metrics_path: Path = METRICS_PATH,
    report_path: Path = REPORT_PATH,
    model_path: Path = MODEL_PATH,
    cap: int = MAX_FRAMES_PER_SOURCE_CLASS,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    for output_path in (metrics_path, report_path, model_path):
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {output_path}"
            )
    print(f"Streaming training data from {data_path}", flush=True)
    x, y, groups, feature_names, sampling_audit = load_group_balanced_sample(
        data_path, max_per_source_class=cap, seed=seed
    )
    print(
        f"Sampled {len(y)} rows from {len(set(groups))} sources; "
        f"classes={sampling_audit['class_names']}",
        flush=True,
    )
    fold_rows, ranking, split_info = evaluate_models(x, y, groups, seed)

    fold_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        fold_metrics[row["model"]].append(row)

    result_summary = {
        name: {
            metric: {
                "mean": float(np.mean([row[metric] for row in rows])),
                "std": float(np.std([row[metric] for row in rows])),
            }
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
            )
        }
        for name, rows in fold_metrics.items()
    }
    best_name = split_info["best_model_name"]
    train_best_model(best_name, x, y, seed, model_path)

    serializable_ranking = {}
    for name, values in ranking.items():
        serializable_ranking[name] = {
            key: value
            for key, value in values.items()
            if key != "out_of_fold_predictions"
        }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_method": (
            f"{N_SPLITS}-fold GroupKFold grouped by source_audio; all reported "
            "metrics are out-of-fold predictions for the sampled rows."
        ),
        "selection_criterion": (
            "Highest pooled out-of-fold balanced accuracy, then macro F1, "
            "then accuracy."
        ),
        "warning": (
            "Only 14 independent source recordings are available. Scores are "
            "exploratory and do not establish generalization to new speakers, "
            "devices, environments, or naturally occurring flaws. The "
            "synthetic labels indicate applied transformations, not verified "
            "natural flaw labels. A fixed cap is sampled per source and class."
        ),
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "sampling": sampling_audit,
        "folds": split_info["splits"],
        "pooled_out_of_fold_metrics": serializable_ranking,
        "per_fold_summary": result_summary,
        "best_model_name": best_name,
        "model_file": str(model_path),
        "model_training_rows": int(len(y)),
        "model_features": feature_names,
        "model_note": (
            "The selected candidate is refit on all sampled rows after "
            "cross-validation. It is a research candidate, not a model "
            "validated for production."
        ),
    }

    _write_metrics(metrics_path, fold_rows)
    if report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {report_path}"
        )
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("\nPOOLED OUT-OF-FOLD RESULTS")
    for name, values in sorted(
        ranking.items(),
        key=lambda item: (
            item[1]["balanced_accuracy"],
            item[1]["macro_f1"],
            item[1]["accuracy"],
        ),
        reverse=True,
    ):
        print(
            f"{name}: accuracy={values['accuracy']:.4f}, "
            f"balanced_accuracy={values['balanced_accuracy']:.4f}, "
            f"macro_f1={values['macro_f1']:.4f}, "
            f"weighted_f1={values['weighted_f1']:.4f}"
        )
    print(f"Selected candidate: {best_name}")
    print(f"Saved grouped CV details: {metrics_path}")
    print(f"Saved evaluation report: {report_path}")
    print(f"Saved refit candidate: {model_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare five classifiers with source-grouped cross-validation "
            "on a bounded reproducible sample of training_frames.csv."
        )
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--cap-per-source-class",
        type=int,
        default=MAX_FRAMES_PER_SOURCE_CLASS,
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    try:
        run(
            args.data,
            args.metrics,
            args.report,
            args.model,
            args.cap_per_source_class,
            args.seed,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
