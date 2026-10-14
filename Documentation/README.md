# Project file guide

Use the repository-root [README](../README.md) for setup and workflow
instructions. This guide identifies the files needed for the upload dashboard,
model experiments, and dataset review.

## Essential upload-dashboard files

| File | Purpose |
| --- | --- |
| `local_upload_server.py` | Local HTTP server, upload endpoint, transcription, acoustic measurements, and interval explanations |
| `dashboard_template.html` | Upload-first dashboard interface, scores, charts, and playback controls |
| `requirements-upload.txt` | Optional dependencies for upload transcription and reference comparison |
| `reports/timestamp_review_baseline.json` | Dashboard's reference recording data and descriptive analysis |
| `reports/timestamp_review_baseline_model.joblib` | Isolation Forest artifact loaded to compare uploaded audio with reference features |
| `train_timestamp_review_model.py` | Rebuilds the reference report and Isolation Forest from available reference features, alignments, transcripts, and audio |
| `tests/test_local_upload_server.py` | Upload processing and dashboard regression tests |

The two files under `reports/` named `timestamp_review_baseline...` are runtime
inputs, not disposable experiment outputs. Keep them with the application, or
rebuild them before launch using the command in the root README.

## Data preparation and evaluation

| File or folder | Purpose |
| --- | --- |
| `clean_reference/` | Original source recordings; preserve these inputs |
| `standardized_reference/` | Standardized WAV inputs used by feature extraction and the dashboard |
| `transcripts/`, `alignments/`, `features/` | Transcript, word alignment, and acoustic feature data |
| `generated_flaws/`, `generated_flaw_features/` | Synthetic transformed examples and their extracted features |
| `flaw_labels.csv` | Labels and source mapping for the synthetic transformations |
| `standardize.py`, `transcribe.py`, `align.py` | Audio normalization, transcription, and forced alignment |
| `extract_features.py`, `extract_pause_features.py`, `extract_generated_features.py` | Feature extraction for source and transformed recordings |
| `generate_flaws.py`, `build_training_dataset.py`, `export_features_csv.py` | Create synthetic variants and frame-level training exports |
| `train_compare_models.py` | Source-grouped comparison of experimental frame classifiers |
| `reports/model_evaluation_grouped_cap500.json` | Retained detailed metrics for the 56,000-row grouped comparison |
| `reports/model_evaluation_grouped_cap500.csv` | Per-fold metrics for that comparison |
| `verify_original_audio.py`, `check_quality.py`, `alignment_quality.py` | Audio and alignment review utilities |
| `tests/` | Regression tests for model preparation, uploads, and original-audio review |

The generated classifier artifacts from the experiment are not used by the
upload dashboard and are ignored. The retained evaluation report is useful for
reproducing the documented comparison results. The experimental labels are
synthetic and do not establish performance on natural speech flaws.

## Other documentation and generated files

`SonicSense-speechlens.docx` is the supplied project overview document. Keep it
with this guide when distributing the repository.

Large CSV exports such as `training_frames.csv` and `features_frames.csv` are
reproducible intermediate data and are ignored by Git. The original and
synthetic audio, feature archives, transcripts, alignments, and labels are
preserved as requested; do not delete them as part of routine cleanup.

The `.gitignore` file prevents virtual environments, caches, large regenerated
exports, and unused experiment model artifacts from being added. Ignore rules
do not remove files that Git already tracks.
