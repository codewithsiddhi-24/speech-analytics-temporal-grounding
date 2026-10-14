# Speech analytics and temporal grounding

This repository contains a local speech-review dashboard, audio/data
preparation utilities, and exploratory model-evaluation scripts. Start with
[the project file guide](Documentation/README.md) for the essential files,
runtime artifacts, and generated outputs.

## Verify original audio before treating it as a clean baseline

Run the original-audio review from the repository root:

```powershell
Set-Location D:\multimodal_TrackC
& .\.venv\Scripts\python.exe .\verify_original_audio.py
```

The script uses only the Python standard library. It inspects WAVs in
`clean_reference\` (the source recordings) and uses matching
`standardized_reference\` WAVs to check the copies used by feature extraction.
It checks PCM readability, duration, sample rate, channel count, sample width,
clipping, RMS/activity, edge silence, long internal pauses, and noise-floor
indicators. It also checks whether each standardized recording has a feature
summary/archive, transcript, and alignment, and compares their reported
durations and transcript words where possible.

The output is `original_audio_review.csv`. Re-running the analysis refreshes
automated measurements but preserves all human-review columns. It does not
modify source audio, features, labels, or the existing `audio_quality_report.csv`.

### Listen and record a human decision

Open an original recording in the system audio player:

```powershell
& .\.venv\Scripts\python.exe .\verify_original_audio.py --listen good_01
```

After listening, record one of the supported statuses. An unreviewed recording
starts as `PENDING_REVIEW`; passing automated checks never marks it good:

```powershell
& .\.venv\Scripts\python.exe .\verify_original_audio.py --review good_01 --status VERIFIED_GOOD --reviewed-by "Your Name" --notes "Listened end to end; no audible issue found."
& .\.venv\Scripts\python.exe .\verify_original_audio.py --review good_02 --status VERIFIED_FLAWED --reviewed-by "Your Name" --notes "Clipping is audible near the end." --flaw-type clipping --flaw-start 112.4 --flaw-end 118.0
& .\.venv\Scripts\python.exe .\verify_original_audio.py --review good_03 --status UNCERTAIN --reviewed-by "Your Name" --notes "Background sound needs a second listen."
& .\.venv\Scripts\python.exe .\verify_original_audio.py --review good_04 --status PENDING_REVIEW --notes "Listen on headphones."
```

`--flaw-type`, `--flaw-start`, and `--flaw-end` are optional for
`VERIFIED_FLAWED`. Supply both timestamps together when an approximate segment
can be identified. Re-run the analysis command after updating statuses to
refresh measurements; existing decisions, reviewer, notes, and flaw timestamps
are retained. Review each full recording before marking the whole recording
`VERIFIED_GOOD`. Automated activity ratios and pause flags are cues for
listening, not proof that natural pauses or quiet speech are flaws.

### Automated thresholds and interpretation

The review uses explicit starting thresholds already present in this project:

| Automated cue | Threshold | Origin and interpretation |
| --- | --- | --- |
| Clipping | Samples at or above 0.99 full scale; flag if more than 1% of samples meet it | `check_quality.py`; an indicator to listen for distortion, not an automatic rejection |
| Energy activity | 25 ms non-overlapping RMS windows within 35 dB of that file's loudest window; flag below 20% | Reuses the relative-silence approach and minimum speech ratio in `check_quality.py`; it is an energy proxy, not a speech recognizer |
| Leading/trailing silence | More than 3 seconds | Existing `check_quality.py` cue; silence may be intentional |
| Long internal pause | At least 3 seconds | Existing `inspect_long_pauses.py` cue; pauses may be natural |
| Low overall level | RMS below -35 dBFS | Existing `check_quality.py` low-volume cue; quiet speech is not automatically bad |
| Duration | Outside 2–600 seconds | Existing `check_quality.py` scan range, not a universal standard |
| Feature/audio duration | Difference over 0.05 seconds | Local consistency check for the 16 kHz standardized audio and saved feature summary |

`noise_indicator_dbfs` is the 10th percentile of 25 ms frame RMS values. This is
a rough low-level-energy indicator, not a calibrated SNR or proof of background
noise. All thresholds are review cues; inspect the recording and retain the
human status separately from `automated_flags`.

### Current dataset and evaluation cautions

- The repository has 14 recordings in `clean_reference\` and matching
  16 kHz mono copies in `standardized_reference\`. Feature archives/summaries,
  transcripts, and alignments are present for these recordings. The review
  script checks their integrity/consistency; it does not infer cleanliness
  from their presence.
- The older `audio_quality_report.csv` was generated from `audio_inp\`, not
  `clean_reference\` or `standardized_reference\`; its PASS rows are not human
  verification of the current baseline.
- Every current row in `flaw_labels.csv` is marked synthetic and has a
  `source_audio`; the 168 rows describe transformations applied to 14 source
  recordings. Synthetic transformation labels do not verify the source as
  clean.
- `build_training_dataset.py` assigns `normal` to frames outside the inserted
  pause in each synthetic pause variant. Those are normal-labelled frames
  from the source recordings, not human-confirmed clean segments. Use the
  optional full export scan below to report exact current frame counts:

```powershell
& .\.venv\Scripts\python.exe .\verify_original_audio.py --audit-training-frames
```

This scans the full `training_frames.csv` (currently about 2.2 GB) and can take
some time. It reports target counts and sources contributing `normal` rows.

The repository includes a grouped model-comparison script. It keeps every
recording and its synthetic variants together within a cross-validation fold,
but only 14 independent source recordings are available. The labels are
synthetic transformations, not verified natural speech flaws, so these results
do not establish accuracy on naturally occurring speech issues or new uploads.

## Compare models without source leakage

The optional model-comparison script reads the existing frame-level
`training_frames.csv`. Install its two dependencies into the project virtual
environment, then run the comparison from the repository root:

```powershell
Set-Location D:\multimodal_TrackC
& .\.venv\Scripts\python.exe -m pip install numpy scikit-learn
& .\.venv\Scripts\python.exe .\train_compare_models.py
```

It streams the CSV and selects at most the configured number of frames per `source_audio` and
`target` stratum using a fixed seed. `time_sec` is excluded to prevent the
inserted-pause location from acting as a shortcut. It compares logistic
regression, an RBF-kernel SVM, Random Forest, Extra Trees, and histogram
gradient boosting using four-fold `GroupKFold`, with every frame and synthetic
variant of one source kept in the same fold. It reports pooled out-of-fold
accuracy, balanced accuracy, macro F1, weighted F1, per-fold scores, class
metrics, and confusion matrices. Imputation and scaling (where used) are
fitted inside each training fold.

The candidate with the highest pooled out-of-fold balanced accuracy (then
macro F1, then accuracy) is refit on the sampled rows. The report, metrics, and
refit model are generated artifacts and are ignored by Git at the default
output paths. Use `--metrics`, `--report`, and `--model` to write them to
explicit locations when retaining a new experiment.

The retained 56,000-row, four-fold source-grouped experiment produced these
pooled out-of-fold scores:

| Model | Accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Histogram gradient boosting | 72.80% | 72.80% | 72.46% |
| Random Forest | 69.62% | 69.62% | 68.54% |
| Extra Trees | 66.94% | 66.94% | 65.38% |
| RBF SVM | 66.69% | 66.69% | 66.54% |
| Logistic regression | 57.18% | 57.18% | 55.77% |

The detailed retained evaluation is in
`reports/model_evaluation_grouped_cap500.json` and
`reports/model_evaluation_grouped_cap500.csv`. The refitted experimental model
is not used by the upload dashboard and is intentionally not retained.
Histogram gradient boosting was the top candidate by the stated selection
rule. These are measured results for this sample and split, not a promise of
upload-time performance.

These scores are exploratory: there are only 14 independent source recordings,
the classes are synthetic transformation classes, and the `normal` samples
include source frames that have not been human-verified. The split prevents
same-source frame/variant leakage, but does not validate naturally occurring
flaw detection or generalization to speakers/devices not represented here.
Use this comparison to find a candidate, not to assert production accuracy.

## Run focused tests

```powershell
Set-Location D:\multimodal_TrackC
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Run the local upload dashboard

The dashboard opens to an empty upload workspace; it does not ask the viewer to
select a reference recording. Each English WAV is transcribed and analyzed
locally, then compared automatically with the saved multi-recording reference
model. Word timestamps, reference-deviation candidates, gaps, low-energy
intervals, and audio visualizations all use the uploaded recording. Install the
optional speech-recognition dependency. To refit the timestamp-review baseline
across all recordings that have audio, alignments, and acoustic features,
regenerate its local dashboard artifacts with `--overwrite`, then start the
upload server:

```powershell
Set-Location D:\multimodal_TrackC
& .\.venv\Scripts\python.exe -m pip install -r .\requirements-upload.txt
& .\.venv\Scripts\python.exe .\train_timestamp_review_model.py --overwrite
& .\.venv\Scripts\python.exe .\local_upload_server.py
```

Open `http://127.0.0.1:8765/` and select a 16-bit PCM `.wav`; analysis starts
automatically. The reference model is loaded behind the scenes, and the upload
is scored without copying any reference recording's transcript or timestamps.
After analysis, the dashboard shows the uploaded recording's transcript, audio
timeline, charts, interval explanations, and reference-deviation candidates.
The first upload downloads the configured `small.en` model; it runs on CPU and
is English-only. Set `SPEECHLENS_WHISPER_MODEL` to another faster-whisper model
name before starting the server if needed. `requirements-upload.txt` includes
the transcription and reference-scoring dependencies and constrains PyAV to a
version compatible with faster-whisper. The upload is held in a temporary file
only while it is processed, then deleted. The server binds to localhost and
accepts uploads up to 100 MB.

Uploaded word timestamps and recognition confidence come from faster-whisper;
they are not forced-alignment scores. The reference model was fit on aligned
source recordings, so upload-time scoring imputes missing forced-alignment
features and should be interpreted as an acoustic/timing review priority, not
as an equivalent validated alignment score. Inter-word gap labels are acoustic review
cues: a quiet gap may be a natural pause, breath, or silence, while continuing
energy may be breath, mumbling, or speech the recognizer missed. The signal
heuristic cannot reliably distinguish these causes, so listen to each segment.
Low recognition confidence, fillers/repetitions, slow/fast local articulation,
and low pitch variation can appear as interval explanation cards with transcript
context and measured cues. Low-energy frames inside recognized word times are
excluded from pause findings, and a quiet inter-word gap is listed only once.
Adjacent findings of the same type are grouped into phrase-sized intervals;
timestamps remain the measured interval, while playback includes up to 0.25
seconds of listening context on either side. Pitch and voice-level cards compare
voiced frames with the same upload's median, and pace is computed over
multiword phrases. These categories are heuristics, not trained natural-flaw
labels; low recognition confidence is not a diagnosis.
The upload dashboard also shows three percentage metrics: mean recognizer
confidence, the share of words at or above the 0.45 confidence review
threshold, and the share of multiword phrase windows within 90–190 recognized
words/min. They describe this upload and are not model accuracy or a validated
overall speech-quality grade. The spectrogram uses 128 frequency bands with a
time/frequency heatmap for easier visual inspection.

The reference-deviation model now fits word examples from every source
recording with matching audio/features/alignment instead of just `good_15`.
Those recordings are still `PENDING_REVIEW`, so this is broader unsupervised
reference coverage—not verified-clean training data or an accuracy claim.

The grouped frame classifier is a separate experiment on synthetic
transformation labels. The current best source-grouped result is 72.80%
out-of-fold accuracy (balanced accuracy 72.80%, macro F1 72.46%) across only
14 independent source recordings. This is below 80%, and does not measure
natural flaw detection or uploaded-audio accuracy. The original recordings
remain pending human review; model scores must not be presented as confirmed
correctness. More recordings and human-verified timestamp/gap labels are
needed to test and train a reliable natural-flaw classifier.
