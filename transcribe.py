import os
import json
from faster_whisper import WhisperModel

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE, "standardized_reference")
DRAFT_FOLDER = os.path.join(BASE, "transcripts_draft")
MODEL_SIZE = "medium.en"   # use "small.en" if your machine is slow
LOW_CONF = 0.60

os.makedirs(DRAFT_FOLDER, exist_ok=True)
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

files = sorted(f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".wav"))
print(f"Found {len(files)} files in {INPUT_FOLDER}\n")

for filename in files:
    stem = os.path.splitext(filename)[0]

    if os.path.exists(os.path.join(DRAFT_FOLDER, f"{stem}.json")):
        print(f"Skipping (already done): {filename}")
        continue

    print(f"Transcribing: {filename}")
    segments, info = model.transcribe(
        os.path.join(INPUT_FOLDER, filename),
        language="en",
        beam_size=5,
        temperature=0.0,
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=False,
    )

    seg_list, n_words, text_lines, flagged = [], 0, [], []
    for seg in segments:
        words = []
        for w in seg.words:
            wd = {"word": w.word.strip(), "start": round(w.start, 3),
                  "end": round(w.end, 3), "prob": round(w.probability, 3)}
            words.append(wd)
            n_words += 1
            if wd["prob"] < LOW_CONF:
                flagged.append(wd)
        seg_list.append({"start": round(seg.start, 3), "end": round(seg.end, 3),
                         "text": seg.text.strip(), "words": words})
        text_lines.append(seg.text.strip())

    with open(os.path.join(DRAFT_FOLDER, f"{stem}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))
    with open(os.path.join(DRAFT_FOLDER, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump({"file": filename, "model": MODEL_SIZE,
                   "duration": round(info.duration, 3), "segments": seg_list},
                  f, indent=2, ensure_ascii=False)

    print(f"  words: {n_words} | low-confidence: {len(flagged)}")
    for w in flagged[:10]:
        print(f"    check '{w['word']}' at {w['start']}s (p={w['prob']})")
    print()

print("Done. Review the .txt files, then save corrected versions to /transcripts")