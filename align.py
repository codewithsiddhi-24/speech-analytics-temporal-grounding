import os
import re
import json
import torch
import torchaudio
import soundfile as sf

BASE = os.path.dirname(os.path.abspath(__file__))
AUDIO_FOLDER = os.path.join(BASE, "standardized_reference")
TRANS_FOLDER = os.path.join(BASE, "transcripts")      # corrected .txt files
OUT_FOLDER = os.path.join(BASE, "alignments")
LOW_SCORE = 0.50                                      # flag poorly aligned words

os.makedirs(OUT_FOLDER, exist_ok=True)
torch.set_num_threads(max(1, os.cpu_count() // 2))

bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).eval()
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

files = sorted(f for f in os.listdir(TRANS_FOLDER) if f.endswith(".txt"))
print(f"Found {len(files)} transcripts\n")

for tf in files:
    stem = os.path.splitext(tf)[0]
    audio_path = os.path.join(AUDIO_FOLDER, stem + ".wav")
    out_path = os.path.join(OUT_FOLDER, stem + ".json")

    if not os.path.exists(audio_path):
        print(f"SKIP {stem}: no matching wav in standardized_reference")
        continue
    if os.path.exists(out_path):
        print(f"SKIP {stem}: already aligned")
        continue

    print(f"Aligning: {stem}")

    # Words as written, plus a normalized form the aligner understands
    text = open(os.path.join(TRANS_FOLDER, tf), encoding="utf-8").read()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    raw_words = text.split()

    digit_words = [w for w in raw_words if any(c.isdigit() for c in w)]
    if digit_words:
        print(f"  WARNING: digits found, spell them out: {digit_words[:10]}")

    norm = [re.sub(r"[^a-z']", "", w.lower()) for w in raw_words]
    keep = [i for i, n in enumerate(norm) if n]

    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav).unsqueeze(0)
    if sr != bundle.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, bundle.sample_rate)

    with torch.inference_mode():
        emission, _ = model(wav)

    tokens = tokenizer([norm[i] for i in keep])
    word_spans = aligner(emission[0], tokens)

    ratio = wav.size(1) / emission.size(1) / bundle.sample_rate
    words, low = [], 0
    for idx, spans in zip(keep, word_spans):
        total = sum(len(s) for s in spans)
        score = sum(s.score * len(s) for s in spans) / total
        words.append({
            "index": len(words),
            "word": raw_words[idx],
            "start": round(spans[0].start * ratio, 3),
            "end": round(spans[-1].end * ratio, 3),
            "score": round(float(score), 3),
        })
        if score < LOW_SCORE:
            low += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"file": stem + ".wav",
                   "duration": round(wav.size(1) / bundle.sample_rate, 3),
                   "words": words}, f, indent=2, ensure_ascii=False)

    print(f"  words: {len(words)} | low-score: {low}")
    for w in [w for w in words if w["score"] < LOW_SCORE][:10]:
        print(f"    check '{w['word']}' at {w['start']}s (score={w['score']})")
    print()

print("Done. Alignments saved in /alignments")