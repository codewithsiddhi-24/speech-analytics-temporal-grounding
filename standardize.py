import os
import librosa
import soundfile as sf

INPUT_FOLDER = "clean_reference"
OUTPUT_FOLDER = "standardized_reference"

TARGET_SR = 16000

print("=" * 70)
print("STANDARDIZING REFERENCE AUDIO")
print("=" * 70)
print()

if not os.path.exists(INPUT_FOLDER):
    print(f"ERROR: Cannot find folder: {INPUT_FOLDER}")
    raise SystemExit

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg"))
]

print(f"Input files found: {len(files)}")
print(f"Target sample rate: {TARGET_SR} Hz")
print("Target channels: Mono")
print()

success = []
failed = []

for filename in sorted(files):

    input_path = os.path.join(INPUT_FOLDER, filename)

    output_name = os.path.splitext(filename)[0] + ".wav"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    print(f"Processing: {filename}")

    try:
        # Load as mono and resample to 16 kHz
        audio, sr = librosa.load(
            input_path,
            sr=TARGET_SR,
            mono=True
        )

        # Save standardized WAV
        sf.write(
            output_path,
            audio,
            TARGET_SR,
            subtype="PCM_16"
        )

        duration = len(audio) / TARGET_SR

        print(f"  Original/loaded SR: {sr} Hz")
        print(f"  Output SR:          {TARGET_SR} Hz")
        print(f"  Channels:           Mono")
        print(f"  Duration:           {duration:.2f} sec")
        print("  -> SUCCESS")
        print()

        success.append(output_name)

    except Exception as e:

        print(f"  -> ERROR: {e}")
        print()

        failed.append(filename)


print("=" * 70)
print("STANDARDIZATION COMPLETE")
print("=" * 70)
print()

print(f"Input files : {len(files)}")
print(f"Successful  : {len(success)}")
print(f"Failed      : {len(failed)}")
print()

if success:
    print("STANDARDIZED FILES")
    print("-" * 40)

    for filename in success:
        print(f"  {filename}")

    print()

if failed:
    print("FAILED FILES")
    print("-" * 40)

    for filename in failed:
        print(f"  {filename}")

    print()

print("Output folder:")
print(os.path.abspath(OUTPUT_FOLDER))

print()
print("=" * 70)
print("DONE")
print("=" * 70)