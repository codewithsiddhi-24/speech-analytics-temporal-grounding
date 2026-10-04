import parselmouth
import matplotlib.pyplot as plt

import numpy as np
import librosa
import matplotlib.pyplot as plt

audio_path = "good_01.wav"

audio, sample_rate = librosa.load(audio_path, sr=None)
# Extract pitch (F0)
sound = parselmouth.Sound(audio_path)

pitch = sound.to_pitch()
pitch_values = pitch.selected_array["frequency"]

pitch_times = pitch.xs()

# Remove unvoiced regions
pitch_values[pitch_values == 0] = np.nan

print("Average Pitch:", np.nanmean(pitch_values), "Hz")
print("Minimum Pitch:", np.nanmin(pitch_values), "Hz")
print("Maximum Pitch:", np.nanmax(pitch_values), "Hz")

duration = len(audio) / sample_rate

print("Sample Rate:", sample_rate, "Hz")
print("Number of Samples:", len(audio))
print("Duration:", round(duration, 2), "seconds")

# Calculate energy
energy = librosa.feature.rms(y=audio)[0]

# Convert energy frames to time
times = librosa.frames_to_time(
    range(len(energy)),
    sr=sample_rate
)

# Plot energy
plt.figure(figsize=(12, 4))
plt.plot(times, energy)
plt.xlabel("Time (seconds)")
plt.ylabel("Energy")
plt.title("Speech Energy Over Time")
plt.grid()
plt.show()