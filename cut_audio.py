from pydub import AudioSegment

# Choose the part you want
start_seconds = 18
end_seconds = 50

# Open JFK audio
audio = AudioSegment.from_file("JFK_inaugural_address.ogg")

# Cut from 18 sec to 50 sec
clip = audio[start_seconds * 1000:end_seconds * 1000]

# Save the cut audio
clip.export("good_01.wav", format="wav")

print("Done! Created good_01.wav")