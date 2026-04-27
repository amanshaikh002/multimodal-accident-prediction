import os
import moviepy.editor as mp

VIDEO_FOLDER = "data/videos"
OUTPUT_FOLDER = "data/audio/raw_audio"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for file in os.listdir(VIDEO_FOLDER):
    if file.endswith(".mp4"):
        video_path = os.path.join(VIDEO_FOLDER, file)
        audio_path = os.path.join(OUTPUT_FOLDER, file.replace(".mp4", ".wav"))

        print(f"Processing: {file}")

        video = mp.VideoFileClip(video_path)
        audio = video.audio

        if audio is not None:
            audio.write_audiofile(audio_path)
        else:
            print("No audio found in:", file)

print("✅ All audio extracted!")