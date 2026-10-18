import os
from scenedetect import detect, ContentDetector

folder = "data/videos"
files = [f for f in os.listdir(folder) if f.endswith((".avi", ".mp4"))]
video = os.path.join(folder, files[0])
print("Video:", video)

shots = detect(video, ContentDetector())

if not shots:
    print("No cuts found: treat the whole video as one shot")
else:
    for i, (start, end) in enumerate(shots):
        print(f"Shot {i+1}: {start.get_timecode()} to {end.get_timecode()}")