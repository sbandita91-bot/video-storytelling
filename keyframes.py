import cv2, os
from tqdm import tqdm

VIDEO_DIR = "data/videos"
FRAME_DIR = "data/frames"

def extract_keyframes(video_path, out_dir, sim_threshold=0.98):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(int(fps), 1)          # about 1 frame per second
    prev_hist, saved, idx = None, 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            # skip the frame if it is almost identical to the last saved one
            if prev_hist is None or cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < sim_threshold:
                cv2.imwrite(os.path.join(out_dir, f"frame_{saved:03d}.jpg"), frame)
                prev_hist = hist
                saved += 1
        idx += 1
    cap.release()
    return saved

if __name__ == "__main__":
    files = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".avi", ".mp4"))]
    for f in tqdm(files):
        name = os.path.splitext(f)[0]
        n = extract_keyframes(os.path.join(VIDEO_DIR, f), os.path.join(FRAME_DIR, name))
        print(f"{name}: {n} keyframes")