import os, shutil
import numpy as np
import cv2
from sklearn.cluster import KMeans
from tqdm import tqdm

FRAME_DIR = "data/frames"
FEAT_DIR = "data/features"
OUT_DIR = "outputs"
K = 5   # number of frames in each summary

def summarize(name, k=K):
    feats = np.load(os.path.join(FEAT_DIR, name + ".npy"))
    frames = sorted(f for f in os.listdir(os.path.join(FRAME_DIR, name)) if f.endswith(".jpg"))
    n = len(frames)
    k = min(k, n)

    # cluster the frames; pick the frame closest to each cluster center
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(feats)
    chosen = []
    for c in range(k):
        idxs = np.where(km.labels_ == c)[0]
        dists = np.linalg.norm(feats[idxs] - km.cluster_centers_[c], axis=1)
        chosen.append(idxs[np.argmin(dists)])
    chosen = sorted(chosen)   # keep the original time order

    # save the selected frames and one strip image
    out = os.path.join(OUT_DIR, name)
    os.makedirs(out, exist_ok=True)
    imgs = []
    for i in chosen:
        src = os.path.join(FRAME_DIR, name, frames[i])
        shutil.copy(src, os.path.join(out, frames[i]))
        imgs.append(cv2.resize(cv2.imread(src), (224, 160)))
    cv2.imwrite(os.path.join(out, "summary_strip.jpg"), np.hstack(imgs))
    return chosen

if __name__ == "__main__":
    for name in tqdm(sorted(f[:-4] for f in os.listdir(FEAT_DIR) if f.endswith(".npy"))):
        picks = summarize(name)
        print(name, "->", picks)