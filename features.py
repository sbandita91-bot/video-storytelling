import os
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

FRAME_DIR = "data/frames"
FEAT_DIR = "data/features"
os.makedirs(FEAT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

model = models.resnet50(weights="DEFAULT")
model.fc = torch.nn.Identity()   # remove the classifier, keep 2048 features
model.eval().to(device)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

videos = sorted(os.listdir(FRAME_DIR))
for name in tqdm(videos):
    folder = os.path.join(FRAME_DIR, name)
    imgs = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
    if not imgs:
        continue
    batch = torch.stack([
        transform(Image.open(os.path.join(folder, f)).convert("RGB"))
        for f in imgs
    ]).to(device)
    with torch.no_grad():
        feats = model(batch).cpu().numpy()   # shape: (num_frames, 2048)
    np.save(os.path.join(FEAT_DIR, name + ".npy"), feats)

print("Done.")