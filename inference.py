import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.sr_dataset import SuperResolutionDataset
from models.edsr import EDSR   # or SwinIR / ViTSR

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Load model
# -----------------------
model = EDSR(scale=4)
model.load_state_dict(torch.load("checkpoints/EDSR_best_47.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------
# Dataset & loader
# -----------------------
test_dataset = SuperResolutionDataset(
    lr_dir="data/test/lr",
    hr_dir=None
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

# -----------------------
# Inference
# -----------------------
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for lr, fname in tqdm(test_loader):
        lr = lr.to(device)
        sr = model(lr).clamp(0,1)

        sr_img = sr[0].permute(1,2,0).cpu().numpy()
        sr_img = (sr_img * 255).astype(np.uint8)

        cv2.imwrite(
            os.path.join(output_dir, fname[0]),
            cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
        )
