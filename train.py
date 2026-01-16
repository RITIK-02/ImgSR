#train.py 
import torch 
from torch.utils.data import DataLoader, Subset 
from datasets.sr_dataset import SuperResolutionDataset 
from losses.sr_loss import SRLoss 
import random 
random.seed(42) 
from math import inf 

from utils.eval_metrics import calculate_joint_metric 

from models.edsr import EDSR 
from models.swinir import SwinIR 
from models.vit_sr import ViTSR 

device = "cuda" if torch.cuda.is_available() else "cpu" 

# ----------------------- 
# Dataset & loaders 
# ----------------------- 
train_dataset = SuperResolutionDataset("data/train/lr", "data/train/hr") 

train_loader = DataLoader( 
    train_dataset, 
    batch_size=1, 
    shuffle=True, 
    num_workers=4 
)

val_indices = random.sample(range(len(train_dataset)), 5) 
val_subset = Subset(train_dataset, val_indices) 

val_loader = DataLoader( 
    val_subset, 
    batch_size=1, 
    shuffle=False 
) 

# ----------------------- 
# Model, loss, optimizer 
# ----------------------- 
# model = EDSR(scale=4).to(device) # swap model here
# model = SwinIR(scale=4).to(device) #NOTE: NOT WORKING
model = ViTSR(scale=4).to(device)
loss_fn = SRLoss(alpha=0.8) 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 

EPOCHS = 50 

best_joint = -inf 

# ----------------------- 
# Training loop 
# ----------------------- 
for epoch in range(EPOCHS): 
    model.train() 
    total_loss = 0 
    for lr, hr, _ in train_loader: 
        lr, hr = lr.to(device), hr.to(device) 
        
        sr = model(lr) 
        # print("LR:", lr.shape)
        # print("HR:", hr.shape)
        # print("SR:", sr.shape)
        loss = loss_fn(sr, hr) 
        
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        
        total_loss += loss.item() 
        
    # -----------------------
    # Validation
    # -----------------------
    model.eval()
    joint_scores, psnr_scores, ssim_scores = [], [], []

    with torch.no_grad():
        for lr, hr, _ in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr).clamp(0, 1)

            joint, psnr, ssim = calculate_joint_metric(sr[0], hr[0])
            joint_scores.append(joint)
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

    avg_joint = sum(joint_scores) / len(joint_scores)
    avg_psnr  = sum(psnr_scores) / len(psnr_scores)
    avg_ssim  = sum(ssim_scores) / len(ssim_scores)
    print(
        f"Epoch {epoch} | "
        f"Loss: {total_loss/len(train_loader):.4f} | "
        f"Joint: {avg_joint:.2f} | "
        f"PSNR: {avg_psnr:.2f} | "
        f"SSIM: {avg_ssim:.4f}"
    )
    if avg_joint > best_joint:
        best_joint = avg_joint
        torch.save(
            model.state_dict(),
            f"checkpoints/{model.print()}_best_{epoch}.pth"
        )
