from utils.eval_metrics import calculate_joint_metric


model.eval()
scores = []

with torch.no_grad():
    for lr, hr, _ in val_loader:
        sr = model(lr.to(device)).clamp(0,1)
        joint, psnr, ssim = calculate_joint_metric(sr[0], hr[0])
        scores.append(joint)

print("Avg Joint Metric:", sum(scores)/len(scores))
