import torch.nn.functional as F
from pytorch_msssim import ssim

class SRLoss:
    def __init__(self, alpha=0.8):
        """
        alpha: weight for L1, (1-alpha) for SSIM
        """
        self.alpha = alpha

    def __call__(self, sr, hr):
        l1 = F.l1_loss(sr, hr)
        ssim_loss = 1 - ssim(sr, hr, data_range=1.0, size_average=True)
        return self.alpha * l1 + (1 - self.alpha) * ssim_loss
