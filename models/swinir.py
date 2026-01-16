# models/swinir.py
from timm.models.swin_transformer import SwinTransformer
import torch.nn as nn

class SwinIR(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.backbone = SwinTransformer(
            img_size=270,
            patch_size=1,
            in_chans=3,
            embed_dim=96,
            depths=(6,6,6,6),
            num_heads=(6,6,6,6),
            window_size=8
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(96, 96 * scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(96, 3, 3, 1, 1)
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        return self.upsample(x)

    def print(self):
        return "SwinIR"