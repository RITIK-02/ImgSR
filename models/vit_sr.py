# # models/vit_sr.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ViTSR(nn.Module):
#     def __init__(self, scale=4, embed_dim=256, num_heads=8, num_layers=6):
#         super().__init__()

#         self.embed_dim = embed_dim
#         self.scale = scale
#         # self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
#         self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=scale, stride=scale)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=num_heads, batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

#         self.upsampler = nn.Sequential(
#             nn.Conv2d(embed_dim, embed_dim * scale * scale, 3, 1, 1),
#             nn.PixelShuffle(scale),
#             nn.Conv2d(embed_dim, 3, 3, 1, 1)
#         )

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.patch_embed(x)               # B, D, H/4, W/4
#         h, w = x.shape[-2:]

#         x = x.flatten(2).transpose(1, 2)      # B, N, D
#         x = self.transformer(x)
#         x = x.transpose(1, 2).reshape(B, -1, h, w)

#         return self.upsampler(x)

#     def print(self):
#         return "VITSR"

import torch
import torch.nn as nn
import torch.nn.functional as F

class ViTSR(nn.Module):
    def __init__(self, scale=4, embed_dim=256, num_heads=8, num_layers=6):
        super().__init__()

        self.scale = scale
        self.embed_dim = embed_dim

        # print(">>> ViTSR INIT scale =", scale)

        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=1, stride=1
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers
        )

        self.upsampler = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim * scale * scale,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.PixelShuffle(scale),
            nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        scale = self.scale

        pad_h = (scale - H % scale) % scale
        pad_w = (scale - W % scale) % scale

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        Hp, Wp = x.shape[-2:]

        x = self.patch_embed(x)
        h, w = x.shape[-2:]

        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, h, w)

        # print(">>> Before upsample:", x.shape)

        x = self.upsampler(x)

        # print(">>> After upsample:", x.shape)

        return x[:, :, :H*scale, :W*scale]

    def print(self):
        return "VITSR"
