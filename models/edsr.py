# models/edsr.py
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class EDSR(nn.Module):
    def __init__(self, scale=4, num_blocks=16, channels=64):
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, 3, 3, 1, 1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        return self.tail(res + x)
    
    def print(self):
        return "EDSR"
