# Residual double-conv blocks with InstanceNorm + LeakyReLU
# Returns raw logits for CrossEntropyLoss

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

# build                
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.do = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.do(x)
        return x

class ResidualBlock(nn.Module):
    """Two 3x3 convs + residual. Projects skip if channels differ."""
    def __init__(self, in_ch, out_ch, dropout=0.0, use_dilation=False):
        super().__init__()
        dilation = 2 if use_dilation else 1
        self.conv1 = ConvBNAct(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, dropout=dropout)
        self.conv2 = ConvBNAct(out_ch, out_ch, kernel_size=3, padding=1, dilation=1, dropout=dropout)
        self.proj = nn.Identity()
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
            )

    def forward(self, x):
        skip = self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + skip

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, dropout=dropout, use_dilation=False)
        self.att = nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.res(x)
        x = self.att(x)
        skip = x
        x = self.pool(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.res = ResidualBlock(out_ch + skip_ch, out_ch, dropout=dropout, use_dilation=False)
        self.att = nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        # odd shapes
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        x = self.att(x)
        return x

# model
class ImprovedUNet(nn.Module):
    """
    U-Net with residual blocks.
    """
    def __init__(self, in_channels=1, num_classes=4, base_ch=32, depth=4, dropout=0.1):
        super().__init__()
        assert depth >= 3, "depth >= 3 recommended"

        chs: List[int] = [base_ch * (2 ** i) for i in range(depth)]
        self.stem = ResidualBlock(in_channels, chs[0], dropout=dropout, use_dilation=False)

        # Encoder
        self.downs = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(DownBlock(chs[i], chs[i + 1], dropout=dropout))

        # Bottleneck
        self.bottleneck = ResidualBlock(chs[-1], chs[-1], dropout=dropout, use_dilation=True)

        # Decoder
        self.ups = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            in_ch  = chs[i + 1]     # input to upconv (from previous stage)
            skip_ch = chs[i + 1]    # skip tensor channels saved from encoder
            out_ch = chs[i]         # halve channels on the way up
            
            self.ups.append(UpBlock(in_ch, skip_ch, out_ch, dropout=dropout))

        # Head
        self.head = nn.Conv2d(chs[0], num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B,1,H,W)
        x0 = self.stem(x)
        skips: List[torch.Tensor] = []
        x = x0
        for down in self.downs:
            x, s = down(x)
            skips.append(s)
        x = self.bottleneck(x)
        for up in self.ups:
            s = skips.pop()
            x = up(x, s)
        logits = self.head(x)  # shape (B, num_classes, H, W)
        return logits

# test
if __name__ == "__main__":
    net = ImprovedUNet(in_channels=1, num_classes=4, base_ch=32, depth=4, dropout=0.1)
    x = torch.randn(2, 1, 128, 128)
    y = net(x)
    print("logits:", tuple(y.shape))
