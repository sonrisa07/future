import torch
import torch.nn as nn
import torch.nn.functional as F


class DCN(nn.Module):
    def __init__(self, channels, kernel, d):
        super(DCN, self).__init__()
        self.lef_padding = (kernel - 1) * d
        self.dw = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel),
            padding=(0, 0),
            dilation=(1, d),
            groups=channels,
        )
        self.fw = nn.Conv2d(
            in_channels=channels, out_channels=2 * channels, kernel_size=1
        )

    def forward(self, x):  # (B, K, N, C)
        x = x.permute(0, 3, 2, 1)  # (B, C, N, K)
        x = F.pad(x, (self.lef_padding, 0, 0, 0))
        x = self.dw(x)  # (B, C, N, K)
        x = self.fw(x)  # (B, 2C, N, K)
        z_f, z_g = torch.chunk(x, 2, dim=1)
        h = F.gelu(z_f) * F.sigmoid(z_g)
        return h.permute(0, 3, 2, 1)
