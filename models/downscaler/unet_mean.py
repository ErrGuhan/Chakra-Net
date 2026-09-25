"""ChakraNet U-Net Base Regression Mean Model

Predicts the smoothed deterministic ensemble mean at 5 km resolution from
coarse 12 km crop fields and static physical topography conditioning (DEM + land-sea mask).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two convolutional layers with GroupNorm and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        num_groups = min(8, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Downscaling block: MaxPool followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpconv(x)


class UpBlock(nn.Module):
    """Upscaling block: Bilinear interpolation followed by DoubleConv with skip connections."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2 spatial dimensions
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetMeanPredictor(nn.Module):
    """2D U-Net mapping [coarse_field, dem, land_mask] -> deterministic mean field mu_5km."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_dim: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_dim)
        self.down1 = DownBlock(base_dim, base_dim * 2)
        self.down2 = DownBlock(base_dim * 2, base_dim * 4)
        self.down3 = DownBlock(base_dim * 4, base_dim * 8)

        # Bottleneck
        self.bot = DoubleConv(base_dim * 8, base_dim * 8)

        # Decoder with skip connections
        self.up1 = UpBlock(base_dim * 8 + base_dim * 8, base_dim * 4)
        self.up2 = UpBlock(base_dim * 4 + base_dim * 4, base_dim * 2)
        self.up3 = UpBlock(base_dim * 2 + base_dim * 2, base_dim)

        # Output projection (Softplus ensures physically non-negative precipitation)
        self.outc = nn.Conv2d(base_dim, out_channels, kernel_size=1)
        self.activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
               - Channel 0: Upsampled coarse precipitation
               - Channel 1: Normalized DEM elevation
               - Channel 2: Land-sea binary mask
               
        Returns:
            mu_5km: Deterministic mean prediction of shape (B, 1, H, W) >= 0.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        b = self.bot(x4)

        u1 = self.up1(b, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)

        # Upsample to original spatial dimension matching x1
        u_final = F.interpolate(u3, size=x1.shape[2:], mode="bilinear", align_corners=False)
        out = self.outc(u_final)
        # Combine residual with base coarse input channel to ensure conservation prior
        base_coarse = x[:, 0:1, :, :]
        return self.activation(base_coarse + out)
