"""ChakraNet Residual Diffusion Downscaler Model

Implements CorrDiff-style residual diffusion conditioned on U-Net mean and static topography.
Learns high-frequency turbulence residuals: x_res = y_true - mu_5km.
Supports both standard DDPM and accelerated DDIM reverse diffusion sampling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Convolutional block with time conditioning injection."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
        )
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        # Inject time embedding
        time_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_proj
        h = self.conv2(h)
        return h + self.res_conv(x)


class ResidualDiffusionUNet(nn.Module):
    """U-Net backbone predicting added noise in residual fields.
    
    Inputs:
        x_t: Noisy residual of shape (B, 1, H, W)
        t: Timestep tensor of shape (B,)
        condition: Conditioning tensor of shape (B, 3, H, W) [mu_5km, dem, land_mask]
    """

    def __init__(
        self,
        in_channels: int = 1,
        cond_channels: int = 3,
        out_channels: int = 1,
        base_dim: int = 32,
        time_dim: int = 128,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        total_in = in_channels + cond_channels  # 1 residual + 3 condition = 4 channels
        self.init_conv = nn.Conv2d(total_in, base_dim, kernel_size=3, padding=1)

        # Downscaling
        self.down1 = ResidualBlock(base_dim, base_dim * 2, time_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ResidualBlock(base_dim * 2, base_dim * 4, time_dim)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.mid1 = ResidualBlock(base_dim * 4, base_dim * 4, time_dim)
        self.mid2 = ResidualBlock(base_dim * 4, base_dim * 4, time_dim)

        # Upscaling
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv2 = ResidualBlock(base_dim * 4 + base_dim * 4, base_dim * 2, time_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv1 = ResidualBlock(base_dim * 2 + base_dim * 2, base_dim, time_dim)

        # Output projection predicting noise epsilon
        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, base_dim), base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)
        # Concatenate noisy residual with physical condition channels
        x = torch.cat([x_t, condition], dim=1)
        x0 = self.init_conv(x)

        d1 = self.down1(x0, t_emb)
        p1 = self.pool1(d1)

        d2 = self.down2(p1, t_emb)
        p2 = self.pool2(d2)

        m1 = self.mid1(p2, t_emb)
        m2 = self.mid2(m1, t_emb)

        u2 = self.up2(m2)
        u2 = torch.cat([u2, d2], dim=1)
        c2 = self.up_conv2(u2, t_emb)

        u1 = self.up1(c2)
        u1 = torch.cat([u1, d1], dim=1)
        c1 = self.up_conv1(u1, t_emb)

        return self.out_conv(c1)


class ResidualDiffusionPipeline:
    """Manages forward noise injection and reverse DDIM sampling."""

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precomputed diffusion parameters
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*noise."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def ddim_sample(
        self,
        condition: torch.Tensor,
        num_inference_steps: int = 25,
        eta: float = 0.0,
        generator_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Fast reverse diffusion using Denoising Diffusion Implicit Models (DDIM).
        
        Args:
            condition: Conditioning tensor [mu_5km, dem, land_mask] of shape (B, 3, H, W)
            num_inference_steps: Number of sampling steps (e.g. 25 instead of 1000)
            eta: DDIM stochasticity factor (0.0 for deterministic DDIM)
            
        Returns:
            x_res: Generated high-frequency residual field of shape (B, 1, H, W)
        """
        self.model.eval()
        b, _, h, w = condition.shape

        if generator_seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(generator_seed)
            x_t = torch.randn((b, 1, h, w), generator=gen, device=self.device)
        else:
            x_t = torch.randn((b, 1, h, w), device=self.device)

        # Uniform sub-sequence of timesteps
        step_ratio = self.timesteps // num_inference_steps
        timesteps_seq = torch.arange(0, self.timesteps, step_ratio, device=self.device).flip(0)

        for i, step in enumerate(timesteps_seq):
            t_batch = torch.full((b,), step, dtype=torch.long, device=self.device)
            # Predict noise
            pred_noise = self.model(x_t, t_batch, condition)

            alpha_bar = self.alphas_cumprod[step]
            prev_step = timesteps_seq[i + 1] if i + 1 < len(timesteps_seq) else -1
            alpha_bar_prev = self.alphas_cumprod[prev_step] if prev_step >= 0 else torch.tensor(1.0, device=self.device)

            # Predict x_0
            pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)

            # DDIM update step
            sigma = eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * (1.0 - alpha_bar / alpha_bar_prev))
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev - sigma ** 2) * pred_noise

            x_t = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
            if sigma > 0:
                x_t = x_t + sigma * torch.randn_like(x_t)

        return x_t
