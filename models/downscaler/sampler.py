"""ChakraNet Stochastic Downscaler Ensemble Sampler

Draws stochastic high-resolution realizations using the hybrid U-Net Mean + Residual Diffusion pipeline:
    y_hat_i = max(0, mu_5km + x_res_i)
Computes ensemble statistics (mean, spread, P90, P99) and extreme precipitation exceedance maps:
    P(Rainfall > tau) for tau in {50, 100, 150, 200} mm.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from models.downscaler.unet_mean import UNetMeanPredictor
from models.downscaler.diffusion_residual import ResidualDiffusionPipeline


class DownscalerEnsembleSampler:
    """Draws stochastic physical realizations and collapses to hazard exceedance probabilities."""

    def __init__(
        self,
        mean_model: Optional[UNetMeanPredictor] = None,
        diffusion_pipeline: Optional[ResidualDiffusionPipeline] = None,
        num_realizations: int = 16,
        inference_steps: int = 25,
        thresholds_mm: Optional[List[float]] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.mean_model = mean_model
        self.diffusion_pipeline = diffusion_pipeline
        self.num_realizations = num_realizations
        self.inference_steps = inference_steps
        self.thresholds_mm = thresholds_mm or [50.0, 100.0, 150.0, 200.0]

    def sample_realizations(
        self,
        coarse_input: torch.Tensor,
        dem: torch.Tensor,
        land_mask: torch.Tensor,
        num_samples: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Runs the 2-stage downscaler to produce N stochastic 5 km realizations.
        
        Args:
            coarse_input: Coarse precipitation field of shape (1, 1, H, W) or (H, W).
            dem: Static elevation field of shape (1, 1, H, W) or (H, W), normalized [0, 1].
            land_mask: Binary land-sea mask of shape (1, 1, H, W) or (H, W).
            num_samples: Number of realization members to generate (default: self.num_realizations).
            seed: Base RNG seed for reproducible ensemble generation.
            
        Returns:
            Dict containing:
                'mu_5km': Deterministic mean tensor of shape (H, W)
                'realizations': Array of shape (N, H, W) in mm/6hr
                'ensemble_mean': Array of shape (H, W) in mm/6hr
                'ensemble_spread': Array of shape (H, W) in mm/6hr
                'p90': Array of shape (H, W)
                'p99': Array of shape (H, W)
                'exceedance_probabilities': Dict[tau, Array of shape (H, W)] in [0, 1]
        """
        n_samples = num_samples or self.num_realizations

        # Ensure correct tensor shapes (1, 1, H, W)
        if isinstance(coarse_input, np.ndarray):
            coarse_input = torch.tensor(coarse_input, dtype=torch.float32)
        if isinstance(dem, np.ndarray):
            dem = torch.tensor(dem, dtype=torch.float32)
        if isinstance(land_mask, np.ndarray):
            land_mask = torch.tensor(land_mask, dtype=torch.float32)

        while coarse_input.ndim < 4:
            coarse_input = coarse_input.unsqueeze(0)
        while dem.ndim < 4:
            dem = dem.unsqueeze(0)
        while land_mask.ndim < 4:
            land_mask = land_mask.unsqueeze(0)

        coarse_input = coarse_input.to(self.device)
        dem = dem.to(self.device)
        land_mask = land_mask.to(self.device)

        # Stage 2A: Deterministic Mean Prediction mu_5km
        # Conditioning input: [coarse_field, dem, land_mask] -> shape (1, 3, H, W)
        cond_input = torch.cat([coarse_input, dem, land_mask], dim=1)

        if self.mean_model is not None:
            self.mean_model.eval()
            with torch.no_grad():
                mu_5km = self.mean_model(cond_input)
        else:
            # Fallback deterministic baseline: orographic amplification over elevation
            orographic_boost = 1.0 + 0.35 * dem * land_mask
            mu_5km = coarse_input * orographic_boost

        # Stage 2B: Stochastic Residual Sampling via DDIM
        # Conditioning for diffusion: [mu_5km, dem, land_mask] -> shape (1, 3, H, W)
        diff_cond = torch.cat([mu_5km, dem, land_mask], dim=1)

        realizations = []
        base_seed = seed if seed is not None else 1000

        for i in range(n_samples):
            sample_seed = base_seed + i * 37
            if self.diffusion_pipeline is not None:
                x_res = self.diffusion_pipeline.ddim_sample(
                    condition=diff_cond,
                    num_inference_steps=self.inference_steps,
                    eta=0.0,
                    generator_seed=sample_seed,
                )
            else:
                # Synthetic turbulent residual generator matching Kolmogorov turbulence spectrum
                rng = np.random.RandomState(sample_seed)
                h, w = coarse_input.shape[-2], coarse_input.shape[-1]
                noise = rng.randn(h, w).astype(np.float32)
                # Apply high-pass spectral filter to simulate fine-scale convective turbulent cells
                fft = np.fft.fft2(noise)
                kx = np.fft.fftfreq(w)
                ky = np.fft.fftfreq(h)
                kxx, kyy = np.meshgrid(kx, ky)
                k = np.sqrt(kxx**2 + kyy**2)
                # Bandpass filter around convective scales
                filt = (k > 0.05).astype(np.float32) * np.exp(-((k - 0.2) ** 2) / 0.04)
                res_filtered = np.real(np.fft.ifft2(fft * filt))
                # Scale residual relative to local mean intensity (higher spread in eyewall/rainbands)
                res_scaled = res_filtered * (mu_5km.squeeze().cpu().numpy() * 0.28 + 2.0)
                x_res = torch.tensor(res_scaled, device=self.device).unsqueeze(0).unsqueeze(0)

            # y_hat_i = clamp(mu_5km + x_res_i, min=0.0) -> physical non-negativity constraint
            y_hat = torch.clamp(mu_5km + x_res, min=0.0)
            realizations.append(y_hat.squeeze().cpu().numpy())

        realizations_np = np.stack(realizations, axis=0)  # Shape: (N, H, W)
        mu_np = mu_5km.squeeze().cpu().numpy()

        # Compute Ensemble Summary Statistics
        ens_mean = np.mean(realizations_np, axis=0)
        ens_spread = np.std(realizations_np, axis=0)
        p90 = np.percentile(realizations_np, 90, axis=0)
        p99 = np.percentile(realizations_np, 99, axis=0)

        # Extreme Precipitation Exceedance Probability Maps: P(R > tau)
        exceedance_probs = {}
        for tau in self.thresholds_mm:
            prob_map = np.mean((realizations_np >= tau).astype(np.float32), axis=0)
            exceedance_probs[tau] = prob_map

        return {
            "mu_5km": mu_np,
            "realizations": realizations_np,
            "ensemble_mean": ens_mean,
            "ensemble_spread": ens_spread,
            "p90": p90,
            "p99": p99,
            "exceedance_probabilities": exceedance_probs,
        }
