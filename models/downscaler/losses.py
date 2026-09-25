"""ChakraNet Physics-Informed Downscaler Loss Functions

Implements composite objective functions for training the Residual Diffusion Downscaler:
    L_total = L_diff + lambda_1 * L_CRPS_tail + lambda_2 * L_PSD
Includes documented mathematical stubs and TODO hooks for thermodynamic mass
and Vertically Integrated Moisture Flux Convergence (VIMFC) conservation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class DiffusionMSELoss(nn.Module):
    """Mean Squared Error loss for predicted diffusion noise epsilon."""

    def __init__(self):
        super().__init__()

    def forward(self, pred_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
        """Args:
            pred_noise: Predicted noise from model (B, C, H, W).
            target_noise: Standard Gaussian ground-truth noise (B, C, H, W).
        """
        return F.mse_loss(pred_noise, target_noise)


class TailWeightedCRPSLoss(nn.Module):
    """Tail-weighted Continuous Ranked Probability Score (twCRPS).
    
    Penalizes forecast distribution error with higher weighting on severe extreme precipitation:
        w(y) = 1(y > threshold) or sigmoid weighting on extreme tail.
    For an ensemble of M realizations {x_1, ..., x_M} and scalar target y:
        CRPS(F, y) = 1/M sum_i |x_i - y| - 1/(2 M^2) sum_{i,j} |x_i - x_j|
    """

    def __init__(self, tail_threshold: float = 30.0, alpha: float = 0.05):
        super().__init__()
        self.tail_threshold = tail_threshold
        self.alpha = alpha

    def forward(self, ensemble: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
            ensemble: Stochastic realizations tensor of shape (M, B, H, W) or (B, M, H, W).
            target: Ground-truth high-resolution field of shape (B, 1, H, W).
            
        Returns:
            Scalar tail-weighted CRPS loss.
        """
        if ensemble.shape[0] != target.shape[0] and ensemble.shape[1] == target.shape[0]:
            # Permute to (B, M, H, W)
            ensemble = ensemble.permute(1, 0, 2, 3)

        b, m, h, w = ensemble.shape
        target_exp = target.expand(-1, m, -1, -1)

        # First term: 1/M sum_i |x_i - y|
        term1 = torch.mean(torch.abs(ensemble - target_exp), dim=1)  # (B, H, W)

        # Second term: 1/(2 M^2) sum_{i,j} |x_i - x_j|
        ens_i = ensemble.unsqueeze(2)  # (B, M, 1, H, W)
        ens_j = ensemble.unsqueeze(1)  # (B, 1, M, H, W)
        term2 = 0.5 * torch.mean(torch.abs(ens_i - ens_j), dim=(1, 2))  # (B, H, W)

        crps_pixel = term1 - term2

        # Tail weighting: emphasize cells where either ground truth or ensemble exceeds threshold
        weight = 1.0 + torch.relu(target.squeeze(1) - self.tail_threshold) / (self.tail_threshold + 1e-6)
        weighted_crps = torch.mean(crps_pixel * weight)

        return weighted_crps


class RadialPSDLoss(nn.Module):
    """Power Spectral Density (PSD) loss in 2D Fourier domain.
    
    Penalizes spectral blurring by enforcing high wavenumber energy retention
    matching the observed Kolmogorov -5/3 or turbulence power laws.
    """

    def __init__(self, lambda_high: float = 1.5):
        super().__init__()
        self.lambda_high = lambda_high

    def compute_radial_psd(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates radially-averaged 1D power spectral density from 2D spatial field.
        
        Args:
            field: Tensor of shape (B, 1, H, W).
            
        Returns:
            radial_bins: Center wavenumber of each bin.
            psd_1d: Mean spectral power per radial wavenumber bin.
        """
        b, c, h, w = field.shape
        # 2D FFT with orthonormal normalization
        fft_2d = torch.fft.fftshift(torch.fft.fft2(field, norm="ortho"))
        power_2d = torch.abs(fft_2d) ** 2  # (B, C, H, W)

        # Coordinate grid centered at DC frequency
        cy, cx = h // 2, w // 2
        y, x = torch.meshgrid(
            torch.arange(h, device=field.device) - cy,
            torch.arange(w, device=field.device) - cx,
            indexing="ij",
        )
        r = torch.sqrt(x**2 + y**2)
        r_int = r.long()

        max_r = min(cy, cx)
        radial_powers = []
        bins = torch.arange(max_r, device=field.device)

        # Radial binning
        for radius in range(max_r):
            mask = (r_int == radius).unsqueeze(0).unsqueeze(0)
            if mask.sum() > 0:
                p_bin = (power_2d * mask).sum() / mask.sum()
            else:
                p_bin = torch.tensor(0.0, device=field.device)
            radial_powers.append(p_bin)

        psd_1d = torch.stack(radial_powers)
        return bins, psd_1d

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes log-spectral MSE between predicted and target radially averaged PSDs."""
        _, psd_pred = self.compute_radial_psd(pred)
        _, psd_target = self.compute_radial_psd(target)

        # Compute log-spectral difference to balance low and high wavenumbers
        log_pred = torch.log(psd_pred + 1e-8)
        log_target = torch.log(psd_target + 1e-8)

        # High-frequency weighting: linearly scale weight with wavenumber
        wavenumber_weights = torch.linspace(1.0, self.lambda_high, len(psd_pred), device=pred.device)
        psd_loss = torch.mean(wavenumber_weights * (log_pred - log_target) ** 2)
        return psd_loss


class PhysicsStubs:
    """Documented mathematical stubs with explicit TODO hooks for atmospheric conservation laws."""

    @staticmethod
    def vimfc_conservation_loss(
        u_wind: torch.Tensor,
        v_wind: torch.Tensor,
        specific_humidity: torch.Tensor,
        precipitation: torch.Tensor,
    ) -> torch.Tensor:
        r"""Vertically Integrated Moisture Flux Convergence (VIMFC) Loss Stub.
        
        Mathematical Formulation:
            P - E = - \frac{1}{g} \int_{p_s}^{p_{top}} \nabla \cdot (q \mathbf{v}) \, dp
        
        In intense cyclonic convection, local precipitation rate P is balanced primarily by horizontal
        moisture convergence: P \approx - \nabla_h \cdot \mathbf{Q}, where Q = 1/g int q v dp.
        
        Args:
            u_wind: Zonal wind component (B, Levels, H, W) [m/s]
            v_wind: Meridional wind component (B, Levels, H, W) [m/s]
            specific_humidity: Specific humidity q (B, Levels, H, W) [kg/kg]
            precipitation: Surface rainfall rate P (B, 1, H, W) [mm/hr]
            
        Returns:
            Moisture flux imbalance penalty tensor.
        """
        # TODO [Phase 3 / Multi-level NCUM Integration]:
        # 1. Integrate specific humidity and horizontal wind vectors over pressure levels [1000hPa - 200hPa]
        #    using trapezoidal vertical quadrature: Q_x = 1/g sum q * u * dp, Q_y = 1/g sum q * v * dp.
        # 2. Compute spatial 2D divergence using central finite differences:
        #    div_Q = d(Q_x)/dx + d(Q_y)/dy
        # 3. Penalize discrepancy: || P - max(0, -div_Q) ||^2.
        
        # Prototype placeholder constraint (verifies tensor plumbing):
        target_scale = torch.mean(precipitation)
        synthetic_flux_balance = torch.tensor(0.0, device=precipitation.device, requires_grad=True)
        return synthetic_flux_balance

    @staticmethod
    def atmospheric_mass_conservation_loss(
        density: torch.Tensor,
        velocity_field: torch.Tensor,
    ) -> torch.Tensor:
        r"""Atmospheric Mass Continuity Loss Stub.
        
        Mathematical Formulation:
            \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0
            For quasi-incompressible anelastic mesoscale flows: \nabla \cdot (\rho_0 \mathbf{v}) = 0.
            
        Args:
            density: Atmospheric air density profile rho_0(z)
            velocity_field: 3D wind velocity vector field (u, v, w)
            
        Returns:
            Continuity divergence penalty.
        """
        # TODO [Phase 3 / 3D Dynamical Core Coupling]:
        # 1. Evaluate 3D spatial divergence of mass flux: div_rho_v = d(rho*u)/dx + d(rho*v)/dy + d(rho*w)/dz
        # 2. Penalize mean squared non-zero divergence across the domain.
        
        return torch.tensor(0.0, device=velocity_field.device, requires_grad=True)


class DownscalerCompositeLoss(nn.Module):
    """Full composite objective combining diffusion MSE, tail-CRPS, and spectral PSD loss."""

    def __init__(
        self,
        lambda_crps: float = 0.1,
        lambda_psd: float = 0.05,
        tail_threshold_mm: float = 30.0,
    ):
        super().__init__()
        self.lambda_crps = lambda_crps
        self.lambda_psd = lambda_psd
        self.diff_loss = DiffusionMSELoss()
        self.crps_loss = TailWeightedCRPSLoss(tail_threshold=tail_threshold_mm)
        self.psd_loss = RadialPSDLoss()

    def forward(
        self,
        pred_noise: torch.Tensor,
        target_noise: torch.Tensor,
        realizations: Optional[torch.Tensor] = None,
        target_hr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        l_diff = self.diff_loss(pred_noise, target_noise)
        total = l_diff

        losses = {"loss_diff": l_diff}

        if realizations is not None and target_hr is not None:
            l_crps = self.crps_loss(realizations, target_hr)
            l_psd = self.psd_loss(realizations.mean(dim=1 if realizations.shape[1] < realizations.shape[0] else 0, keepdim=True), target_hr)
            total = total + self.lambda_crps * l_crps + self.lambda_psd * l_psd
            losses["loss_crps"] = l_crps
            losses["loss_psd"] = l_psd

        losses["loss_total"] = total
        return losses
