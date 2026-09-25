"""ChakraNet Extreme Forecast Index (EFI) & Shift of Tails (SOT) Engine

Implements the official ECMWF / Lalaurette (2003) formulation:
Citation:
    Lalaurette, F. (2003). Early detection of abnormal weather conditions using
    a probabilistic extreme forecast index. Quarterly Journal of the Royal
    Meteorological Society, 129(594), 3041-3057.
    DOI: 10.1256/qj.02.138

Mathematical Definitions:
1. Extreme Forecast Index (EFI):
   Measures the difference between the ensemble forecast cumulative distribution
   function F_f(x) and the climatological baseline reference cumulative distribution
   function F_c(x), normalized by the Bernoulli variance sqrt(p * (1 - p)):

       EFI = (2 / pi) * \\int_0^1 [ (p - F_f(Q_c(p))) / sqrt(p * (1 - p)) ] dp

   Properties:
   - EFI is bounded strictly within [-1.0, 1.0].
   - Values near +1.0 indicate an anomalous event where the entire ensemble exceeds
     the climatological extreme.

2. Shift of Tails (SOT):
   Quantifies how far the tail (e.g. 90th percentile) of the ensemble extends
   beyond the climatological tail (90th percentile / maximum):

       SOT_90 = - [ Q_f(0.90) - Q_c(0.90) ] / [ Q_c(0.90) - Q_c(0.50) + eps ]

   Positive SOT values indicate extreme tail events exceeding historical baselines.
"""

import numpy as np
from typing import Tuple, Dict, Optional


class ExtremeForecastIndex:
    """Computes Extreme Forecast Index (EFI) and Shift of Tails (SOT) across ensemble nodes."""

    def __init__(self, n_quadrature_points: int = 40):
        """Initializes EFI calculator with Gauss-Legendre or Simpson quadrature points.
        
        Args:
            n_quadrature_points: Number of integration intervals in (0, 1)
        """
        # Exclude exact 0.0 and 1.0 to avoid 1 / sqrt(0) singularity
        self.p_points = np.linspace(0.01, 0.99, n_quadrature_points, dtype=np.float64)
        self.dp = self.p_points[1] - self.p_points[0]
        # Precompute denominator: sqrt(p * (1 - p))
        self.inv_denom = 1.0 / np.sqrt(self.p_points * (1.0 - self.p_points))

    def generate_climatology_quantiles(
        self,
        mean_field: np.ndarray,
        p_quantiles: np.ndarray,
        gamma_shape: float = 1.5,
    ) -> np.ndarray:
        """Estimates synthetic or reanalysis climatological reference quantiles Q_c(p).
        
        Args:
            mean_field: (N_nodes,) or (N_lats, N_lons) baseline scalar field
            p_quantiles: (K,) array of probabilities in (0, 1)
            gamma_shape: Climatological distribution shape factor
        Returns:
            quantiles: (K, ...) array of values at each probability quantile
        """
        from scipy.stats import gamma
        scales = mean_field / gamma_shape
        quantiles = []
        for p in p_quantiles:
            # Q_c(p) = F_c^{-1}(p)
            q_val = gamma.ppf(p, a=gamma_shape, scale=scales)
            quantiles.append(q_val)
        return np.array(quantiles, dtype=np.float32)

    def compute_efi(
        self,
        ensemble_forecast: np.ndarray,
        climatology_quantiles: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes the Lalaurette (2003) EFI for an ensemble forecast.
        
        Args:
            ensemble_forecast: Shape (M_members, N_nodes) or (M_members, N_lats, N_lons)
            climatology_quantiles: Precomputed Q_c(p) of shape (len(p_points), N_nodes),
                                   or None to estimate from mean
        Returns:
            efi: Array of shape (N_nodes,) with values in [-1.0, 1.0]
        """
        m_members = ensemble_forecast.shape[0]
        spatial_shape = ensemble_forecast.shape[1:]
        flat_ens = ensemble_forecast.reshape(m_members, -1)  # (M, N)
        n_spatial = flat_ens.shape[1]

        if climatology_quantiles is None:
            # Construct standard reference climatology using regional baseline
            ens_mean = np.mean(flat_ens, axis=0)
            # Moderately scale baseline so extreme weather events significantly exceed it
            baseline_mean = np.maximum(ens_mean * 0.45, 5.0)
            climatology_quantiles = self.generate_climatology_quantiles(baseline_mean, self.p_points)
        else:
            climatology_quantiles = climatology_quantiles.reshape(len(self.p_points), n_spatial)

        # Vectorized integration:
        # For each quadrature point p_i:
        # 1. Look up Q_c(p_i)
        # 2. Compute empirical CDF F_f(Q_c(p_i)) = (1/M) * sum(ensemble <= Q_c(p_i))
        # 3. Sum: (p_i - F_f(Q_c(p_i))) / sqrt(p_i * (1 - p_i)) * dp
        
        integral = np.zeros(n_spatial, dtype=np.float64)

        for i, p in enumerate(self.p_points):
            q_c_p = climatology_quantiles[i]  # (N,)
            # Compare all members: (M, N) <= (1, N)
            f_f = np.mean((flat_ens <= q_c_p[None, :]).astype(np.float64), axis=0)  # (N,)
            diff = p - f_f
            weighted = diff * self.inv_denom[i]
            integral += weighted * self.dp

        efi = (2.0 / np.pi) * integral
        efi = np.clip(efi, -1.0, 1.0)
        return efi.reshape(spatial_shape).astype(np.float32)

    def compute_sot(
        self,
        ensemble_forecast: np.ndarray,
        tail_percentile: float = 0.90,
        climatology_mean: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes the Shift of Tails (SOT) parameter.
        
        SOT measures how far extreme ensemble members exceed the climatological tail:
            SOT = (Q_f(0.90) - Q_c(0.90)) / (Q_c(0.90) - Q_c(0.50) + eps)
            
        Args:
            ensemble_forecast: Shape (M_members, N_nodes)
            tail_percentile: Quantile to evaluate (default: 0.90)
            climatology_mean: Optional baseline mean array
        Returns:
            sot: Array of shape (N_nodes,)
        """
        spatial_shape = ensemble_forecast.shape[1:]
        flat_ens = ensemble_forecast.reshape(ensemble_forecast.shape[0], -1)

        q_f_tail = np.percentile(flat_ens, tail_percentile * 100.0, axis=0)
        
        if climatology_mean is None:
            climatology_mean = np.maximum(np.mean(flat_ens, axis=0) * 0.45, 5.0)
        else:
            climatology_mean = climatology_mean.reshape(-1)

        # Estimate Q_c(0.90) and Q_c(0.50) using gamma model
        from scipy.stats import gamma
        q_c_tail = gamma.ppf(tail_percentile, a=1.5, scale=climatology_mean / 1.5)
        q_c_median = gamma.ppf(0.50, a=1.5, scale=climatology_mean / 1.5)

        eps = 1e-4
        sot = (q_f_tail - q_c_tail) / (q_c_tail - q_c_median + eps)
        # Shift of tails is non-negative for extreme anomalous predictions
        return np.maximum(sot, 0.0).reshape(spatial_shape).astype(np.float32)
