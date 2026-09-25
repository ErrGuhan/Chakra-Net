"""ChakraNet Downscaler Crop & Physical Topography Conditioning

Extracts dynamic 12 km ensemble forecast crops from Stage 1 bounding boxes
and provides normalized 5 km static elevation (DEM) and land-sea mask conditioning.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from config import EVENT_CONFIG, MODEL_CONFIG


class CropConditioner:
    """Prepares spatio-temporal crops and high-resolution physical conditioning."""

    def __init__(self, target_size: int = 128):
        self.target_size = target_size

    def extract_crop(
        self,
        full_field: np.ndarray,
        crop_bounds: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Extracts a regional crop from the full Bay of Bengal field.
        
        Args:
            full_field: 2D array of shape (n_lat, n_lon) or 3D/4D array.
            crop_bounds: Optional dict with keys 'lat_min', 'lat_max', 'lon_min', 'lon_max'.
        
        Returns:
            cropped array interpolated or sliced to match crop dimensions.
        """
        if crop_bounds is None:
            # Default Phailin landfall bounding box: [15N-21N, 82E-88E]
            crop_bounds = {
                "lat_min": 15.0,
                "lat_max": 21.0,
                "lon_min": 82.0,
                "lon_max": 88.0,
            }

        lat_min = EVENT_CONFIG.bbox_lat_min
        lat_max = EVENT_CONFIG.bbox_lat_max
        lon_min = EVENT_CONFIG.bbox_lon_min
        lon_max = EVENT_CONFIG.bbox_lon_max

        n_lat, n_lon = full_field.shape[-2], full_field.shape[-1]
        lats = np.linspace(lat_min, lat_max, n_lat)
        lons = np.linspace(lon_min, lon_max, n_lon)

        # Find slice indices
        i0 = np.searchsorted(lats, crop_bounds["lat_min"])
        i1 = np.searchsorted(lats, crop_bounds["lat_max"])
        j0 = np.searchsorted(lons, crop_bounds["lon_min"])
        j1 = np.searchsorted(lons, crop_bounds["lon_max"])

        i0, i1 = max(0, i0), min(n_lat, max(i0 + 10, i1))
        j0, j1 = max(0, j0), min(n_lon, max(j0 + 10, j1))

        if full_field.ndim == 2:
            cropped = full_field[i0:i1, j0:j1]
        elif full_field.ndim == 3:
            cropped = full_field[:, i0:i1, j0:j1]
        elif full_field.ndim == 4:
            cropped = full_field[:, :, i0:i1, j0:j1]
        else:
            raise ValueError(f"Unsupported array dimension: {full_field.ndim}")

        return cropped

    def generate_static_topography(
        self,
        size: int = 128,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synthesizes realistic 5 km static elevation (DEM) and land-sea mask for Odisha/AP coast.
        
        The Eastern Ghats run southwest-to-northeast with elevations peaking at 900-1500m inland,
        descending to coastal alluvial plains and sea level (0m) in the Bay of Bengal.
        
        Returns:
            dem: 2D array of shape (size, size) normalized to [0, 1].
            land_mask: 2D binary array of shape (size, size) where 1=land, 0=sea.
        """
        rng = np.random.RandomState(seed)
        y = np.linspace(0, 1, size)
        x = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, y)

        # Coastline profile: diagonal from bottom-left to top-center (Bay of Bengal on right)
        coast_line = 0.35 + 0.3 * yy + 0.05 * np.sin(yy * 3 * np.pi)
        land_mask = (xx < coast_line).astype(np.float32)

        # Elevation model: inland ridge (Eastern Ghats) peaking parallel to coast
        dist_from_coast = np.maximum(0.0, coast_line - xx)
        base_elevation = 1200.0 * np.sin(np.pi * np.clip(dist_from_coast / 0.35, 0, 1))

        # Add realistic fractal topographic roughness
        roughness = (
            150.0 * np.sin(8 * np.pi * xx) * np.cos(8 * np.pi * yy)
            + 80.0 * np.sin(16 * np.pi * xx) * np.cos(16 * np.pi * yy)
            + rng.normal(0, 15.0, size=(size, size))
        )
        dem = np.maximum(0.0, (base_elevation + roughness) * land_mask)
        # Normalize DEM to [0, 1] relative to peak elevation (~1500m)
        dem_norm = (dem / 1500.0).astype(np.float32)

        return dem_norm, land_mask

    def prepare_conditioning_tensor(
        self,
        coarse_crop: np.ndarray,
        dem: np.ndarray,
        land_mask: np.ndarray,
    ) -> torch.Tensor:
        """Interpolates coarse crop and stacks with static physical topography conditioning.
        
        Args:
            coarse_crop: 2D array of shape (H_c, W_c) or torch.Tensor.
            dem: 2D array of shape (H_f, W_f) representing 5 km elevation.
            land_mask: 2D array of shape (H_f, W_f) representing 5 km land-sea mask.
            
        Returns:
            conditioning: torch.Tensor of shape (1, 3, target_size, target_size)
                          Channel 0: Coarse precipitation upsampled to 5 km grid
                          Channel 1: Normalized DEM elevation
                          Channel 2: Binary land-sea mask
        """
        if isinstance(coarse_crop, np.ndarray):
            coarse_t = torch.from_numpy(coarse_crop).float()
        else:
            coarse_t = coarse_crop.float()

        if coarse_t.ndim == 2:
            coarse_t = coarse_t.unsqueeze(0).unsqueeze(0)
        elif coarse_t.ndim == 3:
            coarse_t = coarse_t.unsqueeze(1)

        # Bilinear upsample coarse 12 km grid to target 5 km size
        coarse_upsampled = F.interpolate(
            coarse_t,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )

        dem_t = torch.from_numpy(dem).float().unsqueeze(0).unsqueeze(0)
        mask_t = torch.from_numpy(land_mask).float().unsqueeze(0).unsqueeze(0)

        # Concatenate into 3-channel conditioning tensor
        cond = torch.cat([coarse_upsampled, dem_t, mask_t], dim=1)
        return cond
