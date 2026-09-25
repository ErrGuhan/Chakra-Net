"""Unit tests for ChakraNet Downscaler Crop & Conditioning Data Module"""

import pytest
import numpy as np
import torch
from models.downscaler.crop_condition import CropConditioner


def test_crop_conditioner_extraction():
    conditioner = CropConditioner(target_size=128)
    full_field = np.ones((80, 80), dtype=np.float32) * 25.0
    cropped = conditioner.extract_crop(full_field)
    assert cropped.ndim == 2
    assert cropped.shape[0] > 0
    assert cropped.shape[1] > 0


def test_static_topography_generation():
    conditioner = CropConditioner(target_size=128)
    dem, land_mask = conditioner.generate_static_topography(size=64, seed=42)

    assert dem.shape == (64, 64)
    assert land_mask.shape == (64, 64)

    # Elevation normalized between 0 and 1
    assert dem.min() >= 0.0
    assert dem.max() <= 1.0

    # Land mask is binary
    assert np.all(np.isin(land_mask, [0.0, 1.0]))

    # Coastline separation: sea should have 0 elevation
    sea_pixels = (land_mask == 0.0)
    assert np.allclose(dem[sea_pixels], 0.0)


def test_prepare_conditioning_tensor():
    conditioner = CropConditioner(target_size=64)
    dem, land_mask = conditioner.generate_static_topography(size=64, seed=42)
    coarse_precip = np.random.uniform(0, 100, (32, 32)).astype(np.float32)
    cond = conditioner.prepare_conditioning_tensor(coarse_precip, dem, land_mask)

    assert isinstance(cond, torch.Tensor)
    assert cond.shape == (1, 3, 64, 64)
    # Channel 0: precip, Channel 1: dem, Channel 2: land_mask
    assert cond[0, 1].min() >= 0.0
    assert cond[0, 1].max() <= 1.0
