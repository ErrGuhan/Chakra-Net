"""Unit tests for ChakraNet Stage 2 Residual Diffusion Model and Ensemble Sampler"""

import pytest
import torch
import numpy as np
from models.downscaler.diffusion_residual import (
    ResidualDiffusionUNet,
    ResidualDiffusionPipeline,
    SinusoidalTimeEmbedding,
)
from models.downscaler.sampler import DownscalerEnsembleSampler


def test_sinusoidal_time_embedding():
    dim = 64
    embedder = SinusoidalTimeEmbedding(dim=dim)
    t = torch.tensor([0, 10, 50, 100], dtype=torch.long)
    emb = embedder(t)

    assert emb.shape == (4, dim)
    assert not torch.isnan(emb).any()


def test_residual_diffusion_unet_forward():
    model = ResidualDiffusionUNet(in_channels=1, cond_channels=3, base_dim=16, time_dim=32)
    b, h, w = 2, 32, 32
    x_t = torch.randn(b, 1, h, w)
    t = torch.randint(0, 100, (b,))
    condition = torch.randn(b, 3, h, w)

    pred_noise = model(x_t, t, condition)
    assert pred_noise.shape == (b, 1, h, w)


def test_forward_diffusion_q_sample():
    model = ResidualDiffusionUNet(base_dim=16, time_dim=32)
    pipeline = ResidualDiffusionPipeline(model, timesteps=100)
    x_0 = torch.randn(2, 1, 32, 32)
    t = torch.tensor([10, 90])

    x_t = pipeline.q_sample(x_0, t)
    assert x_t.shape == (2, 1, 32, 32)


def test_fast_ddim_sample():
    model = ResidualDiffusionUNet(base_dim=16, time_dim=32)
    pipeline = ResidualDiffusionPipeline(model, timesteps=50)
    cond = torch.randn(1, 3, 32, 32)

    # Fast 5-step DDIM sampling
    x_res = pipeline.ddim_sample(condition=cond, num_inference_steps=5, generator_seed=42)
    assert x_res.shape == (1, 1, 32, 32)
    assert not torch.isnan(x_res).any()


def test_downscaler_ensemble_sampler_exceedance_maps():
    sampler = DownscalerEnsembleSampler(
        num_realizations=4,
        thresholds_mm=[50.0, 100.0, 150.0],
    )
    h, w = 32, 32
    coarse = np.ones((h, w), dtype=np.float32) * 60.0
    dem = np.zeros((h, w), dtype=np.float32)
    mask = np.ones((h, w), dtype=np.float32)

    results = sampler.sample_realizations(coarse, dem, mask, num_samples=4)

    assert "realizations" in results
    assert results["realizations"].shape == (4, h, w)
    # Physical non-negativity constraint
    assert np.all(results["realizations"] >= 0.0)

    # Exceedance probabilities bounded in [0, 1]
    assert 50.0 in results["exceedance_probabilities"]
    assert 100.0 in results["exceedance_probabilities"]
    p_50 = results["exceedance_probabilities"][50.0]
    assert np.all(p_50 >= 0.0) and np.all(p_50 <= 1.0)
