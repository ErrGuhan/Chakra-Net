"""Unit tests for ChakraNet Downscaler Loss Functions and Physics Stubs"""

import pytest
import torch
from models.downscaler.losses import (
    DiffusionMSELoss,
    TailWeightedCRPSLoss,
    RadialPSDLoss,
    PhysicsStubs,
    DownscalerCompositeLoss,
)


def test_diffusion_mse_loss():
    loss_fn = DiffusionMSELoss()
    pred = torch.randn(2, 1, 16, 16)
    target = torch.randn(2, 1, 16, 16)
    val = loss_fn(pred, target)

    assert val.ndim == 0
    assert val.item() >= 0.0


def test_tail_weighted_crps_loss():
    loss_fn = TailWeightedCRPSLoss(tail_threshold=50.0)
    # Ensemble of 8 members, batch 2, 16x16
    ensemble = torch.randn(2, 8, 16, 16).abs() * 60.0
    target = torch.randn(2, 1, 16, 16).abs() * 55.0

    val = loss_fn(ensemble, target)
    assert val.ndim == 0
    assert not torch.isnan(val)


def test_radial_psd_loss():
    loss_fn = RadialPSDLoss()
    field1 = torch.randn(2, 1, 32, 32)
    field2 = torch.randn(2, 1, 32, 32)

    val = loss_fn(field1, field2)
    assert val.ndim == 0
    assert val.item() >= 0.0

    # Self-distance should be zero
    zero_dist = loss_fn(field1, field1)
    assert zero_dist.item() < 1e-4


def test_physics_stubs():
    # VIMFC stub
    u = torch.randn(1, 4, 16, 16)
    v = torch.randn(1, 4, 16, 16)
    q = torch.rand(1, 4, 16, 16)
    p = torch.rand(1, 1, 16, 16) * 50.0

    vimfc_loss = PhysicsStubs.vimfc_conservation_loss(u, v, q, p)
    assert isinstance(vimfc_loss, torch.Tensor)

    # Mass conservation stub
    rho = torch.ones(1, 4, 16, 16)
    vel = torch.randn(1, 3, 16, 16)
    mass_loss = PhysicsStubs.atmospheric_mass_conservation_loss(rho, vel)
    assert isinstance(mass_loss, torch.Tensor)


def test_composite_loss():
    comp_loss = DownscalerCompositeLoss()
    pred_noise = torch.randn(2, 1, 16, 16)
    target_noise = torch.randn(2, 1, 16, 16)
    ens = torch.randn(2, 4, 16, 16).abs()
    target_hr = torch.randn(2, 1, 16, 16).abs()

    out = comp_loss(pred_noise, target_noise, ens, target_hr)
    assert "loss_diff" in out
    assert "loss_crps" in out
    assert "loss_psd" in out
    assert "loss_total" in out
    assert out["loss_total"] > 0.0
