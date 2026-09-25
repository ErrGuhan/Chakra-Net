"""Unit tests for ChakraNet Stage 2 U-Net Mean Predictor"""

import pytest
import torch
from models.downscaler.unet_mean import UNetMeanPredictor, DoubleConv, DownBlock, UpBlock


def test_unet_mean_architecture():
    model = UNetMeanPredictor(in_channels=3, out_channels=1, base_dim=16)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 100_000, "Model should have substantial capacity"


def test_unet_mean_forward_shape():
    model = UNetMeanPredictor(in_channels=3, out_channels=1, base_dim=16)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)

    assert out.shape == (2, 1, 64, 64)
    # Physical precipitation non-negativity constraint via Softplus
    assert torch.all(out >= 0.0)


def test_unet_mean_gradient_flow():
    model = UNetMeanPredictor(in_channels=3, out_channels=1, base_dim=16)
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
