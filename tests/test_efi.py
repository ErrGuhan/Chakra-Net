"""Unit Tests for Lalaurette (2003) EFI and Shift of Tails (SOT)"""

import pytest
import numpy as np
from models.tracker.efi import ExtremeForecastIndex


def test_efi_bounds_and_symmetry():
    """Asserts that EFI is strictly bounded within [-1.0, 1.0]."""
    efi_calc = ExtremeForecastIndex(n_quadrature_points=30)
    
    # 23 members over 100 spatial nodes
    n_members = 23
    n_nodes = 100
    
    # Case 1: Standard normal ensemble matching climatology -> EFI near 0.0
    rng = np.random.RandomState(42)
    ens_normal = rng.gamma(shape=1.5, scale=10.0, size=(n_members, n_nodes)).astype(np.float32)
    efi_normal = efi_calc.compute_efi(ens_normal)
    
    assert efi_normal.shape == (n_nodes,)
    assert np.all(efi_normal >= -1.0) and np.all(efi_normal <= 1.0)

    # Case 2: Extreme anomalous storm ensemble -> EFI should be high (> 0.60)
    ens_extreme = rng.gamma(shape=5.0, scale=30.0, size=(n_members, n_nodes)).astype(np.float32)
    efi_extreme = efi_calc.compute_efi(ens_extreme)
    
    assert np.all(efi_extreme >= -1.0) and np.all(efi_extreme <= 1.0)
    assert np.mean(efi_extreme) > 0.50, f"Expected anomalous EFI > 0.50, got {np.mean(efi_extreme)}"


def test_sot_computation():
    """Asserts that SOT detects high ensemble tail departures."""
    efi_calc = ExtremeForecastIndex()
    n_members = 23
    n_nodes = 50
    rng = np.random.RandomState(42)
    
    # Extreme tail event
    ens = rng.gamma(shape=1.5, scale=5.0, size=(n_members, n_nodes)).astype(np.float32)
    # Inject extreme 90th percentile values into first 10 nodes
    ens[-3:, :10] += 120.0
    
    sot = efi_calc.compute_sot(ens, tail_percentile=0.90)
    assert sot.shape == (n_nodes,)
    assert np.all(sot >= 0.0)
    # The first 10 nodes should have strong shift of tails
    assert np.mean(sot[:10]) > np.mean(sot[10:])
