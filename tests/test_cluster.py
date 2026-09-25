"""Unit Tests for Anomaly Cluster and 4D Bounding Box Extraction"""

import pytest
import numpy as np
from data.mesh import IcosahedralMesh
from models.tracker.cluster import AnomalyClusterExtractor


def test_4d_bounding_box_extraction():
    """Asserts that 4D crop box correctly encapsulates high anomaly nodes."""
    mesh = IcosahedralMesh()
    extractor = AnomalyClusterExtractor(mesh, anomaly_threshold=0.60)
    
    # Simulate high anomaly localized around Odisha coast (19°N, 85°E)
    dist = np.sqrt((mesh.lats - 19.26)**2 + (mesh.lons - 84.91)**2)
    fake_probs = np.exp(-(dist / 1.5)**2).astype(np.float32)
    
    bbox_info = extractor.extract_4d_bounding_box(
        fake_probs, time_index=9, lead_time_hours=108, padding_deg=1.5
    )
    
    assert "bbox_4d" in bbox_info
    b = bbox_info["bbox_4d"]
    
    assert b["lat_min"] < 19.26 < b["lat_max"]
    assert b["lon_min"] < 84.91 < b["lon_max"]
    assert b["lead_time_range_hours"] == [96, 120]
    assert bbox_info["active_node_count"] > 0


def test_uncertainty_cone_calculation():
    """Asserts that multi-member uncertainty cone broadens with forecast lead time."""
    mesh = IcosahedralMesh()
    extractor = AnomalyClusterExtractor(mesh)
    
    n_members = 23
    n_times = 11
    lats = np.linspace(10, 24, 50, dtype=np.float32)
    lons = np.linspace(82, 96, 50, dtype=np.float32)
    lead_times = np.arange(0, 132, 12, dtype=np.int32)
    
    # Synthetic MSLP vortex fields
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    mslp = np.full((n_members, n_times, 50, 50), 1010.0, dtype=np.float32)
    
    rng = np.random.RandomState(42)
    for m in range(n_members):
        for t in range(n_times):
            # Member center with expanding dispersion
            c_lat = 10.0 + t * 0.9 + rng.normal(0, 0.05 * (t + 1))
            c_lon = 94.0 - t * 0.9 + rng.normal(0, 0.05 * (t + 1))
            dist = np.sqrt((lat_grid - c_lat)**2 + (lon_grid - c_lon)**2)
            mslp[m, t] = 1010.0 - 50.0 * np.exp(-(dist / 1.0)**2)
            
    cone = extractor.compute_ensemble_uncertainty_cone(mslp, lats, lons, lead_times)
    
    assert "cone_polygon" in cone
    assert len(cone["cone_polygon"]) > 10
    
    radii = cone["dispersion_radii_deg"]
    assert len(radii) == n_times
    # Verify uncertainty expands over lead time
    assert radii[-1] > radii[0], f"Expected expanding cone radius, got {radii[0]} -> {radii[-1]}"
