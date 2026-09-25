"""Unit Tests for Icosahedral M9 Mesh and Spherical Regridder"""

import pytest
import numpy as np
from data.mesh import IcosahedralMesh, create_base_icosahedron, cartesian_to_latlon, latlon_to_cartesian
from data.regrid import SphericalRegridder


def test_base_icosahedron_geometry():
    """Validates icosahedron base geometry on the unit sphere."""
    vertices, faces = create_base_icosahedron()
    assert vertices.shape == (12, 3)
    assert faces.shape == (20, 3)
    
    # All vertices must lie on the unit sphere
    norms = np.linalg.norm(vertices, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_latlon_cartesian_roundtrip():
    """Validates coordinate conversion roundtrip."""
    test_lats = np.array([12.5, 18.0, 22.0], dtype=np.float64)
    test_lons = np.array([85.0, 88.0, 92.0], dtype=np.float64)
    
    xyz = latlon_to_cartesian(test_lats, test_lons)
    reconstructed_lats, reconstructed_lons = cartesian_to_latlon(xyz)
    
    np.testing.assert_allclose(reconstructed_lats, test_lats, atol=1e-5)
    np.testing.assert_allclose(reconstructed_lons, test_lons, atol=1e-5)


def test_m9_regional_mesh():
    """Validates the M9 regional icosahedral mesh structure and edge index."""
    mesh = IcosahedralMesh()
    assert mesh.num_nodes > 1000, f"Expected >1000 nodes for regional M9 mesh, got {mesh.num_nodes}"
    assert mesh.edge_index.shape[0] == 2
    assert mesh.num_edges > mesh.num_nodes * 4
    
    # Check that lats and lons fall within target region
    assert np.all(mesh.lats >= 4.0) and np.all(mesh.lats <= 31.0)
    assert np.all(mesh.lons >= 74.0) and np.all(mesh.lons <= 101.0)


def test_spherical_regridder_forward_backward():
    """Validates regridding from regular lat/lon grid to mesh and back."""
    mesh = IcosahedralMesh()
    source_lats = np.linspace(10.0, 24.0, 30, dtype=np.float32)
    source_lons = np.linspace(82.0, 96.0, 30, dtype=np.float32)
    
    lon_grid, lat_grid = np.meshgrid(source_lons, source_lats)
    # Synthetic smooth vortex test field
    center_lat, center_lon = 18.0, 88.0
    dist = np.sqrt((lat_grid - center_lat)**2 + (lon_grid - center_lon)**2)
    test_field = np.exp(-(dist / 2.0)**2).astype(np.float32)
    
    regridder = SphericalRegridder(mesh, source_lats, source_lons)
    
    # 1. Forward regrid: grid -> mesh
    mesh_field = regridder.grid_to_mesh(test_field)
    assert mesh_field.shape == (mesh.num_nodes,)
    assert np.max(mesh_field) > 0.8  # Peak intensity preserved
    
    # 2. Backward regrid: mesh -> grid
    reconstructed_grid = regridder.mesh_to_grid(mesh_field, source_lats, source_lons)
    assert reconstructed_grid.shape == (30, 30)
    
    # Maximum difference within reasonable interpolation tolerance
    max_diff = np.max(np.abs(test_field - reconstructed_grid))
    assert max_diff < 0.25, f"Interpolation error too high: {max_diff}"
