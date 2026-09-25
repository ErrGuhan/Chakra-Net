"""ChakraNet Spherical Regridding Utilities

Provides high-performance spherical regridding:
1. Regular Lat/Lon 12 km NWP ensemble grid -> Icosahedral M9 Mesh nodes
2. Icosahedral M9 Mesh nodes -> Target regular grid for visualization and verification
Uses inverse distance weighting (IDW) on 3D Cartesian geodesic coordinates.
"""

import numpy as np
from typing import Tuple, Optional
from data.mesh import IcosahedralMesh, latlon_to_cartesian


class SphericalRegridder:
    """Interpolates between regular lat/lon grids and icosahedral mesh nodes."""

    def __init__(self, mesh: IcosahedralMesh, source_lats: np.ndarray, source_lons: np.ndarray, k_neighbors: int = 4):
        """Precomputes interpolation weights from regular grid to mesh nodes.
        
        Args:
            mesh: Target IcosahedralMesh instance
            source_lats: 1D array of latitude coordinates of the regular grid
            source_lons: 1D array of longitude coordinates of the regular grid
            k_neighbors: Number of nearest source points for IDW
        """
        self.mesh = mesh
        self.source_lats = source_lats
        self.source_lons = source_lons
        self.k_neighbors = k_neighbors
        
        # Flatten regular 2D grid to 1D point list
        lon_grid, lat_grid = np.meshgrid(source_lons, source_lats)
        self.grid_shape = lon_grid.shape  # (n_lat, n_lon)
        self.n_grid_points = lon_grid.size
        
        flat_lats = lat_grid.ravel()
        flat_lons = lon_grid.ravel()
        
        # Build KDTree on source grid Cartesian points
        from scipy.spatial import cKDTree
        grid_xyz = latlon_to_cartesian(flat_lats, flat_lons)
        self.grid_kdtree = cKDTree(grid_xyz)
        
        # Query nearest k grid points for each mesh node
        distances, indices = self.grid_kdtree.query(self.mesh.vertices_xyz, k=k_neighbors)
        
        # Compute Inverse Distance Weights (avoid division by zero)
        eps = 1e-8
        weights = 1.0 / (distances + eps)
        weights_sum = np.sum(weights, axis=-1, keepdims=True)
        self.grid_to_mesh_weights = (weights / weights_sum).astype(np.float32)  # (N_nodes, k)
        self.grid_to_mesh_indices = indices  # (N_nodes, k)

    def grid_to_mesh(self, grid_data: np.ndarray) -> np.ndarray:
        """Regrids scalar field from regular (..., n_lat, n_lon) grid to mesh nodes (..., N_nodes).
        
        Args:
            grid_data: Tensor or ndarray with trailing dimensions (n_lat, n_lon)
        Returns:
            mesh_data: ndarray with trailing dimension (num_nodes,)
        """
        prefix_shape = grid_data.shape[:-2]
        flat_grid = grid_data.reshape(-1, self.n_grid_points)  # (B, n_grid_points)
        
        # Gather k neighbor values for all mesh nodes
        # shape: (B, N_nodes, k)
        gathered = flat_grid[:, self.grid_to_mesh_indices]
        
        # Apply weights: sum over k
        mesh_flat = np.sum(gathered * self.grid_to_mesh_weights[None, :, :], axis=-1)
        
        output_shape = prefix_shape + (self.mesh.num_nodes,)
        return mesh_flat.reshape(output_shape).astype(np.float32)

    def mesh_to_grid(self, mesh_data: np.ndarray, target_lats: np.ndarray, target_lons: np.ndarray, k: int = 3) -> np.ndarray:
        """Projects mesh node values (..., N_nodes) onto a target regular lat/lon grid (..., n_lat, n_lon)."""
        target_lon_grid, target_lat_grid = np.meshgrid(target_lons, target_lats)
        target_xyz = latlon_to_cartesian(target_lat_grid.ravel(), target_lon_grid.ravel())
        
        distances, indices = self.mesh.kdtree.query(target_xyz, k=k)
        eps = 1e-8
        weights = 1.0 / (distances + eps)
        weights_sum = np.sum(weights, axis=-1, keepdims=True)
        norm_weights = (weights / weights_sum).astype(np.float32)
        
        prefix_shape = mesh_data.shape[:-1]
        flat_mesh = mesh_data.reshape(-1, self.mesh.num_nodes)
        
        gathered = flat_mesh[:, indices]  # (B, N_target, k)
        interpolated = np.sum(gathered * norm_weights[None, :, :], axis=-1)
        
        out_shape = prefix_shape + (len(target_lats), len(target_lons))
        return interpolated.reshape(out_shape).astype(np.float32)
