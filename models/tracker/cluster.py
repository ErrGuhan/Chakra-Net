"""ChakraNet Spatio-Temporal Clustering & 4D Bounding Box Extractor

Extracts coherent anomaly clusters from GNN node probabilities and EFI:
1. Derives 4D Bounding Box: [lat_min, lat_max, lon_min, lon_max, vertical_level, t_start, t_end]
2. Computes Cross-Member Uncertainty Cone from ensemble vortex dispersions
3. Exports GeoJSON structures for Stage 2 downscaler cropping and serving
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from data.mesh import IcosahedralMesh


class AnomalyClusterExtractor:
    """Extracts spatio-temporal clusters and uncertainty cones from GNN outputs."""

    def __init__(self, mesh: IcosahedralMesh, anomaly_threshold: float = 0.50):
        self.mesh = mesh
        self.anomaly_threshold = anomaly_threshold

    def extract_4d_bounding_box(
        self,
        node_probs: np.ndarray,
        time_index: int,
        lead_time_hours: int,
        padding_deg: float = 1.5,
    ) -> Dict[str, Any]:
        """Calculates dynamic 4D bounding box around the active anomaly cluster at lead time.
        
        Args:
            node_probs: (N_nodes,) anomaly probabilities from GNN
            time_index: Time step index
            lead_time_hours: Forecast lead time in hours
            padding_deg: Geographic padding around cluster bounds (degrees)
        Returns:
            Dict containing bbox coordinates and metadata
        """
        active_mask = node_probs >= self.anomaly_threshold

        if not np.any(active_mask):
            # Fallback to top 5% highest probability nodes
            top_k = max(5, int(0.05 * len(node_probs)))
            active_indices = np.argsort(node_probs)[-top_k:]
        else:
            active_indices = np.where(active_mask)[0]

        active_lats = self.mesh.lats[active_indices]
        active_lons = self.mesh.lons[active_indices]

        # Calculate weighted centroid
        weights = node_probs[active_indices] + 1e-4
        weights /= np.sum(weights)
        centroid_lat = float(np.sum(active_lats * weights))
        centroid_lon = float(np.sum(active_lons * weights))

        # Dynamic bounding box
        lat_min = float(max(0.0, np.min(active_lats) - padding_deg))
        lat_max = float(min(40.0, np.max(active_lats) + padding_deg))
        lon_min = float(max(60.0, np.min(active_lons) - padding_deg))
        lon_max = float(min(110.0, np.max(active_lons) + padding_deg))

        return {
            "time_index": time_index,
            "lead_time_hours": lead_time_hours,
            "centroid": [centroid_lat, centroid_lon],
            "bbox_4d": {
                "lat_min": round(lat_min, 4),
                "lat_max": round(lat_max, 4),
                "lon_min": round(lon_min, 4),
                "lon_max": round(lon_max, 4),
                "vertical_level_hpa": [850, 500, 200],  # Deep cyclonic tropospheric levels
                "lead_time_range_hours": [max(0, lead_time_hours - 12), lead_time_hours + 12],
            },
            "active_node_count": len(active_indices),
        }

    def compute_ensemble_uncertainty_cone(
        self,
        ensemble_mslp: np.ndarray,
        lats_1d: np.ndarray,
        lons_1d: np.ndarray,
        lead_times: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculates multi-member vortex centroids and builds the uncertainty cone polygon.
        
        Args:
            ensemble_mslp: Shape (members, times, lat, lon)
            lats_1d: 1D latitude coordinates
            lons_1d: 1D longitude coordinates
            lead_times: 1D lead time hours
        Returns:
            Dict containing centroid tracks, dispersion radius, and cone polygon coordinates
        """
        n_members, n_times, _, _ = ensemble_mslp.shape
        member_centroids = np.zeros((n_members, n_times, 2), dtype=np.float32)

        # Track each member's vortex center (minimum MSLP)
        for m in range(n_members):
            for t in range(n_times):
                field = ensemble_mslp[m, t]
                min_idx = np.unravel_index(np.argmin(field), field.shape)
                member_centroids[m, t, 0] = lats_1d[min_idx[0]]
                member_centroids[m, t, 1] = lons_1d[min_idx[1]]

        # Compute ensemble mean track and standard deviation (cone radius)
        mean_track = np.mean(member_centroids, axis=0)  # (times, 2)
        std_track = np.std(member_centroids, axis=0)    # (times, 2)
        dispersion_radii_deg = np.linalg.norm(std_track, axis=-1)  # (times,)

        # Build cone polygon envelope (left boundary and right boundary)
        cone_left = []
        cone_right = []

        for t in range(n_times):
            c_lat, c_lon = mean_track[t, 0], mean_track[t, 1]
            r = max(0.20, dispersion_radii_deg[t] * 1.645)  # 90% confidence interval multiplier
            cone_left.append([float(c_lon - r * 0.7), float(c_lat + r * 0.7)])
            cone_right.append([float(c_lon + r * 0.7), float(c_lat - r * 0.7)])

        cone_polygon = cone_left + cone_right[::-1] + [cone_left[0]]

        return {
            "mean_track": mean_track.tolist(),
            "dispersion_radii_deg": dispersion_radii_deg.tolist(),
            "cone_polygon": cone_polygon,
            "member_centroids": member_centroids.tolist(),
        }
