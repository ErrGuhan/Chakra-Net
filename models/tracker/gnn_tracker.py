"""ChakraNet Stage 1 Icosahedral Mesh GNN Anomaly Tracker

Implements a spatio-temporal Graph Neural Network on the M9 spherical geodesic mesh.
- Input Features per node: [EFI, SOT, ensemble_mean, ensemble_spread, mslp_anomaly, wind_speed]
- Spatial Graph Convolution: Message-passing / Graph Attention across spherical mesh neighbors
- Temporal Propagation: Gated Recurrent Unit (GRU) tracking anomaly clusters across lead times
- Output: Per-node anomaly probability, centroid displacement offset, and uncertainty spread
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List


class MeshConvLayer(nn.Module):
    """Spherical Message Passing Convolution on the Icosahedral Mesh."""

    def __init__(self, in_dim: int, out_dim: int, edge_feat_dim: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Message MLP: transforms source node, target node, and spherical geodesic distance
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + edge_feat_dim, out_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(out_dim, out_dim),
        )
        
        # Attention scoring
        self.attn_mlp = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.LeakyReLU(0.2)
        )
        
        # Self node update & normalization
        self.update_linear = nn.Linear(in_dim + out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for mesh convolution.
        
        Args:
            x: Node features of shape (N_nodes, in_dim)
            edge_index: Graph edges of shape (2, N_edges) [src, dst]
            edge_attr: Geodesic edge distances of shape (N_edges, 1)
        Returns:
            out: Updated node features of shape (N_nodes, out_dim)
        """
        src, dst = edge_index[0], edge_index[1]
        x_src = x[src]
        x_dst = x[dst]

        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), 1), device=x.device)

        msg_input = torch.cat([x_src, x_dst, edge_attr], dim=-1)
        messages = self.msg_mlp(msg_input)  # (N_edges, out_dim)
        
        # Aggregate messages into destination nodes
        n_nodes = x.size(0)
        out_dim = messages.size(1)
        aggregated = torch.zeros((n_nodes, out_dim), device=x.device, dtype=messages.dtype)
        
        # Scatter add destination messages
        aggregated.index_add_(0, dst, messages)

        # Count incoming edges for mean normalization
        ones = torch.ones((edge_index.size(1), 1), device=x.device)
        degree = torch.zeros((n_nodes, 1), device=x.device)
        degree.index_add_(0, dst, ones)
        aggregated = aggregated / torch.clamp(degree, min=1.0)

        # Residual update
        combined = torch.cat([x, aggregated], dim=-1)
        updated = self.update_linear(combined)
        return self.layer_norm(F.relu(updated) + (x if self.in_dim == self.out_dim else 0))


class MeshGNNTracker(nn.Module):
    """Spatio-Temporal Mesh GNN Anomaly Detection and Track Predictor."""

    def __init__(
        self,
        node_in_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input feature projection
        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Spatial mesh convolution layers
        self.conv_layers = nn.ModuleList([
            MeshConvLayer(hidden_dim, hidden_dim, edge_feat_dim=1)
            for _ in range(num_layers)
        ])

        # Temporal GRU cell to track anomaly state across lead times
        self.temporal_gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Prediction heads
        # 1. Anomaly node probability (Extreme weather cluster vs background)
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 2. Centroid displacement offset [d_lat, d_lon]
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        # 3. Uncertainty dispersion radius (degrees)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()
        )

    def forward(
        self,
        x_sequence: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Runs the GNN over a sequence of forecast lead times.
        
        Args:
            x_sequence: Tensor of shape (T_times, N_nodes, node_in_dim)
            edge_index: Tensor of shape (2, N_edges)
            edge_attr: Geodesic edge distances of shape (N_edges, 1)
        Returns:
            Dict containing:
                - 'anomaly_probs': (T, N_nodes)
                - 'node_embeddings': (T, N_nodes, hidden_dim)
                - 'offsets': (T, N_nodes, 2)
                - 'dispersion': (T, N_nodes, 1)
        """
        n_times, n_nodes, _ = x_sequence.shape
        device = x_sequence.device

        gru_state = torch.zeros((n_nodes, self.hidden_dim), device=device)

        all_anomaly_probs = []
        all_embeddings = []
        all_offsets = []
        all_dispersions = []

        for t in range(n_times):
            xt = x_sequence[t]  # (N_nodes, node_in_dim)
            h = self.node_embed(xt)  # (N_nodes, hidden_dim)

            # Spatial message passing
            for conv in self.conv_layers:
                h = conv(h, edge_index, edge_attr)

            # Temporal GRU update
            gru_state = self.temporal_gru(h, gru_state)

            # Heads
            prob = self.anomaly_head(gru_state).squeeze(-1)  # (N_nodes,)
            offset = self.offset_head(gru_state)            # (N_nodes, 2)
            disp = self.uncertainty_head(gru_state)         # (N_nodes, 1)

            all_anomaly_probs.append(prob)
            all_embeddings.append(gru_state)
            all_offsets.append(offset)
            all_dispersions.append(disp)

        return {
            "anomaly_probs": torch.stack(all_anomaly_probs, dim=0),
            "node_embeddings": torch.stack(all_embeddings, dim=0),
            "offsets": torch.stack(all_offsets, dim=0),
            "dispersion": torch.stack(all_dispersions, dim=0),
        }
