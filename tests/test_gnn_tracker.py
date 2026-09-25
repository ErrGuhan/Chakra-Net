"""Unit Tests for Mesh GNN Tracker Model Forward Pass and Shape Contract"""

import pytest
import torch
import numpy as np
from models.tracker.gnn_tracker import MeshGNNTracker, MeshConvLayer


def test_mesh_conv_layer():
    """Validates single mesh convolution message passing step."""
    n_nodes = 20
    in_dim = 16
    out_dim = 32
    
    layer = MeshConvLayer(in_dim, out_dim)
    
    # Ring graph edges: (2, 20)
    src = torch.arange(n_nodes)
    dst = (src + 1) % n_nodes
    edge_index = torch.stack([src, dst], dim=0)
    
    x = torch.randn(n_nodes, in_dim)
    edge_attr = torch.ones((n_nodes, 1))
    
    out = layer(x, edge_index, edge_attr)
    assert out.shape == (n_nodes, out_dim)
    assert not torch.isnan(out).any()


def test_mesh_gnn_tracker_temporal_forward():
    """Validates full spatio-temporal forward pass across multiple lead times."""
    n_times = 5
    n_nodes = 30
    in_dim = 6
    hidden_dim = 32
    
    model = MeshGNNTracker(node_in_dim=in_dim, hidden_dim=hidden_dim, num_layers=2)
    model.eval()
    
    # Fake random graph
    src = torch.randint(0, n_nodes, (60,))
    dst = torch.randint(0, n_nodes, (60,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.rand((60, 1))
    
    x_sequence = torch.randn(n_times, n_nodes, in_dim)
    
    with torch.no_grad():
        outputs = model(x_sequence, edge_index, edge_attr)
        
    assert "anomaly_probs" in outputs
    assert "offsets" in outputs
    assert "dispersion" in outputs
    
    probs = outputs["anomaly_probs"]
    assert probs.shape == (n_times, n_nodes)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    
    offsets = outputs["offsets"]
    assert offsets.shape == (n_times, n_nodes, 2)
