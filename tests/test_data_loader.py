"""Unit Tests for ChakraNet Data Loader and Cyclone Phailin Ground Truth"""

import pytest
import numpy as np
from pathlib import Path
from data.loader import PhailinDataLoader
from config import EVENT_CONFIG, ARTIFACTS_DIR


def test_ground_truth_track_structure():
    """Validates structure and completeness of Phailin ground truth track."""
    loader = PhailinDataLoader()
    ground_truth = loader.best_track
    
    assert ground_truth["event_id"] == "phailin_2013"
    assert "landfall" in ground_truth
    assert ground_truth["landfall"]["central_pressure_hpa"] == 940
    
    track = ground_truth["best_track"]
    assert len(track) >= 11  # T+0 to T+120h at 12h intervals
    
    # Verify trajectory heads northwest toward Odisha coast
    first_pt = track[0]
    landfall_pt = track[9]  # T+108h near landfall
    assert first_pt["lat"] < landfall_pt["lat"]
    assert first_pt["lon"] > landfall_pt["lon"]


def test_ensemble_generation_and_shapes():
    """Validates that loaded/generated ensemble matches required tensor shapes and value bounds."""
    loader = PhailinDataLoader()
    dataset = loader.load_or_generate_ensemble()
    
    assert "tp" in dataset
    assert "wind_speed" in dataset
    assert "mslp" in dataset
    assert "imerg_tp" in dataset
    
    tp = dataset["tp"]
    wind = dataset["wind_speed"]
    mslp = dataset["mslp"]
    imerg = dataset["imerg_tp"]
    
    n_members = EVENT_CONFIG.ensemble_members
    n_times = len(EVENT_CONFIG.lead_times_hours)
    n_lats = EVENT_CONFIG.n_coarse_lat
    n_lons = EVENT_CONFIG.n_coarse_lon
    
    assert tp.shape == (n_members, n_times, n_lats, n_lons)
    assert wind.shape == (n_members, n_times, n_lats, n_lons)
    assert mslp.shape == (n_members, n_times, n_lats, n_lons)
    assert imerg.shape == (n_times, n_lats, n_lons)
    
    # Physical value bounds
    assert np.all(tp >= 0.0)
    assert np.all(wind >= 0.0)
    assert np.all(mslp > 900.0) and np.all(mslp < 1030.0)
    
    # Extreme weather verification: peak precipitation near landfall exceeds 150 mm/day
    max_tp_landfall = np.max(tp[:, 9])
    assert max_tp_landfall > 150.0, f"Expected extreme rainfall > 150mm, got {max_tp_landfall}"
