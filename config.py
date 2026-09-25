"""ChakraNet Configuration & Constants

Centralized configuration for ChakraNet (SIH 2026 PS 26078):
- Historical Case Study: Cyclone Phailin (October 2013, Bay of Bengal / Odisha)
- Grid & Mesh Parameters: 12 km NWP ensemble -> M9 icosahedral mesh -> 5 km downscaling
- Model & Loss Hyperparameters
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

import os

import tempfile

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# On serverless platforms (e.g. Vercel), use /tmp for caching
if os.environ.get("VERCEL") or not os.access(PROJECT_ROOT, os.W_OK):
    CACHE_DIR = Path(tempfile.gettempdir()) / "chakranet_cache"
else:
    CACHE_DIR = DATA_DIR / "cache"

MODELS_DIR = PROJECT_ROOT / "models"
SERVING_DIR = PROJECT_ROOT / "serving"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"
ARTIFACTS_DIR = TESTS_DIR / "artifacts"
API_EXAMPLES_DIR = ARTIFACTS_DIR / "api_examples"

# Auto-load .env file from project root
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k, _v = _k.strip(), _v.strip()
                    if _k not in os.environ:
                        os.environ[_k] = _v
    except Exception:
        pass

# Ensure directories exist safely without throwing on read-only environments
for p in [CACHE_DIR, DATA_DIR, MODELS_DIR, SERVING_DIR, NOTEBOOKS_DIR, TESTS_DIR, ARTIFACTS_DIR, API_EXAMPLES_DIR]:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        pass


@dataclass(frozen=True)
class EventConfig:
    """Historical Case Study Event Configuration (Cyclone Phailin, 2013)"""
    event_id: str = "phailin_2013"
    name: str = "Cyclone Phailin"
    basin: str = "North Indian Ocean / Bay of Bengal"
    landfall_location: str = "Gopalpur, Ganjam district, Odisha"
    dates: Tuple[str, ...] = (
        "2013-10-08",
        "2013-10-09",
        "2013-10-10",
        "2013-10-11",
        "2013-10-12",
        "2013-10-13",
        "2013-10-14",
    )
    # Bounding box covering full life cycle: 10N to 24N, 82E to 96E
    bbox_lat_min: float = 10.0
    bbox_lat_max: float = 24.0
    bbox_lon_min: float = 82.0
    bbox_lon_max: float = 96.0

    # 12 km Coarse Grid specifications (~0.12° resolution)
    coarse_res_deg: float = 0.12
    n_coarse_lat: int = 117  # (24 - 10) / 0.12 + 1
    n_coarse_lon: int = 117  # (96 - 82) / 0.12 + 1

    # 5 km Downscaled Grid specifications (~0.05° resolution)
    fine_res_deg: float = 0.05
    n_fine_lat: int = 281    # (24 - 10) / 0.05 + 1
    n_fine_lon: int = 281    # (96 - 82) / 0.05 + 1

    # Ensemble members & lead times
    ensemble_members: int = 23
    lead_times_hours: Tuple[int, ...] = (0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120)


@dataclass(frozen=True)
class MeshConfig:
    """Icosahedral Mesh Configuration (Resolution Level M9)"""
    # M9 is subdivision level 9 of an icosahedron (roughly 12-25 km node spacing)
    subdivision_level: int = 9
    # Regional clipping bounds for graph representation
    lat_min: float = 5.0
    lat_max: float = 30.0
    lon_min: float = 75.0
    lon_max: float = 100.0


@dataclass(frozen=True)
class ModelConfig:
    """Model hyperparameters for Tracker and Downscaler"""
    # Stage 1: GNN Tracker
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 4
    efi_threshold: float = 0.65
    sot_threshold: float = 0.0

    # Stage 2: Residual Diffusion Downscaler
    downscaler_crop_size: int = 64  # ~64x64 grid in coarse coords (~750x750 km)
    downscaled_size: int = 128      # ~128x128 grid in fine coords (~5 km cells)
    unet_base_channels: int = 32
    diffusion_timesteps: int = 1000
    inference_steps: int = 25       # DDIM sampling steps for fast inference
    num_diffusion_samples: int = 16 # Realizations per event to collapse probability

    # Physics loss weights
    lambda_tail_crps: float = 0.35
    lambda_psd: float = 0.25
    lambda_vimfc: float = 0.15      # Stubbed physical term
    lambda_mass: float = 0.10       # Stubbed physical term


@dataclass(frozen=True)
class AlertConfig:
    """CAP 1.2 Severity Thresholds (mm/24h precipitation probability)"""
    # P(rainfall > 100mm) probability thresholds for severity classification
    prob_minor: float = 0.20
    prob_moderate: float = 0.40
    prob_severe: float = 0.65
    prob_extreme: float = 0.85

    # Corresponding CAP 1.2 severity strings
    severities: Tuple[str, ...] = ("Minor", "Moderate", "Severe", "Extreme")


# Instantiate default singletons
EVENT_CONFIG = EventConfig()
MESH_CONFIG = MeshConfig()
MODEL_CONFIG = ModelConfig()
ALERT_CONFIG = AlertConfig()


@dataclass(frozen=True)
class SupabaseConfig:
    """Supabase Backend Database Configuration"""
    url: str = os.getenv("SUPABASE_URL", "")
    publishable_key: str = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
    secret_key: str = os.getenv("SUPABASE_SECRET_KEY", "")
    jwks_url: str = os.getenv("SUPABASE_JWKS_URL", "")


SUPABASE_CONFIG = SupabaseConfig()
