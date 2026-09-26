# ChakraNet: Dual-Stage Extreme Weather Tracking & Diffusion Downscaling Pipeline

**SIH 2026 Problem Statement 26078 — NCMRWF / MoES**

ChakraNet is an operational atmospheric physics and deep learning pipeline designed to solve the spatial and temporal resolution gap in numerical weather prediction (NWP) ensemble forecasts for high-impact extreme weather events (tropical cyclones, severe precipitation, and coastal storm surges) across the Indian subcontinent and Bay of Bengal.

---

## 1. Architecture Overview

ChakraNet utilizes a two-stage hybrid architecture:

```
[Raw 12 km Multi-Member NWP Ensemble (23-50 members)]
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: M9 Icosahedral Mesh Anomaly Tracker                │
│  - Lalaurette (2003) Extreme Forecast Index (EFI) & SOT     │
│  - Spherical Graph Convolutional Network (Mesh GNN)         │
│  - Dynamic 4D Spatio-Temporal Bounding Box & Cone Extraction│
└─────────────────────────────────────────────────────────────┘
                       │ Dynamic Bounding Crop
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Physics-Constrained Residual Diffusion Downscaler  │
│  - 5 km DEM Topography & Land-Sea Mask Conditioning         │
│  - Deterministic 2D U-Net Mean Precipitation Model          │
│  - CorrDiff-style Residual Diffusion Model (DDIM Reverse)   │
│  - 16 Stochastic Realization Members & Exceedance Maps      │
└─────────────────────────────────────────────────────────────┘
                       │ 5 km Probability Grid P(Rain > τ)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Operational Serving & Multi-Hazard Alerting        │
│  - High-performance FastAPI REST API & Vector GeoJSON       │
│  - MapLibre GL JS Flat Tactical Defense Dashboard           │
│  - OASIS CAP 1.2 Compliant Multi-Lingual Alert Engine       │
│  - Groq LLM Operational Bulletins (English, Hindi, Odia)   │
│  - Simulated NDMA SACHET / Bhashini Dispatch Gateway        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formulations

### Stage 1: Lalaurette (2003) Extreme Forecast Index (EFI)
$$\text{EFI} = \frac{2}{\pi} \int_0^1 \frac{p - F_f(Q_c(p))}{\sqrt{p(1-p)}} \, dp$$
Where $F_f$ represents the empirical cumulative distribution function (ECDF) of the forecast ensemble, and $Q_c(p)$ is the inverse CDF (quantile function) of the model reforecast climatology.

### Stage 2: Composite Downscaler Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda_1 \text{CRPS}_{\text{tail}} + \lambda_2 \mathcal{L}_{\text{PSD}}$$
- **Diffusion Loss ($\mathcal{L}_{\text{diff}}$)**: $\mathbb{E}_{x_0, \epsilon, t} \left[ \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2 \right]$
- **Tail-Weighted CRPS ($\text{CRPS}_{\text{tail}}$)**: Penalizes probabilistic distribution error with elevated weighting on extreme tail precipitation ($> 30\text{ mm}$).
- **Radial Spectral Loss ($\mathcal{L}_{\text{PSD}}$)**: Enforces preservation of high-wavenumber kinetic energy in the 2D Fourier domain, matching the Kolmogorov $-5/3$ spectral cascade.
- **Physics-Guided Conservation Stubs**: VIMFC ($\mathcal{L}_{\text{VIMFC}}$) and Atmospheric Mass Continuity ($\mathcal{L}_{\text{mass}}$) are documented v2 milestones with forward-integration hooks implemented in `models/losses.py`. The current pipeline is **physics-constrained** (via PSD, CRPS, and MSE loss), not yet **physics-closed**.

---

## 3. Quantitative KPI Benchmarks (Milestone 3)

Measured directly on the Cyclone Phailin evaluation domain (128x128 grid at 5 km resolution, 16 ensemble realizations):

| KPI / Evaluation Dimension | Bilinear Baseline (Current NWP) | ChakraNet Measured (Prototype) | Deck Target | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- |
| **High-Frequency PSD Retention ($k > 0.3 k_N$)** | `50.5%` | **`98.7%`** | `> 70.0%` | **PASSED** (Eliminates spectral blurring; raw amplification ratio 14.5× — see `metrics.json`) |
| **P99 Extreme Rainfall Bias** | `4.87%` | **`5.72%`** | `< 12.0%` | **PASSED** (Accurate extreme eyewall capture) |
| **CRPS Ensemble Improvement** | `0.0%` (Ref: `4.21 mm`) | **`+16.44%`** (`3.52 mm`) | `> 15.0%` | **PASSED** (Calibrated probabilistic spread) |
| **Inference Latency (Stage 2 kernel, per 128×128 crop)** | `~4.2 ms` | **`7.6 ms`** | `< 2500 ms` (CPU) | **PASSED** — Full pipeline (16 members, L4 GPU): < 3 min; CPU: < 10 min |

Full details and honest disclosures are documented in [LIMITATIONS.md](LIMITATIONS.md) and [metrics.json](metrics.json).

---

## 4. Repository Structure

```
ChakraNet/
├── config.py                 # Core domain constants, mesh params & event config
├── metrics.json              # Standardized quantitative benchmark metrics
├── LIMITATIONS.md            # Real vs Simulated vs Stubs matrix & safety disclosure
├── data/
│   ├── loader.py             # Multi-member ensemble loader & caching
│   ├── mesh.py               # M9 icosahedral mesh generator & adjacency
│   ├── regrid.py             # Spherical interpolation & coordinate mapping
│   ├── visualize.py          # Basemap plotting & artifact renderer
│   ├── phailin_ground_truth.json # IMD historical best track reference
│   └── odisha_coastal_districts.geojson # Administrative boundaries
├── models/
│   ├── tracker/
│   │   ├── efi.py            # Lalaurette EFI & SOT numerical implementation
│   │   ├── gnn_tracker.py    # Temporal message-passing GNN
│   │   ├── cluster.py        # 4D bounding box & uncertainty cone extraction
│   │   └── evaluate.py       # Track validation vs IMD ground truth
│   ├── downscaler/
│   │   ├── crop_condition.py # Topography (DEM) & regional cropping
│   │   ├── unet_mean.py      # Deterministic 2D U-Net mean model
│   │   ├── diffusion_residual.py # Residual diffusion backbone & DDIM reverse sampler
│   │   ├── sampler.py        # 16-member stochastic realization & exceedance sampler
│   │   ├── losses.py         # MSE, Tail-CRPS, Radial-PSD & physics stubs
│   │   └── evaluate.py       # 4-panel comparison & 2D FFT PSD analysis
│   └── metrics.py            # Quantitative KPI calculation & benchmark suite
├── serving/
│   ├── api/
│   │   ├── main.py           # FastAPI application entry point
│   │   ├── routes.py         # REST endpoints for events, tracks, hazards, dispatch
│   │   ├── db.py             # Geospatial data service
│   │   └── cap_engine.py     # OASIS CAP 1.2 XML generator & Groq LLM advisory
│   └── frontend/
│       ├── index.html        # Clean, flat, high-contrast command dashboard
│       ├── styles.css        # Precision design system (zero gradients)
│       └── app.js            # MapLibre GL controller & layer management
├── notebooks/
│   ├── 00_data_plumbing.ipynb       # Milestone 0 Walkthrough
│   ├── 01_mesh_gnn_tracker.ipynb    # Milestone 1 Walkthrough
│   └── 02_residual_diffusion.ipynb  # Milestone 2 Walkthrough
├── tests/                    # Comprehensive unit and integration test suite (45 tests)
│   └── artifacts/            # Output figures, comparison charts, and API samples
└── tasks/                    # Task checklists and implementation plans
```

---

## 5. Quickstart & Verification

### Running the Complete Test Suite
```bash
# Run all 45 unit and integration tests
uv run pytest tests/ -v
```

### Running the Operational Dashboard
```bash
# Launch FastAPI backend server
uv run python -m uvicorn serving.api.main:app --port 8000
```
Open `http://localhost:8000` in your web browser.

---

## 6. Generated Visual Artifacts

All key milestone artifacts are saved in `tests/artifacts/`:
- `milestone0_ensemble_mean.png`: Raw 12 km ensemble mean precipitation over the Bay of Bengal.
- `milestone1_tracker_cone.png`: GNN predicted trajectory and uncertainty cone vs IMD historical ground truth.
- `milestone2_downscaler_comparison.png`: 4-panel comparison (Coarse 12 km, Bilinear, ChakraNet Diffusion, IMERG).
- `milestone2_psd_analysis.png`: Radially averaged 2D FFT Power Spectral Density curve against Kolmogorov $-5/3$ turbulence.
- `milestone3_kpi_table.md`: Benchmark KPI validation table.
- `api_examples/`: Verified OASIS CAP 1.2 XML and GeoJSON response payloads.
- `verified_4d_volumetric_map.png`: 4D spatiotemporal volumetric MapLibre GL visualization with 3D convective column extrusion.
- `verified_5km_radius_buffer.png`: Accurate 5.0 km WGS-84 geodesic radius buffer ($78.54\text{ km}^2$) and $5\text{ km} \times 5\text{ km}$ ($25\text{ km}^2$) grid cell drill-down.
- `verified_dashboard_overview.png`: Responsive, zero-gradient flat defense command interface at 1280x720.
