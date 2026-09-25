# ChakraNet — Implementation Plan

**Project:** ChakraNet (SIH 2026 · Problem Statement 26078 · NCMRWF / MoES)  
**Goal:** Build a working prototype of a two-stage hybrid AI pipeline that tracks extreme-weather anomalies in a 12 km medium-range ensemble forecast, downscales the tracked crop region to ~5 km probabilistic hazard maps via physics-constrained residual diffusion, and emits CAP 1.2 XML alerts with an interactive MapLibre GL JS dashboard.  
**Historical Case Study:** Cyclone Phailin (October 2013, Bay of Bengal / Odisha coast) — well-documented ground truth via IMD, JTWC, and GPM IMERG.

---

## 1. Overview & Architecture Decisions

ChakraNet bridges the operational gap between global medium-range coarse ensemble forecasts (~12–28 km) and hyper-local, actionable disaster risk alerting (<5 km) without full-grid super-resolution compute costs.

### Architectural Decisions

1. **Two-Stage Pipeline (Track then Crop then Downscale):**
   - *Stage 1 (Anomaly Tracker):* Operates globally/regionally on an icosahedral M9 mesh (~28 km to 12 km cell representation). Computes Extreme Forecast Index (EFI) and Shift of Tails (SOT) per node against a climatological baseline (Lalaurette 2003 formulation). An icosahedral Mesh Graph Neural Network (GNN) groups anomaly nodes into spatiotemporally coherent 4D bounding boxes `[lat_min, lat_max, lon_min, lon_max, vertical_level, t_start, t_end]` and calculates an ensemble uncertainty cone.
   - *Stage 2 (Tile-Only Residual Diffusion Downscaler):* Instead of downscaling the entire subcontinent (which would require immense compute), only the 4D tracker crop is routed to Stage 2. A U-Net generates the deterministic base mean, and a residual diffusion model (CorrDiff / HuggingFace Diffusers architecture) restores high-frequency stochastic turbulence conditioned on 5 km static elevation (DEM) and land-use rasters.
   - *Sampling & Probability of Exceedance:* Generates 16 diffusion samples per time step to collapse into per-cell probability of exceedance: $P(\text{rainfall} > \text{threshold})$.
2. **Loss Function Realism & Transparency:**
   - Composite loss: $\mathcal{L} = \mathcal{L}_{\text{diffusion}} + \lambda_1 \text{CRPS}_{\text{tail}} + \lambda_2 \mathcal{L}_{\text{PSD}} + \lambda_3 \mathcal{L}_{\text{VIMFC}} + \lambda_4 \mathcal{L}_{\text{mass}}$.
   - Fully implemented in prototype: $\mathcal{L}_{\text{diffusion}}$, $\text{CRPS}_{\text{tail}}$ (P99 tail-weighted), and $\mathcal{L}_{\text{PSD}}$ (Power Spectral Density matching).
   - Documented physical stubs: $\mathcal{L}_{\text{VIMFC}}$ (Vertically Integrated Moisture Flux Convergence) and mass-conservation penalty are documented with rigorous physical equations and clear TODO hooks; no synthetic or fabricated loss numbers are reported.
3. **Serving & Storage Architecture:**
   - Lightweight FastAPI backend exposing REST endpoints for tracks, GeoJSON hazard maps, and CAP 1.2 XML alerts.
   - SQLite with GeoJSON storage fallback (with PostGIS schemas provided) to ensure zero-friction, self-contained local testing in any execution environment.
4. **Alerting & Multilingual Delivery:**
   - Alerts adhere strictly to the Common Alerting Protocol (CAP v1.2) XML standard, validated against the official OASIS XML Schema, classified into 4 severity tiers (Minor, Moderate, Severe, Extreme).
   - Mocked SACHET / Bhashini dispatch clearly badged as `SIMULATED` in code, logs, and frontend UI.
5. **Execution & Hardware Hybrid Strategy:**
   - All modules provide standard Python CLI entry points and unit tests for local reproduction (CPU fallback mode).
   - Jupyter notebooks (`.ipynb`) with Google Colab / Kaggle 1-click execution support are maintained alongside CLI scripts for GPU-accelerated diffusion inference.

---

## 2. Milestones & Checkpoint Structure

As mandated by the build guidelines, work is split into **7 sequential milestones (0 through 6)**. Each milestone has an explicit, verifiable exit criterion (plots, test results, or recorded artifacts) and requires human review at its checkpoint before moving to the next.

```
Milestone 0: Repo & Data Plumbing
      │
      ▼ [Checkpoint: Historical Ensemble Loaded & Visualized]
Milestone 1: Stage 1 — Mesh GNN Anomaly Tracker
      │
      ▼ [Checkpoint: Tracker Box & Cone Plotted vs. Real Storm Track]
Milestone 2: Stage 2 — Residual Diffusion Downscaler
      │
      ▼ [Checkpoint: Downscaler 5 km vs. Bilinear vs. Ground Truth + PSD]
Milestone 3: Metrics & Honesty Pass
      │
      ▼ [Checkpoint: metrics.json & KPI Table Grounded in Artifacts]
Milestone 4: Serving & CAP 1.2 Alerting Layer
      │
      ▼ [Checkpoint: FastAPI Tests Pass + Validated CAP 1.2 XML Examples]
Milestone 5: Frontend Dashboard & Demo Assembly
      │
      ▼ [Checkpoint: MapLibre Interactive Map + Hindi/English Alert Modal]
Milestone 6: Honest Documentation & Reproducibility
      │
      ▼ [Checkpoint: Full Traceability & Verification Review]
```

---

## 3. Detailed Milestone Tasks

### Milestone 0 — Repo & Data Plumbing
- **Task 0.1: Repository Scaffolding & Environment Setup**
  - Create directory layout: `data/`, `models/tracker/`, `models/downscaler/`, `serving/api/`, `serving/frontend/`, `notebooks/`, `tests/artifacts/`.
  - Configure Python dependencies (`requirements.txt`, `pyproject.toml`) including PyTorch, PyTorch Geometric / DGL, Diffusers, xarray, Zarr, FastAPI, and pytest.
  - Setup Python 3.11 environment.
- **Task 0.2: Historical Event Data Ingestion & Caching**
  - Implement data pipeline for Cyclone Phailin (October 2013) ensemble forecast data (ECMWF open data / ERA5 / IMDAA 12 km ensemble substitute) and IMERG ground truth.
  - Provide local caching with Zarr / NetCDF format.
- **Task 0.3: M9 Icosahedral Mesh Generator & Regridding**
  - Generate icosahedral spherical mesh nodes at subdivision level M9.
  - Implement bilinear/nearest spherical regridding from regular lat/lon 12 km grid to M9 mesh nodes.
- **Task 0.4: Raw Ensemble Mean Visualizer & Test Verification**
  - Script to compute and render the 12 km raw ensemble mean as a geographic map plot.
  - Generate Colab-friendly notebook (`notebooks/00_data_plumbing.ipynb`).
  - *Exit Criterion:* Map image committed to `tests/artifacts/milestone0_ensemble_mean.png` and passing unit test.

### Milestone 1 — Tracker (Stage 1)
- **Task 1.1: Climatology Baseline, EFI & SOT Computation**
  - Implement Lalaurette (2003) Extreme Forecast Index (EFI) and Shift of Tails (SOT) against historical quantiles.
  - Unit tests verifying EFI is bounded in $[-1, 1]$ and detects simulated high tails.
- **Task 1.2: Mesh GNN Architecture**
  - Implement PyTorch Geometric / message-passing GNN over the M9 icosahedral graph.
  - Predict anomaly probability and spatio-temporal cluster centroids across lead times ($T+24\text{h}$ to $T+120\text{h}$).
- **Task 1.3: 4D Bounding Box & Uncertainty Cone Extraction**
  - Extract dynamic 4D bounding box `[lat, lon, level, time]` around detected anomaly cluster.
  - Compute cross-ensemble dispersion cone representing track uncertainty.
- **Task 1.4: Track Overlap Validation & Visualization**
  - Plot tracker output bounding box and uncertainty cone overlaid on Cyclone Phailin's known IMD/JTWC track.
  - *Exit Criterion:* `tests/artifacts/milestone1_tracker_cone.png` showing visual overlap with ground truth track, plus passing pytest.

### Milestone 2 — Downscaler (Stage 2)
- **Task 2.1: Tracker Crop & Static Topography Conditioning**
  - Crop 12 km fields using Stage 1 bounding box.
  - Align and normalize 5 km SRTM/terrain elevation and land-use rasters as conditioning channels.
- **Task 2.2: U-Net Base Regression Mean Model**
  - Implement 2D U-Net mapping 12 km crop + terrain/land-use to 5 km deterministic mean field.
- **Task 2.3: Residual Diffusion Model & Sampler**
  - Adapt HuggingFace Diffusers DDPM/DDIM pipeline to learn high-frequency residual departures: $x_{\text{res}} = y_{5\text{km}} - \mu_{5\text{km}}$.
  - Implement 16-member stochastic sampling.
- **Task 2.4: Physics-Informed Composite Loss**
  - Implement diffusion loss $\mathcal{L}_{\text{diff}} + \lambda_1 \text{CRPS}_{\text{tail}} + \lambda_2 \mathcal{L}_{\text{PSD}}$.
  - Implement mathematical stubs with full formulas and physical explanations for $\mathcal{L}_{\text{VIMFC}}$ and $\mathcal{L}_{\text{mass}}$.
- **Task 2.5: Downscaler Comparative Evaluation**
  - Collapse 16 samples to $P(\text{rain} > \text{threshold})$.
  - Compute Power Spectral Density (PSD) curves comparing raw 12 km, bilinear upsample, ChakraNet diffusion, and IMERG ground truth.
  - *Exit Criterion:* Side-by-side comparison plot and PSD curve saved to `tests/artifacts/milestone2_downscaler_comparison.png` and `tests/artifacts/milestone2_psd_analysis.png`.

### Milestone 3 — Metrics & Honesty Pass
- **Task 3.1: Quantitative Metric Calculation**
  - Calculate PSD high-frequency retention score, P99 rainfall bias (mm/day), Continuous Ranked Probability Score (CRPS) vs. bilinear baseline, and Brier score.
  - Profile wall-clock latency (Stage 1 inference, Stage 2 sampling) with explicit hardware specs.
- **Task 3.2: Metrics JSON & Honesty Table Generation**
  - Export all metrics to `metrics.json` annotated with metadata ($N=1$ event, hardware spec, sample size).
  - Generate markdown KPI summary table comparing target KPIs vs. measured prototype results.
  - *Exit Criterion:* `metrics.json` and verification report in `tests/artifacts/milestone3_kpi_table.md`.

### Milestone 4 — Serving Layer & CAP 1.2 Alerting
- **Task 4.1: Database & GeoJSON Storage Setup**
  - Create SQLite/GeoJSON database schema storing tracked events, cone geometries, 5 km hazard grids, and alert records.
- **Task 4.2: FastAPI Application & REST Endpoints**
  - Implement endpoints:
    - `GET /events`
    - `GET /events/{id}/track`
    - `GET /events/{id}/hazard-map`
    - `GET /events/{id}/alerts`
    - `POST /events/{id}/alerts/dispatch` (mocked, logged, labeled SIMULATED).
- **Task 4.3: CAP 1.2 XML Alert Engine & Schema Validation**
  - Generate compliant OASIS CAP 1.2 XML with 4-tier severity levels (Minor, Moderate, Severe, Extreme) mapped from exceedance probabilities.
  - Validate output against official CAP 1.2 XSD schema.
- **Task 4.4: API Integration Tests & Payload Capture**
  - Pytest suite covering all endpoints, CORS, error handling, and XML validation.
  - *Exit Criterion:* All pytests passing; captured JSON/XML request/responses saved to `tests/artifacts/api_examples/`.

### Milestone 5 — Frontend & Demo Assembly
- **Task 5.1: MapLibre GL JS Dashboard Architecture**
  - Create modern, responsive HTML5/Vanilla JS/CSS dashboard with glassmorphism, dark mode, and sidebar controls.
  - Load MapLibre GL map with cartographic base layers (India boundary, coastal districts).
- **Task 5.2: Multi-Layer Visualization**
  - Toggle layers: (a) Raw 12 km NEPS-G field, (b) Tracked anomaly bounding box & uncertainty cone, (c) 5 km downscaled hazard probability map.
  - Add interactive time slider across lead times ($T+0\text{h}$ to $T+120\text{h}$).
- **Task 5.3: Cell Inspection & Multilingual CAP Alert Modal**
  - Click on 5 km cell to trigger village/block-level inspection popup.
  - Display CAP alert metadata in English and Hindi (templated translation, badged `SIMULATED Bhashini`).
  - Add "Simulate SACHET Dispatch" button with visual confirmation modal and simulated audit log.
- **Task 5.4: Automated Demo Verification**
  - Browser testing and screenshot capture of interactive workflows.
  - *Exit Criterion:* Step-by-step screenshots saved to `tests/artifacts/milestone5_dashboard_*.png`.

### Milestone 6 — README & Honest Limitations Doc
- **Task 6.1: Master Documentation & Architectural Blueprints**
  - Comprehensive `README.md` containing problem statement, system architecture, mathematical formulas, and reproduction commands.
  - Clearly delineate Real (data, models, metrics) vs. Simulated (SACHET/Bhashini dispatch) vs. Future Work (VIMFC/mass conservation loss terms).
- **Task 6.2: Traceability Audit**
  - Audit every reported number in documentation against `metrics.json` and `tests/artifacts/`.
  - *Exit Criterion:* Verification check passes with zero unattributed or fabricated claims.

---

## 4. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Large model weights or compute constraints on local environment | High | Implement dual execution paths: lightweight CPU-executable models/checkpoints for local unit tests and automated verification + standalone Google Colab / Kaggle notebooks for full GPU fine-tuning/inference. |
| Incomplete real-time NEPS-G access in hackathon sandbox | Med | Use open ECMWF operational / ERA5 / IMDAA historical archive data for Cyclone Phailin (2013) as ground truth. |
| Complex C-dependencies (GDAL, SpatiaLite, PostGIS) on Windows | Med | Use pure-Python geospatial parsing (`shapely`, `pyproj`, `geopandas`, `sqlite3`) with GeoJSON interchange format, avoiding heavy native binaries. |
| Numerical instability in composite physics loss | Med | Fully implement stable $\mathcal{L}_{\text{diff}} + \text{CRPS}_{\text{tail}} + \mathcal{L}_{\text{PSD}}$; document VIMFC/mass terms as rigorous mathematical formulas with transparent TODO stubs. |

---

---

## 5. Completed Work & Goal Audit (4D Map & 5 km Geodesic Radius Accuracy)

### Verified Enhancements:
1. **Accurate 5 km Spatial Geodesic Mapping**:
   - Implemented exact WGS-84 geodesic circle formula (`db.py::generate_geodesic_circle`) calculating great-circle radius of exactly 5.0 km ($78.54\text{ km}^2$ impact area).
   - Applied latitude-corrected longitudinal width `dlon_5km = 5.0 / (111.139 * cos(lat))` ensuring exact $5.0\text{ km} \times 5.0\text{ km}$ ($25.0\text{ km}^2$) grid cells without high-latitude distortion.
   - Distinctly labeled and reported both the physical cell grid dimensions ($25\text{ km}^2$, in-radius $2.5\text{ km}$) and the emergency alert perimeter ($78.54\text{ km}^2$, geodesic radius $5.0\text{ km}$).
2. **Interactive 4D Volumetric Spatiotemporal Visualization**:
   - Integrated 4D camera angle with 58° pitch and -14° oblique bearing in MapLibre GL JS.
   - Added volumetric 3D column extrusion (`layer-hazard-extrusion-4d`) mapping cell probability and rainfall intensity to vertical convective storm columns up to 14,000 m.
   - Implemented 4D spatiotemporal animation across lead times $T+0\text{h}$ to $T+120\text{h}$ with smooth camera and layer sync.
   - Added selected cell 3D extrusion buffer cylinder and 5 km geodesic warning perimeter rings (`layer-radius-5km-rings`).
3. **Comprehensive End-to-End Verification**:
   - `pytest` (45/45 passed in 8.50s).
   - Playwright E2E browser suite (`tests/verify_visual_and_layout.js`, 8/8 passed):
     - Zero console errors.
     - Strict flat solid styling (zero linear gradients).
     - 1280×720 viewport responsiveness.
     - 4D volumetric mode switching.
     - Accurate 5.0 km geodesic radius card audit and multilingual CAP 1.2 modal inspection.
   - Verified screenshots archived in [`tests/artifacts/`](file:///c:/SIH2026-78/ChakraNet/tests/artifacts/).
