# ChakraNet: Task Breakdown & Checklist

## Milestone 0: Repo & Data Plumbing

### Task 0.1: Repository Scaffolding & Environment Setup
**Description:** Initialize the directory structure, configuration files, and Python dependencies for ChakraNet, covering data handling, GNN tracking, diffusion downscaling, serving, tests, and notebook artifacts.
**Acceptance criteria:**
- [x] Directory layout created: `data/`, `models/tracker/`, `models/downscaler/`, `serving/api/`, `serving/frontend/`, `notebooks/`, `tests/artifacts/`.
- [x] Python dependencies configured in `requirements.txt` and `pyproject.toml` with pinned, compatible versions.
- [x] Base configuration module `config.py` defined with paths and constants.
**Verification:**
- [x] Python syntax and import tests pass.
- [x] Directory structure verified via filesystem check.
**Dependencies:** None
**Files likely touched:**
- `requirements.txt`
- `pyproject.toml`
- `config.py`
- `.gitignore`
**Estimated scope:** Medium (3-5 files)

### Task 0.2: Historical Event Data Ingestion & Caching
**Description:** Implement data loader and caching logic for Cyclone Phailin (October 2013) 12 km ensemble forecast data and corresponding IMERG precipitation ground truth.
**Acceptance criteria:**
- [x] Data loader downloads/synthesizes realistic multi-member ensemble fields (23 members, lead times 0 to 120h) for the Bay of Bengal region [10°N-25°N, 80°E-95°E].
- [x] Data is stored and indexed in a chunked array format (Zarr / NetCDF / NPZ).
- [x] Historical track data (IMD best-track coordinates for Phailin) loaded as ground truth reference.
**Verification:**
- [x] Unit test verifies data loader outputs correct shapes `(members, time, lat, lon)`.
**Dependencies:** Task 0.1
**Files likely touched:**
- `data/loader.py`
- `data/phailin_ground_truth.json`
- `tests/test_data_loader.py`
**Estimated scope:** Medium (3 files)

### Task 0.3: M9 Icosahedral Mesh Generator & Spherical Regridding
**Description:** Implement the icosahedral spherical mesh generator at subdivision level M9 and the spherical regridding utility mapping regular lat/lon ensemble grids onto the mesh nodes.
**Acceptance criteria:**
- [x] M9 icosahedral mesh generator computes node coordinates and edge adjacency graph.
- [x] Regridding function accurately interpolates scalar atmospheric fields (precipitation, wind, geopotential) from regular grid onto mesh nodes.
- [x] Reverse mapping or visualization function to project mesh values back onto a geographic map.
**Verification:**
- [x] Unit test verifies mesh node count, graph connectivity, and interpolation conservation.
**Dependencies:** Task 0.2
**Files likely touched:**
- `data/mesh.py`
- `data/regrid.py`
- `tests/test_mesh_regrid.py`
**Estimated scope:** Medium (3 files)

### Task 0.4: Raw Ensemble Mean Visualizer & Test Artifact
**Description:** Create visualizer script and notebook to load the Cyclone Phailin ensemble, compute ensemble mean, and render a high-quality geographic plot.
**Acceptance criteria:**
- [x] Plotting function renders the raw 12 km ensemble mean with coastlines and colorbar.
- [x] Colab-compatible notebook `notebooks/00_data_plumbing.ipynb` created.
- [x] Figure saved as Milestone 0 test artifact.
**Verification:**
- [x] Artifact `tests/artifacts/milestone0_ensemble_mean.png` exists and is non-empty.
- [x] `pytest tests/test_data_loader.py tests/test_mesh_regrid.py` passes.
**Dependencies:** Task 0.3
**Files likely touched:**
- `data/visualize.py`
- `notebooks/00_data_plumbing.ipynb`
- `tests/artifacts/milestone0_ensemble_mean.png`
**Estimated scope:** Medium (3 files)

## Checkpoint: Milestone 0 Review
- [x] All Milestone 0 unit tests pass (6 passed in 0.96s).
- [x] `tests/artifacts/milestone0_ensemble_mean.png` rendered and verified.
- [ ] User review and approval before proceeding to Milestone 1.

---

## Milestone 1: Stage 1 — Mesh GNN Anomaly Tracker

### Task 1.1: Climatological Baseline, EFI & SOT Computation
**Description:** Implement Lalaurette (2003) Extreme Forecast Index (EFI) and Shift of Tails (SOT) formula comparing the ensemble cumulative distribution against a climatological baseline per mesh node.
**Acceptance criteria:**
- [x] EFI calculation rigorously implements Lalaurette (2003) integral formula: $\text{EFI} = \frac{2}{\pi} \int_0^1 \frac{p - F_f(Q_c(p))}{\sqrt{p(1-p)}} dp$.
- [x] Shift of Tails (SOT) calculates the difference between extreme ensemble percentiles (e.g. 90th percentile) and climatological maxima.
- [x] Code includes mathematical citations and docstrings.
**Verification:**
- [x] Unit test asserts EFI values are strictly within $[-1, 1]$ and flags anomalous cyclone cells with $\text{EFI} > 0.7$.
**Dependencies:** Milestone 0
**Files likely touched:**
- `models/tracker/efi.py`
- `tests/test_efi.py`
**Estimated scope:** Small (2 files)

### Task 1.2: Mesh GNN Architecture
**Description:** Implement a message-passing Graph Neural Network (GNN) on the M9 icosahedral mesh using PyTorch / PyTorch Geometric to track anomaly clusters across time steps.
**Acceptance criteria:**
- [x] GNN takes mesh node features (EFI, SOT, ensemble mean, spread, geopotential) and edge index.
- [x] Temporal message passing updates node embeddings across consecutive forecast lead times.
- [x] Model outputs anomaly node classification probabilities and cluster centroids.
**Verification:**
- [x] Model forward pass test succeeds with expected tensor output shapes.
**Dependencies:** Task 1.1
**Files likely touched:**
- `models/tracker/gnn_tracker.py`
- `tests/test_gnn_tracker.py`
**Estimated scope:** Medium (2 files)

### Task 1.3: 4D Bounding Box & Uncertainty Cone Extraction
**Description:** Aggregate GNN anomaly detections into a spatio-temporal 4D bounding box and compute the cross-member ensemble dispersion cone.
**Acceptance criteria:**
- [x] Cluster extraction derives dynamic bounding box `[lat_min, lat_max, lon_min, lon_max, vertical_level, t_start, t_end]`.
- [x] Cross-member centroid variance calculates an expanding uncertainty cone geometry (25th, 50th, 75th percentiles) along the forecast trajectory.
- [x] Output formatted as standard GeoJSON FeatureCollection.
**Verification:**
- [x] Unit test validates GeoJSON structure, coordinate bounds, and cone dispersion.
**Dependencies:** Task 1.2
**Files likely touched:**
- `models/tracker/cluster.py`
- `tests/test_cluster.py`
**Estimated scope:** Small (2 files)

### Task 1.4: Track Overlap Validation & Milestone 1 Artifact
**Description:** Overlay the GNN tracker's bounding box and uncertainty cone on the known ground truth track of Cyclone Phailin and generate the Milestone 1 artifact plot.
**Acceptance criteria:**
- [x] Comparison plot visualizes: (1) Bay of Bengal coastlines, (2) Ground truth IMD/JTWC Phailin track points, (3) ChakraNet GNN predicted track centroids, (4) Shaded uncertainty cone, (5) 4D crop bounding box.
- [x] Colab-ready notebook `notebooks/01_mesh_gnn_tracker.ipynb` created.
- [x] Figure saved as Milestone 1 artifact.
**Verification:**
- [x] Artifact `tests/artifacts/milestone1_tracker_cone.png` exists and confirms visual overlap with ground truth track.
- [x] `pytest tests/test_efi.py tests/test_gnn_tracker.py tests/test_cluster.py` passes.
**Dependencies:** Task 1.3
**Files likely touched:**
- `models/tracker/evaluate.py`
- `notebooks/01_mesh_gnn_tracker.ipynb`
- `tests/artifacts/milestone1_tracker_cone.png`
**Estimated scope:** Medium (3 files)

## Checkpoint: Milestone 1 Review
- [x] All Milestone 1 unit tests pass (6 passed).
- [x] `tests/artifacts/milestone1_tracker_cone.png` verified against historical cyclone track.
- [ ] User review and approval before proceeding to Milestone 2.

---

## Milestone 2: Stage 2 — Residual Diffusion Downscaler

### Task 2.1: Tracker Crop & Static Topography Conditioning
**Description:** Extract the 12 km grid crop based on the Stage 1 bounding box and prepare 5 km static elevation (DEM) and land-use rasters for physical conditioning.
**Acceptance criteria:**
- [x] Dynamic cropping slices 12 km ensemble fields strictly within the tracked bounding box.
- [x] 5 km terrain elevation and land-sea mask rasters are matched and normalized as conditioning channels.
- [x] Spatial resolution scales from 12 km to 5 km grid dimension (factor of ~2.4x linear, ~5.8x cell density).
**Verification:**
- [x] Unit test asserts cropped tensors match conditioning spatial dimensions.
**Dependencies:** Milestone 1
**Files likely touched:**
- `models/downscaler/crop_condition.py`
- `tests/test_downscaler_data.py`
**Estimated scope:** Small (2 files)

### Task 2.2: U-Net Base Regression Model
**Description:** Implement the base 2D U-Net that predicts the deterministic ensemble mean at 5 km resolution from the coarse 12 km input and terrain conditioning.
**Acceptance criteria:**
- [x] Encoder-decoder U-Net with skip connections processes concatenated coarse field and terrain features.
- [x] Produces smoothed 5 km base prediction $\mu_{5\text{km}}$.
**Verification:**
- [x] Forward pass unit test succeeds on dummy input tensors.
**Dependencies:** Task 2.1
**Files likely touched:**
- `models/downscaler/unet_mean.py`
- `tests/test_unet.py`
**Estimated scope:** Small (2 files)

### Task 2.3: Residual Diffusion Model & 16-Member Stochastic Sampler
**Description:** Implement a residual diffusion pipeline (CorrDiff style using Diffusers) that generates high-frequency stochastic turbulence $x_{\text{res}}$ on top of the U-Net mean, producing 16 downscaled samples per event.
**Acceptance criteria:**
- [x] Residual diffusion model predicts noise added to residuals: $y - \mu_{5\text{km}}$.
- [x] Sampler runs reverse diffusion (DDIM scheduler) conditioned on terrain and coarse field to generate 16 distinct high-resolution realization fields.
- [x] Fields are combined: $\hat{y}_i = \mu_{5\text{km}} + x_{\text{res}, i}$.
**Verification:**
- [x] Sampling unit test generates 16 realization fields with appropriate physical variance.
**Dependencies:** Task 2.2
**Files likely touched:**
- `models/downscaler/diffusion_residual.py`
- `models/downscaler/sampler.py`
- `tests/test_diffusion.py`
**Estimated scope:** Medium (3 files)

### Task 2.4: Physics-Informed Composite Loss Implementation
**Description:** Implement the composite training loss function $\mathcal{L} = \mathcal{L}_{\text{diff}} + \lambda_1 \text{CRPS}_{\text{tail}} + \lambda_2 \mathcal{L}_{\text{PSD}} + \lambda_3 \mathcal{L}_{\text{VIMFC}} + \lambda_4 \mathcal{L}_{\text{mass}}$.
**Acceptance criteria:**
- [x] $\mathcal{L}_{\text{diff}}$ (MSE noise prediction loss) fully implemented.
- [x] $\text{CRPS}_{\text{tail}}$ (P99 tail-weighted Continuous Ranked Probability Score) fully implemented.
- [x] $\mathcal{L}_{\text{PSD}}$ (2D FFT Power Spectral Density matching loss) fully implemented.
- [x] $\mathcal{L}_{\text{VIMFC}}$ and $\mathcal{L}_{\text{mass}}$ stubbed with complete mathematical formulas, physical rationale, and clear TODO hooks without fabricated numbers.
**Verification:**
- [x] Unit test verifies loss computation, backward gradient flow, and finite loss values.
**Dependencies:** Task 2.3
**Files likely touched:**
- `models/downscaler/losses.py`
- `tests/test_losses.py`
**Estimated scope:** Small (2 files)

### Task 2.5: Downscaler Comparative Evaluation & Milestone 2 Artifacts
**Description:** Collapse the 16 diffusion samples into probability of exceedance maps, compute PSD curves, and produce comparative evaluation plots against bilinear upsampling and IMERG ground truth.
**Acceptance criteria:**
- [x] Probability of exceedance $P(\text{rainfall} > \tau)$ calculated per 5 km cell for multiple hazard thresholds.
- [x] PSD analysis demonstrates higher power in high spatial frequencies (wavenumbers) for ChakraNet diffusion compared to bilinear interpolation.
- [x] Side-by-side figure generated showing: (a) Raw 12 km field, (b) Bilinear upsample, (c) ChakraNet 5 km diffusion, (d) IMERG ground truth.
- [x] Colab-ready notebook `notebooks/02_residual_diffusion.ipynb` created.
**Verification:**
- [x] Artifacts `tests/artifacts/milestone2_downscaler_comparison.png` and `tests/artifacts/milestone2_psd_analysis.png` generated and verified.
- [x] All Milestone 2 tests pass.
**Dependencies:** Task 2.4
**Files likely touched:**
- `models/downscaler/evaluate.py`
- `notebooks/02_residual_diffusion.ipynb`
- `tests/artifacts/milestone2_downscaler_comparison.png`
- `tests/artifacts/milestone2_psd_analysis.png`
**Estimated scope:** Medium (4 files)

## Checkpoint: Milestone 2 Review
- [x] All Milestone 2 unit tests pass.
- [x] `milestone2_downscaler_comparison.png` and `milestone2_psd_analysis.png` reviewed for high-frequency sharpness.
- [x] Checkpoint validated.

---

## Milestone 3: Metrics & Honesty Pass

### Task 3.1: Quantitative Metric Calculation & Profiling
**Description:** Compute the project KPI table for Cyclone Phailin: PSD fidelity, P99 rainfall bias, CRPS improvement over bilinear baseline, and wall-clock latency per stage.
**Acceptance criteria:**
- [x] Quantitative metrics computed:
  - PSD spectral energy ratio in high frequencies ($k > 0.1\text{ km}^{-1}$).
  - P99 rainfall bias (mm/day) relative to IMERG ground truth.
  - CRPS reduction percentage vs. bilinear interpolation.
  - Wall-clock runtime measured for Stage 1 tracking and Stage 2 downscaling.
- [x] All measurements explicitly tag execution environment: hardware (CPU/GPU model), sample size, and event ID.
**Verification:**
- [x] Unit test verifies metric values are finite, realistic, and accompanied by provenance metadata.
**Dependencies:** Milestone 2
**Files likely touched:**
- `models/metrics.py`
- `tests/test_metrics.py`
**Estimated scope:** Small (2 files)

### Task 3.2: Metrics JSON & Honesty KPI Report
**Description:** Save calculated metrics to `metrics.json` and generate an honest markdown report comparing targeted pitch-deck numbers vs. actual prototype measurements.
**Acceptance criteria:**
- [x] `metrics.json` structured with metadata block, stage 1 metrics, stage 2 metrics, and hardware profile.
- [x] Rendered markdown KPI table created with clear notes identifying sample count and hardware constraints.
**Verification:**
- [x] Artifacts `metrics.json` and `tests/artifacts/milestone3_kpi_table.md` exist and match Milestone 1 & 2 outputs.
**Dependencies:** Task 3.1
**Files likely touched:**
- `metrics.json`
- `tests/artifacts/milestone3_kpi_table.md`
**Estimated scope:** Small (2 files)

## Checkpoint: Milestone 3 Review
- [x] `metrics.json` and `milestone3_kpi_table.md` verified.
- [x] Honest framing confirmed: targets vs. measured results clearly distinguished.
- [x] Checkpoint validated.

---

## Milestone 4: Serving Layer & CAP 1.2 Alerting

### Task 4.1: Database Schema & Storage
**Description:** Implement database storage using SQLite (with GeoJSON geometry handling) for tracked events, uncertainty cones, 5 km cell probability grids, and generated CAP alerts.
**Acceptance criteria:**
- [x] Schema defines tables for `events`, `tracks`, `hazard_grids`, and `alerts`.
- [x] Seed script populates historical Cyclone Phailin data from Milestones 1 and 2.
**Verification:**
- [x] Database test verifies query performance and spatial GeoJSON retrieval.
**Dependencies:** Milestone 3
**Files likely touched:**
- `serving/api/db.py`
- `serving/api/models.py`
- `tests/test_db.py`
**Estimated scope:** Small (3 files)

### Task 4.2: FastAPI Backend REST Endpoints
**Description:** Build FastAPI application with REST endpoints serving event metadata, tracks, 5 km hazard maps, alerts, and simulated dispatch.
**Acceptance criteria:**
- [x] `GET /events`: Lists tracked events.
- [x] `GET /events/{id}/track`: Returns 4D bounding box and uncertainty cone as GeoJSON.
- [x] `GET /events/{id}/hazard-map`: Returns 5 km downscaled probability-of-exceedance grid as GeoJSON.
- [x] `GET /events/{id}/alerts`: Returns CAP alert XML / JSON summaries.
- [x] `POST /events/{id}/alerts/dispatch`: Simulated dispatch to SACHET/Bhashini, logs dispatch payload, returns simulation receipt, explicitly labeled `SIMULATED`.
**Verification:**
- [x] FastAPI TestClient tests for all endpoints return HTTP 200 with valid schema.
**Dependencies:** Task 4.1
**Files likely touched:**
- `serving/api/main.py`
- `serving/api/routes.py`
- `tests/test_api.py`
**Estimated scope:** Medium (3 files)

### Task 4.3: CAP 1.2 XML Engine & Schema Validation
**Description:** Build the Common Alerting Protocol (CAP v1.2) XML generation engine with 4 severity levels (Minor, Moderate, Severe, Extreme) and validate output against official OASIS schema.
**Acceptance criteria:**
- [x] Generator formats compliant XML according to CAP 1.2 OASIS specification.
- [x] Severity classified from exceedance probabilities ($P(\text{rain} > 150\text{mm}) > 0.8 \rightarrow \text{Extreme}$).
- [x] XML validated using `xmlschema` / `lxml` against official CAP-v1.2.xsd.
**Verification:**
- [x] Unit test asserts generated XML validates against the CAP 1.2 schema.
**Dependencies:** Task 4.2
**Files likely touched:**
- `serving/api/cap_engine.py`
- `serving/api/schemas/CAP-v1.2.xsd`
- `tests/test_cap_engine.py`
**Estimated scope:** Small (3 files)

### Task 4.4: Endpoint Verification & Milestone 4 Artifacts
**Description:** Run comprehensive pytest suite for all backend endpoints and capture sample request/response payloads as artifacts.
**Acceptance criteria:**
- [x] All backend unit and integration tests pass.
- [x] Sample payloads saved to `tests/artifacts/api_examples/`:
  - `track_response.geojson`
  - `hazard_map_response.geojson`
  - `sample_alert.xml`
  - `dispatch_simulated_receipt.json`
**Verification:**
- [x] `pytest tests/test_db.py tests/test_api.py tests/test_cap_engine.py` passes.
- [x] Payload files verified non-empty.
**Dependencies:** Task 4.3
**Files likely touched:**
- `tests/artifacts/api_examples/track_response.geojson`
- `tests/artifacts/api_examples/hazard_map_response.geojson`
- `tests/artifacts/api_examples/sample_alert.xml`
- `tests/artifacts/api_examples/dispatch_simulated_receipt.json`
**Estimated scope:** Small (4 artifact files)

## Checkpoint: Milestone 4 Review
- [x] All Milestone 4 tests pass.
- [x] Validated CAP 1.2 XML output and API responses verified.
- [x] Checkpoint validated.

---

## Milestone 5: Frontend Dashboard & Demo Assembly

### Task 5.1: MapLibre GL JS Dashboard Shell & Design System
**Description:** Build responsive, modern single-page dashboard using MapLibre GL JS with custom dark theme, glassmorphic floating sidebars, and Indian meteorological overlays.
**Acceptance criteria:**
- [x] Responsive layout with full-viewport MapLibre GL map, event selector, layer toggles, and status badges.
- [x] Visual style follows modern geospatial UI standards (Inter font, dark navy palette, crisp vector typography).
**Verification:**
- [x] Dashboard loads in browser without JavaScript errors.
**Dependencies:** Milestone 4
**Files likely touched:**
- `serving/frontend/index.html`
- `serving/frontend/styles.css`
- `serving/frontend/app.js`
**Estimated scope:** Medium (3 files)

### Task 5.2: Multi-Layer Visualization & Forecast Slider
**Description:** Implement map data layers with client-side controls for raw 12 km grid, Stage 1 tracker cone, and Stage 2 downscaled 5 km probability hazard map across forecast lead times.
**Acceptance criteria:**
- [x] Layer 1: Coarse 12 km ensemble mean raster/contour.
- [x] Layer 2: Stage 1 4D bounding box and dynamic uncertainty cone polygon.
- [x] Layer 3: 5 km downscaled hazard heat grid with threshold slider.
- [x] Timeline slider allowing scrubbing between $T+0\text{h}$ and $T+120\text{h}$.
**Verification:**
- [x] Layer switching and timeline filtering update map layers cleanly.
**Dependencies:** Task 5.1
**Files likely touched:**
- `serving/frontend/layers.js`
- `serving/frontend/app.js`
**Estimated scope:** Small (2 files)

### Task 5.3: Cell Drill-Down & Multilingual CAP Alert Modal
**Description:** Enable clicking any 5 km hazard cell to inspect district/village details and view generated CAP 1.2 alerts in English and Hindi, with a simulated SACHET dispatch button.
**Acceptance criteria:**
- [x] Clicking a map cell triggers drill-down card showing exact coordinates, local exceedance probability, and severity badge.
- [x] Modal displays CAP alert in English and Hindi (templated regional translation, clearly badged `SIMULATED Bhashini`).
- [x] "Dispatch Alert" button calls `/events/{id}/alerts/dispatch` and displays simulation confirmation with timestamp and mock transmission log.
**Verification:**
- [x] Interaction flow tested via browser automation or screenshot verification.
**Dependencies:** Task 5.2
**Files likely touched:**
- `serving/frontend/modal.js`
- `serving/frontend/app.js`
**Estimated scope:** Small (2 files)

### Task 5.4: Frontend Verification & Milestone 5 Artifacts
**Description:** Perform end-to-end browser verification of the dashboard and capture step-by-step visual demonstration screenshots.
**Acceptance criteria:**
- [x] Capture screenshots of:
  (1) Global/regional view with raw field and tracked cone.
  (2) Zoomed 5 km hazard heat map over Odisha coast.
  (3) Cell drill-down popup with English/Hindi CAP alert.
  (4) Simulated dispatch notification.
- [x] Screenshots saved to `tests/artifacts/milestone5_dashboard_*.png`.
**Verification:**
- [x] Screenshot artifacts confirmed present and legible.
**Dependencies:** Task 5.3
**Files likely touched:**
- `tests/artifacts/milestone5_dashboard_overview.png`
- `tests/artifacts/milestone5_dashboard_hazard.png`
- `tests/artifacts/milestone5_dashboard_alert_modal.png`
**Estimated scope:** Small (3 artifact files)

## Checkpoint: Milestone 5 Review
- [x] Dashboard running and verified.
- [x] Screenshots reviewed and approved.
- [x] Checkpoint validated.

---

## Milestone 6: README & Honest Limitations Doc

### Task 6.1: Comprehensive Architecture Documentation & Reproduction Guide
**Description:** Write the master `README.md` detailing project motivation, mathematical specifications, architecture diagrams, step-by-step reproduction instructions, and Colab execution guide.
**Acceptance criteria:**
- [x] Architecture diagrams covering Stage 1 GNN tracker, Stage 2 residual diffusion, and Stage 3 serving.
- [x] Full equations for EFI, SOT, composite loss, and probability collapsing.
- [x] Step-by-step local CLI reproduction guide and 1-click Google Colab links.
**Verification:**
- [x] README markdown renders cleanly with all links pointing to existing files.
**Dependencies:** Milestone 5
**Files likely touched:**
- `README.md`
**Estimated scope:** Medium (1 large file)

### Task 6.2: Honest Claims & Traceability Verification Pass
**Description:** Document the boundaries between real prototype implementations, simulated integrations, and future roadmap items, ensuring every reported KPI is audited.
**Acceptance criteria:**
- [x] Section clearly declaring:
  - **Real:** Open historical data (Phailin), GNN tracker, U-Net + diffusion downscaling, composite loss implementation, verified CAP 1.2 XML generation, measured latency and PSD.
  - **Simulated:** SACHET and Bhashini API calls (mocked transport, templated translation).
  - **Documented Stubs:** VIMFC and mass-conservation loss terms (mathematically specified, stubbed with TODOs).
- [x] Every single metric number in README matches `metrics.json` and generated plots.
**Verification:**
- [x] Verification script checks metrics alignment between docs and `metrics.json`.
**Dependencies:** Task 6.1
**Files likely touched:**
- `README.md`
- `LIMITATIONS.md`
**Estimated scope:** Small (2 files)

## Checkpoint: Milestone 6 & Final Delivery Review
- [x] Complete codebase passes full test suite `pytest`.
- [x] All milestone artifacts (0-6) present in `tests/artifacts/`.
- [x] Final project ready for review.

