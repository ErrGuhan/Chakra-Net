# ChakraNet — Build Prompt for Google Antigravity

Copy everything in the code block below into Antigravity's **Manager surface** as a single mission. It is written to match how Antigravity actually works: it states the goal, hands over a milestone checklist it can turn into Artifacts, names the exact stack, and tells it what "done" looks like for each milestone so its self-verification loop (tests, screenshots, run logs) has something concrete to check against. Run the milestones **sequentially, one agent per milestone**, and review each Artifact/plan before letting it execute, per the guidance below the prompt.

---

## How to use this

1. Open Antigravity → Manager view → New Mission.
2. Paste the full prompt block.
3. When Antigravity proposes its plan (Artifact), check that it has split the work into the same 6 milestones below before approving execution — if it collapses them into one giant task, tell it explicitly: "Keep these as 6 separate milestones with a checkpoint after each."
4. Approve Milestone 1, let it run to green, review the Artifact (test results / screenshots), then approve Milestone 2, and so on. Don't approve all 6 at once — the physics/model milestones (2 and 3) need a human sanity check before the serving layer is built on top of them.
5. If you have GPU access via Colab/Kaggle rather than locally, tell Antigravity in Milestone 0 to generate notebook-compatible scripts (`.ipynb` or `%%writefile`-friendly `.py`) in addition to the CLI scripts, since its own sandboxed terminal likely has no GPU.

---

```
MISSION: Build ChakraNet — a working prototype of a two-stage AI pipeline that
tracks extreme-weather anomalies in a 12 km medium-range ensemble forecast and
downscales the tracked region to ~5 km probabilistic hazard maps, then emits
CAP-format alerts. This is a Smart India Hackathon 2026 prototype (PS 26078,
NCMRWF/MoES), not a production system, so scope everything to run end-to-end
on ONE fully-worked historical case study first, on free-tier compute (a
single consumer/Colab-class GPU, or CPU fallback for non-model steps), before
attempting to generalize.

Ground truth for what to build (do not deviate from this architecture):

STAGE 1 — Mesh GNN tracker
  - Input: a 12 km, multi-member ensemble forecast (23 members if available;
    fewer is fine for MVP), 3-10 day lead time.
  - Regrid the lat/lon ensemble onto an icosahedral mesh (resolution level M9;
    do not use M6, that is ~110 km and too coarse).
  - Compute per-node Extreme Forecast Index (EFI) and Shift of Tails (SOT)
    against a climatological/re-forecast baseline (Lalaurette 2003 definition
    — implement or approximate this formula explicitly, cite it in a code
    comment).
  - Run a graph neural network (Graph Attention or message-passing GNN, built
    with PyTorch Geometric or DGL) over the mesh to detect and track anomaly
    clusters through time, warm-started from open GraphCast weights if
    feasible; if warm-starting is impractical in the time available,
    implement the GNN architecture faithfully and train briefly on the
    available re-forecast data, and say so plainly in the README rather than
    silently skipping it.
  - Output: a 4D bounding box (lat, lon, vertical level, time) tracking the
    anomaly, plus an uncertainty cone across ensemble members.

STAGE 2 — Residual diffusion downscaler
  - Input: the Stage 1 crop only (NOT the full global grid — this is what
    keeps inference cheap; downscaling the full grid is out of scope).
  - Architecture: regression mean via a U-Net, then a residual diffusion
    model (adapt an open Diffusers pipeline in the style of NVIDIA's CorrDiff)
    to add high-frequency detail, conditioned on 5 km static terrain and
    land-use rasters.
  - Loss function: implement as a weighted sum —
      L = L_diffusion + lambda1 * CRPS_tail(P99-weighted)
        + lambda2 * PSD_loss (power spectral density matching)
        + lambda3 * VIMFC_consistency (vertically integrated moisture flux
          convergence)
        + lambda4 * mass_conservation_penalty
    It is acceptable for an MVP to implement L_diffusion + CRPS_tail + PSD
    fully and stub VIMFC/mass terms as documented TODOs with the physical
    justification written in comments — do not fake numbers for terms you
    have not implemented.
  - Sample 16 fields per event from the diffusion model; collapse the
    ensemble of downscaled fields into a per-5km-cell probability-of-exceedance
    map (e.g., P(rainfall > threshold)).
  - Output: ~5 km resolution probability maps + a 4-level CAP severity
    classification per cell.

STAGE 3 — Serving & alerting layer
  - Backend: FastAPI. Expose REST endpoints:
      GET  /events                     -> list tracked anomaly events
      GET  /events/{id}/track           -> tracker output (4D box + cone) as GeoJSON
      GET  /events/{id}/hazard-map       -> downscaled P(exceed) grid as GeoJSON/raster tiles
      GET  /events/{id}/alerts           -> generated CAP 1.2 XML alerts
      POST /events/{id}/alerts/dispatch  -> MOCKED dispatch to SACHET/Bhashini
                                             (log the payload, do not call a
                                             real external API; clearly label
                                             this as simulated in both code
                                             and UI)
  - Storage: PostGIS (or SQLite+SpatiaLite if PostGIS is too heavy for the
    sandbox) for cell-level probability and alert records.
  - Frontend: a MapLibre GL JS map, district-to-village drill-down, showing
    (a) the raw 12 km NEPS-G field, (b) the tracked anomaly cone, (c) the 5 km
    downscaled hazard map, and (d) clickable cells that show the generated CAP
    alert text in English + at least one Indian language (template the
    translation; do not call a real Bhashini API — mock it and label it as
    mocked).
  - Alerts must be valid CAP 1.2 XML (validate against the schema) with 4
    severity levels matching the deck's classification.

NON-GOALS for this build (do not attempt, do not claim as done):
  - No live/real-time NEPS-G feed — use archived/open data for one historical
    event (recommend: Cyclone Phailin, October 2013 — well-documented public
    ground truth exists via IMD/IMERG/ERA5) as the running example throughout.
  - No real SACHET, Bhashini, or NDMA API calls — everything past the FastAPI
    boundary to those systems is mocked and must be visibly labeled "SIMULATED"
    in the UI.
  - No training from scratch on full-resolution global data — use open
    pretrained weights (GraphCast, any open CorrDiff-style checkpoint) where
    possible, or clearly document what was trained from scratch and on how
    little data.
  - Do not report any accuracy/latency number without logging exactly how it
    was measured (which event, how many samples, what hardware) next to it.

DATA SOURCES (all open/free — use these, do not invent placeholder data
without saying so):
  - ECMWF open data (open-data operational forecasts, or AIFS-ENS output as a
    12-28 km ensemble stand-in if raw NEPS-G is unavailable in this sandbox)
  - ERA5 / ERA5-Land reanalysis (Copernicus Climate Data Store)
  - IMDAA regional reanalysis (12 km, India-specific)
  - GPM IMERG precipitation (for 5 km-ish ground truth)
  - IMD gridded rainfall data (ground truth)
  If any of these require authenticated API keys the sandbox doesn't have,
  stop and tell me exactly which credential is missing rather than
  substituting silent synthetic data — I will provide it or approve a
  documented synthetic-data fallback.

TECH STACK (use exactly this unless a library is genuinely unavailable, in
which case tell me before substituting):
  - Python 3.11, PyTorch, PyTorch Geometric or DGL (mesh GNN)
  - HuggingFace Diffusers (residual diffusion)
  - xarray + Zarr (chunked NWP data handling)
  - FastAPI + Uvicorn (backend)
  - PostGIS via SQLAlchemy/GeoAlchemy2, or SQLite+SpatiaLite fallback
  - MapLibre GL JS (frontend map)
  - pytest for backend/model unit tests
  - MLflow (lightweight local tracking is fine) to log every run's metrics

MILESTONES (build and verify in this order; produce a checklist Artifact for
each; do not start milestone N+1 until milestone N's exit criterion is
demonstrably met with a screenshot, test output, or plotted figure):

  Milestone 0 — Repo & data plumbing
    - Scaffold the repo: /data, /models/tracker, /models/downscaler,
      /serving/api, /serving/frontend, /notebooks, /tests, README.md
    - Data loader that pulls/caches one historical event's ensemble data,
      regrids it to the M9 icosahedral mesh, and can plot the raw 12 km field.
    - Exit criterion: a notebook/script that loads the event and renders the
      raw ensemble mean as a map image, committed as a test artifact.

  Milestone 1 — Tracker (Stage 1)
    - Implement EFI/SOT computation and the mesh GNN tracker.
    - Exit criterion: for the chosen historical event, the tracker's output
      bounding box + uncertainty cone, plotted over the real event's known
      track, visually overlaps it. Save this comparison plot to
      /tests/artifacts/.

  Milestone 2 — Downscaler (Stage 2)
    - Implement the U-Net regression mean + residual diffusion model with
      terrain/land-use conditioning and the (at least partially implemented)
      composite loss.
    - Exit criterion: a side-by-side plot of (a) raw 12 km field, (b) bilinear
      baseline upsample, (c) ChakraNet 5 km diffusion output, (d) IMERG/IMD
      ground truth for the same event/time, plus a computed PSD comparison
      showing the diffusion output preserves more high-frequency structure
      than the bilinear baseline.

  Milestone 3 — Metrics & honesty pass
    - Compute the deck's KPI table for this one event: PSD fidelity, P99 rain
      bias, CRPS vs. baseline, and wall-clock latency on the actual hardware
      used. Write every number to a metrics.json with the measurement
      conditions (N=1 event, hardware spec) attached — never report a bare
      number.
    - Exit criterion: metrics.json + a rendered metrics table exist and match
      what Milestone 1/2 plots show.

  Milestone 4 — Serving layer
    - Build the FastAPI endpoints and PostGIS/SQLite storage, wire them to
      Milestone 1-2's outputs for the historical event, and implement CAP 1.2
      XML alert generation with schema validation.
    - Exit criterion: `pytest` passes for all endpoint tests, and a captured
      request/response example for each endpoint is saved to
      /tests/artifacts/api_examples/.

  Milestone 5 — Frontend & demo assembly
    - Build the MapLibre dashboard: raw field / tracked cone / hazard map /
      clickable CAP alerts, with the mocked-dispatch button clearly labeled
      "SIMULATED — not connected to live SACHET/Bhashini".
    - Exit criterion: a screen recording (or step-by-step screenshots) showing
      a user opening the map, seeing the historical event, drilling into a
      village-level cell, and viewing its generated CAP alert in English and
      one regional language.

  Milestone 6 — README & honest limitations doc
    - Write a top-level README that states: what is real (open data, trained
      models, actual measured metrics) vs. simulated (SACHET/Bhashini
      dispatch) vs. not yet implemented (any stubbed loss terms), and the
      exact commands to reproduce every result.
    - Exit criterion: a fresh read-through of the README with no unexplained
      claims; every number in it traces to a file in /tests/artifacts/.

Work through these milestones one at a time. After each, summarize what you
built, show me the exit-criterion evidence, and wait for my go-ahead before
starting the next milestone.
```

---

### Notes on why the prompt is structured this way
- **Milestone gating** mirrors Antigravity's own recommended workflow (break large missions into milestones, review the plan Artifact before execution) so the agent's plan mode and checklist artifacts have real structure to hang off of.
- **Explicit non-goals and "no silent fabrication" rules** are there because agentic coding tools left unconstrained on a "build a working AI weather system" brief will happily mock an entire ML pipeline with random numbers and call it done — the prompt forces it to say what's real.
- **One historical event, not a live feed** keeps the whole build achievable inside a hackathon's compute and time budget, exactly as scoped in the phased plan document.
