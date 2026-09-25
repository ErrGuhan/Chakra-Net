# ChakraNet — Feasibility, Market Analysis & Phased Implementation Plan
**SIH 2026 · PS 26078 · AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts**
Ministry of Earth Sciences (MoES) · NCMRWF

---

## 1. What the deck is actually proposing (analysis of the PPT)

ChakraNet is a **two-stage hybrid AI pipeline** that takes India's existing 12 km NEPS-G ensemble forecast (23 members, 70 levels, 10-day range) and turns it into something no current system offers:

| Stage | Function | Core tech |
|---|---|---|
| **1 — Mesh GNN tracker** | Detects and tracks extreme-weather anomalies on an icosahedral mesh (M9), computes EFI/SOT (Extreme Forecast Index / Shift of Tails), warm-started from GraphCast weights | DGL / PyTorch Geometric |
| **2 — Residual diffusion downscaler** | Takes the tracked/cropped region and super-resolves it from 12 km → ~5 km, constrained by terrain, land use, moisture/mass-conservation physics loss | HuggingFace Diffusers / NVIDIA PhysicsNeMo (CorrDiff-style) |
| **Serving** | Cell-level probability-of-exceedance maps → 4-level CAP alerts → dashboard/SMS/voice | FastAPI · PostGIS · MapLibre · NDMA SACHET · Bhashini |

Claimed outputs: ~5.8× finer cells than raw NEPS-G, 3–10 day lead time, <10 min inference/event on one GPU, ₹0 MVP cost using open data (NEPS-G, ECMWF open data, ERA5, IMERG, IMDAA) and free compute (Kaggle/Colab, Oracle free tier / HF Spaces).

This is a credible, well-referenced architecture (GraphCast/Lam 2023, GenCast/Price 2023, CorrDiff/Mardani) — the deck's core intellectual contribution is **combining automatic anomaly tracking with physics-constrained downscaling and India-specific alert delivery**, not inventing new model architectures. That's the right call for a hackathon: recombining proven components is far more implementable in the available time than novel research.

---

## 2. Market & competitive landscape (verified, Sept 2026)

I checked the deck's comparison table against current public information:

- **ECMWF AIFS / AIFS-ENS** went operational Feb–Jul 2025 and is now ECMWF's co-equal AI system alongside IFS, running a 51-member ensemble at **0.25° (~28 km)**, with up to 20% gains on tropical-cyclone tracks and roughly 1,000× lower compute cost than physics runs.
- **Google DeepMind GenCast / WeatherNext 2** outperformed ECMWF's own ensemble on roughly 97.2% of ~1,320 evaluated targets, rising to about 99.8% beyond 36 hours, also at **0.25° (~28 km)**, ~8 minutes per member on a cloud TPU. As of early 2026 it is still not the primary deployed operational system at major forecasting centers (ECMWF continues to run AIFS operationally).
- Neither AIFS nor GenCast does **automatic anomaly tracking**, **sub-10 km probabilistic downscaling**, or **India-specific hyper-local alerting** — they stop at global 28 km grids and leave the "what does this mean for my village" translation to national met services.
- **BharatFS** (IITM/IMD, 6 km) is India's own high-resolution effort, but it is **deterministic** (single forecast, no ensemble/probability) and only reaches block-level granularity.
- **NEPS-G** (NCMRWF's operational ensemble, the deck's actual input data) is 12 km, 22+1 members, 10 days — high enough resolution to be useful but coarse enough that a coastal district and its neighboring open ocean sit in the same cell, and it is **read manually by forecasters today** ("manual overlay of 23 members" per the deck).

**The market gap is real, not marketing:** globally, the frontier of AI weather (AIFS, GenCast/WeatherNext) is racing toward *global accuracy at 28 km*, not *local actionability at 5 km*. India-specific high-resolution work (BharatFS) has traded away ensemble/probabilistic information to get resolution. **No system in the deck's table — or in the broader market — currently combines ensemble probability + automatic tracking + <10 km resolution + a CAP/SACHET-integrated alert path.** That combination, not raw model accuracy, is ChakraNet's differentiation, and it is defensible because it's an *integration and engineering* moat (tying into NCMRWF's actual operational data and India's actual alert infrastructure) rather than a claim of beating DeepMind or ECMWF at global NWP.

Supporting market sizing from the deck (Allied Market Research, third-party estimate) puts the global weather-services market at **~$1.63B (2020) → $4.19B (2030), ~10.3% CAGR** — directionally consistent with the sector's current AI-driven investment (ECMWF, Google, Microsoft Aurora, NVIDIA all shipped operational or near-operational AI weather systems in 2025–26), so the underlying trend the deck cites is current and accurate, not a stale figure.

### Where ChakraNet is genuinely differentiated
1. **Automatic vs. manual anomaly detection** — replaces forecaster overlay of 23 ensemble members with a GNN tracker (EFI+SOT) that flags anomalies computationally.
2. **Resolution where it matters, not everywhere** — instead of globally downscaling (expensive), it tracks first, then only diffusion-downscales the cropped anomaly region ("tracker crop = downscaler input"). This is the single biggest reason the ₹0/<10 min claims are plausible: it avoids full-grid super-resolution.
3. **Probabilistic, not deterministic, at high resolution** — BharatFS gives you one number; ChakraNet gives you P(exceed) per 5 km cell, which is what actually feeds a risk-based evacuation or crop decision.
4. **Last-mile delivery baked in** — CAP-format alerts wired to NDMA SACHET and Bhashini (multilingual voice/SMS) are part of the architecture, not an afterthought, which is what most research-grade weather AI (GenCast, AIFS) explicitly does not address.
5. **Cost structure fits India's public-sector reality** — ₹0 MVP, ~₹32k/year sustained inference is a claim that stands up because of point 2 (tile-only diffusion) — it is not just an aspirational number.

### Where the deck is weaker / needs honest framing for judges
- The "5.8× finer cells" and "<10 min" figures are **targets validated on a subset of cases**, not a production-proven SLA — say so explicitly rather than presenting them as measured facts. (The deck itself flags this: "Market figures are third-party estimates; KPIs are targets, not results.")
- **No public benchmark yet exists comparing ChakraNet-style tracker+downscaler pipelines against AIFS-ENS or GenCast head-to-head on Indian monsoon/cyclone cases** — this is a real gap the team should not paper over; it's also the strongest "future work" pitch (India-specific verification the global players haven't done).
- CorrDiff/PhysicsNeMo-style residual diffusion is proven on US/global reanalysis-scale problems, but **has not been widely validated on Indian orography** (Western Ghats, Himalayan foothills) — terrain-conditioning quality here is the highest-risk technical unknown, not the tracker.

---

## 3. Feasibility verdict

| Dimension | Verdict | Why |
|---|---|---|
| **Data access** | Feasible | NEPS-G via NCMRWF (organizing body — an SIH advantage), ECMWF open data, ERA5/ERA5-Land, IMERG, IMDAA are all real, currently-open sources |
| **Compute** | Feasible for MVP, constrained beyond | Free-tier GPUs (Kaggle/Colab T4/P100) can run inference on pretrained/warm-started models; **training from scratch is not feasible on free tiers** — the plan must lean on transfer learning from open GraphCast/CorrDiff weights |
| **Model risk** | Medium | Warm-starting a Mesh GNN from GraphCast and a diffusion decoder from CorrDiff-style checkpoints is realistic; getting physics losses (VIMFC, mass conservation) numerically stable in a hackathon timeframe is the hardest engineering task |
| **Integration (CAP/SACHET/Bhashini)** | Feasible to *simulate* | You will not get production SACHET/Bhashini credentials during a hackathon — build to the CAP 1.2 XML spec and mock the transport layer; this is honest and still fully demonstrates the pipeline |
| **5 km ground truth for validation** | Partially feasible | IMD gridded + IMERG + radar composites are public enough to validate on a handful of *historical* events (e.g., a past cyclone) — do not promise real-time validation in the demo |
| **Overall** | **Buildable as a working, demoable prototype in phases**, not as a production forecasting system in hackathon time | The right scope is: real pipeline architecture, real open data, pretrained/lightly fine-tuned models, 1–2 historical case studies fully worked end-to-end, honestly-labeled "targets vs. measured" KPIs |

**Net assessment: implementable and differentiated, provided the team scopes the hackathon prototype to "pipeline + 1-2 proven historical case studies" rather than "general real-time operational system."** That scoping is what the phased plan below is built around.

---

## 4. Phased production plan

### Phase 0 — Foundation & data plumbing (Days 1–2)
- Stand up repo structure, environment, and CI.
- Pull and cache a working slice of NEPS-G-equivalent data (use ECMWF open data / GFS ensemble as a stand-in if raw NEPS-G access is delayed) + ERA5 + IMERG + IMDAA for **one historical extreme event** (recommend Cyclone Phailin 2013 or a recent well-documented monsoon extreme — good public ground truth exists).
- Build the Zarr-chunked data loader and icosahedral (M9) regridding utility.
- **Exit criterion:** one event's 12 km ensemble loads, regrids, and visualizes end-to-end.

### Phase 1 — Tracking stage (Days 2–4)
- Implement/adapt Mesh GNN on the regridded mesh; warm-start from open GraphCast weights.
- Compute EFI + SOT per node against the M-climate / re-forecast baseline.
- Cluster + track anomalies into the 4D (lat, lon, level, time) box.
- **Exit criterion:** tracker draws an uncertainty cone around the known historical event that visibly matches the real storm track.

### Phase 2 — Downscaling stage (Days 4–6)
- Crop tracker output; feed into residual diffusion (HF Diffusers pipeline, CorrDiff-style U-Net + diffusion residual), conditioned on 5 km terrain/land-use.
- Implement the composite loss (tail-weighted CRPS + PSD + VIMFC + mass-consistency) — even a partial/simplified version is fine for MVP; document which terms are active.
- Sample 16 fields → collapse to P(exceed) probability maps.
- **Exit criterion:** downscaled 5 km field is visually and statistically sharper than bilinear/U-Net baseline on the same case (PSD comparison plot is a strong demo artifact).

### Phase 3 — Alerting & serving layer (Days 6–8)
- FastAPI backend serving tracker + downscaler outputs as GeoJSON/tiles.
- PostGIS for cell-level storage; MapLibre front-end dashboard (district → village drill-down).
- Generate CAP 1.2-compliant alert XML at 4 severity levels; mock the SACHET/Bhashini dispatch (log + simulated multilingual SMS text) rather than claiming live integration.
- **Exit criterion:** clicking a flagged cell on the map produces a real CAP alert payload, in at least Hindi + English via Bhashini-style templated translation.

### Phase 4 — Validation & KPI reporting (Days 8–9)
- Score the pipeline on the historical case against the deck's KPI table (PSD fidelity, P99 rain bias, CRPS vs. bilinear/U-Net baseline, hit/false-alarm rate, latency).
- Clearly label every number as "measured on N=1–2 historical events" — this is honest and still compelling for judges.
- Produce the risk/mitigation and cost slides using **actual measured GPU time and storage**, not just the deck's estimates.

### Phase 5 — Demo polish & pitch (Day 10)
- Package a single end-to-end runnable notebook/script + the live dashboard.
- Record a "storm replay": input ensemble → tracker → downscaler → alert, on the historical case, in under the target latency.
- Prepare the honest limitations slide (small-N validation, mocked transport layer, terrain generalization risk) — SIH judges consistently reward teams that are precise about what is real vs. simulated.

### Beyond the hackathon (roadmap, for the "future work" slide)
1. Formal partnership with NCMRWF for live NEPS-G feed and forecaster-in-the-loop validation.
2. Multi-event, multi-season validation (not just cyclones — heat domes, cold waves per the deck's environmental section).
3. Real CAP/SACHET/Bhashini integration through MoES/NDMA channels.
4. MLflow + DVC retraining pipeline, annual retrain as scoped in the deck's ₹32k/yr sustain cost.
5. Independent benchmark of ChakraNet's downscaled output against AIFS-ENS/GenCast at native 5 km on Indian terrain — this is the paper-worthy contribution the deck sets up but doesn't yet have data for.

---

## 5. Bottom line
The architecture is sound, the differentiation (tracker-then-crop-then-downscale, at village scale, wired to India's real alert infrastructure) is real and not offered by any current global or Indian system, and the cost model survives scrutiny because it avoids full-grid downscaling. The one thing the team must not do is present hackathon-stage, small-sample results as production-validated — scope the demo to 1–2 fully-worked historical events, label targets as targets, and the project is both feasible to build in the time available and genuinely differentiated in the current market.
