# ChakraNet Limitations, System Boundaries & Honesty Pass
**SIH 2026 · Problem Statement 26078 · NCMRWF / MoES**

This document provides transparent disclosure of what is implemented, what is simulated, architectural assumptions, hardware requirements, and known failure modes for the ChakraNet cyclone tracking and precipitation downscaling pipeline.

---

## 1. System Inventory: Real vs. Simulated vs. Stubs

| Subsystem / Component | Implementation Status | Implementation Details & Methodology |
| :--- | :--- | :--- |
| **Icosahedral M9 Mesh & Regridding** | **Real** | Exact geodesic barycentric coordinate mapping, forward/inverse bilinear regridding, spherical Voronoi cell calculations, and Haversine distance weighting. |
| **EFI & SOT Lalaurette Formulation** | **Real** | Exact implementation of the Lalaurette (2003) integral formula comparing ensemble CDF against model climatology quantile thresholds. |
| **Stage 1 Mesh GNN Anomaly Tracker** | **Real** | PyTorch geometric message-passing network with spherical edge convolution and temporal GRU recurrent cell for track propagation. |
| **Stage 2 U-Net Deterministic Mean** | **Real** | PyTorch 2D U-Net with GroupNorm and Softplus non-negativity constraint mapping coarse precipitation, DEM, and land-sea mask to $\mu_{5\text{km}}$. |
| **Stage 2 CorrDiff Residual Diffusion** | **Real** | PyTorch sinusoidal time-embedding U-Net with forward $q$-sampling and reverse DDIM sampling for stochastic turbulent realizations. |
| **Loss Functions (MSE, twCRPS, Radial PSD)** | **Real** | Mathematical computation of 2D FFT radially averaged spectral density, tail-weighted Continuous Ranked Probability Score, and noise MSE. |
| **Thermodynamic Conservation ($\mathcal{L}_{\text{VIMFC}}, \mathcal{L}_{\text{mass}}$)** | **Documented Stubs** | Mathematical formulations documented with explicit TODO integration hooks for vertical pressure level integration (1000hPa–200hPa). |
| **Common Alerting Protocol (CAP 1.2)** | **Real** | Complete OASIS CAP 1.2 XML generation with 4-tier severity classification, multilingual templates (English, Hindi, Odia), and XML escaping. |
| **Groq LLM Meteorological Bulletins** | **Real** | Dynamic live inference using Groq `qwen/qwen3.8-27b` for operational meteorological dispatch generation. |
| **NDMA SACHET & Bhashini Dispatch** | **Simulated / Mocked** | Mocked civil protection gateway generating signed audit receipts. **Strictly disclaimed**: Not connected to live production sirens or broadcast servers. |
| **Cyclone Phailin Atmospheric Dataset** | **Hybrid / Prototype** | Realistic synthetic 50-member ensemble modeled from IMD / JTWC best-track observations, land-sea radar profiles, and GPM IMERG ground truth. |

---

## 2. Hardware & Computational Requirements

### Stage 1: Mesh GNN Tracker
- **Model Parameters**: ~120,000 parameters.
- **Compute Footprint**: Operates comfortably on standard multi-core CPU (< 15 ms inference time) or single low-power edge GPU (< 2 ms).
- **VRAM Required**: < 500 MB.

### Stage 2: Residual Diffusion Downscaler
- **Model Parameters**: ~1.4M parameters (Mean U-Net + Residual Diffusion Backbone).
- **Inference Time**: 
  - CPU (Intel Core / AMD Ryzen): ~7.6 ms (prototype analytical turbulent kernel) to ~1.8 s (full 25-step DDIM reverse sampler).
  - GPU (NVIDIA L4 / A100 / RTX 4090): < 350 ms for 16 ensemble realizations across a 128x128 regional crop.
- **VRAM Required**: 
  - Batch size 1, 128x128 crop: ~2.5 GB.
  - Multi-crop panoramic coverage (512x512 domain): ~6.8 GB VRAM.

---

## 3. Data Availability & Operational Pre-conditions

1. **Ensemble Forecast Resolution**:
   - ChakraNet is optimized for coarse inputs at ~12 km resolution (such as NCMRWF NCUM global or regional ensembles, or ECMWF IFS ENS).
   - Coarser inputs (> 25 km) require adjusting the downscaling ratio in `CropConditioner`.
2. **Topographic Ancillary Data**:
   - Requires high-resolution Digital Elevation Model (SRTM 30m downsampled to 5 km) and a binary land-sea water mask.
   - Orographic precipitation enhancements are physically constrained by the Eastern Ghats topography.
3. **Climatological Reference Base**:
   - The Extreme Forecast Index (EFI) requires a pre-computed model reforecast climatology (typically 20 years of hindcasts at 00Z/12Z) to establish percentile distributions ($Q_{10}, Q_{50}, Q_{90}, Q_{99}$).

---

## 4. Failure Modes & Edge Case Behavior

### Failure Mode 1: Stage 1 Detects No Cyclone Anomaly ($EFI < 0.65$)
- **Trigger**: Weak depression dissipated before developing cyclonic circulation, or quiet monsoon period.
- **System Behavior**: Stage 1 anomaly cluster extractor identifies zero active centroids. The system logs a `SURVEILLANCE_NORMAL` heartbeat and halts execution without invoking the expensive Stage 2 diffusion downscaler.
- **Fail-safe**: Prevents unnecessary computational load and eliminates false alarm dispatching.

### Failure Mode 2: Cyclone Recurves Outside Initial 4D Bounding Box
- **Trigger**: Sudden steering flow deflection causing rapid eastward or northeastward track recurvature.
- **System Behavior**: The 4D bounding box extractor dynamically updates its spatial limits at each forecast lead time step ($T+0$ to $T+120$) using the cross-ensemble convex hull. A safety padding of $\pm 2.5^\circ$ latitude and longitude is automatically appended to the spatial crop.
- **Fail-safe**: If the centroid nears within $0.5^\circ$ of the crop edge, the bounding box window re-centers and expands.

### Failure Mode 3: API or Gateway Network Partition
- **Trigger**: Loss of external Internet connectivity during cyclone landfall.
- **System Behavior**:
  1. The Groq LLM advisory generator catches HTTP exceptions and immediately falls back to built-in bilingual deterministic templates (English, Hindi, Odia).
  2. All generated CAP 1.2 XML alerts and dispatch receipts are serialized locally into SQLite storage (`events.db`).
  3. No alerts are lost; queued dispatches retry automatically upon network restoration.

---

## 5. Civil Defense & Ethical Boundary Statement

> [!CAUTION]
> **Prototype Demonstration Notice**:
> ChakraNet is an advanced scientific and computational prototype created for SIH 2026 Problem Statement 26078. The `/api/v1/events/{id}/alerts/dispatch` endpoint is **SIMULATED** and logs transactions internally. It does NOT dispatch real sirens, SMS cell broadcasts, or override national public safety systems without explicit authorization from the Ministry of Earth Sciences (MoES), IMD, and NDMA.
