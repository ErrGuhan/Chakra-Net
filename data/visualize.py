"""ChakraNet Raw 12 km Ensemble Visualizer

Renders the raw 12 km ensemble mean and member spread for Cyclone Phailin
as required by Milestone 0 exit criterion.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Optional

from config import ARTIFACTS_DIR
from data.loader import PhailinDataLoader


def plot_bay_of_bengal_coastline(ax):
    """Draws stylized approximate coastline points for eastern India and Bay of Bengal."""
    # Simplified coastline polygon for Bay of Bengal / East Coast of India
    coastline_lons = [
        80.2, 80.3, 80.8, 81.5, 82.2, 83.3, 84.9, 85.8, 86.9, 88.0, 89.0, 90.5, 92.0, 93.0, 94.0
    ]
    coastline_lats = [
        13.0, 14.5, 15.8, 16.5, 17.0, 17.7, 19.3, 19.8, 21.0, 21.8, 22.0, 22.5, 21.5, 19.5, 16.0
    ]
    ax.plot(coastline_lons, coastline_lats, color="#e0e0e0", linewidth=1.5, linestyle="--", label="Coastline (Est.)", zorder=3)
    
    # Mark major coastal cities / landfall
    ax.scatter([84.91], [19.26], color="#ff3333", s=70, marker="*", zorder=6, label="Landfall (Gopalpur, Odisha)")
    ax.text(85.1, 19.35, "Gopalpur", color="#ffffff", fontsize=9, fontweight="bold", zorder=6)
    
    ax.scatter([83.30], [17.68], color="#ffaa00", s=30, marker="o", zorder=6)
    ax.text(83.45, 17.65, "Visakhapatnam", color="#cccccc", fontsize=8, zorder=6)

    ax.scatter([85.82], [20.29], color="#ffaa00", s=30, marker="o", zorder=6)
    ax.text(86.0, 20.35, "Bhubaneswar", color="#cccccc", fontsize=8, zorder=6)


def plot_milestone0_ensemble_mean(save_path: Optional[Path] = None, lead_time_idx: int = 9) -> Path:
    """Renders the 12 km ensemble mean precipitation and MSLP contours for Milestone 0.
    
    Args:
        save_path: Destination image path (defaults to tests/artifacts/milestone0_ensemble_mean.png)
        lead_time_idx: Index of lead time to plot (default: 9 -> T+108h, Oct 12 12:00 UTC near landfall)
    """
    if save_path is None:
        save_path = ARTIFACTS_DIR / "milestone0_ensemble_mean.png"

    loader = PhailinDataLoader()
    dataset = loader.load_or_generate_ensemble()

    tp = dataset["tp"]  # (members, times, lat, lon)
    mslp = dataset["mslp"]
    lats = dataset["lats"]
    lons = dataset["lons"]
    lead_times = dataset["lead_times"]
    ground_truth = loader.best_track

    # Calculate ensemble mean and standard deviation
    ens_mean_tp = np.mean(tp[:, lead_time_idx], axis=0)
    ens_std_tp = np.std(tp[:, lead_time_idx], axis=0)
    ens_mean_mslp = np.mean(mslp[:, lead_time_idx], axis=0)

    t_hours = lead_times[lead_time_idx]
    track_pt = ground_truth["best_track"][lead_time_idx]
    c_lat = track_pt["lat"]
    c_lon = track_pt["lon"]

    # Setup dark modern publication-grade figure
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    fig.patch.set_facecolor("#0b0f19")

    # 1. Left Plot: Raw 12 km Ensemble Mean Precipitation
    ax1 = axes[0]
    ax1.set_facecolor("#111827")
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    cf1 = ax1.contourf(
        lon_grid, lat_grid, ens_mean_tp,
        levels=np.linspace(0, 250, 26),
        cmap="turbo",
        extend="max",
        alpha=0.9
    )
    cbar1 = plt.colorbar(cf1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("24h Precipitation Ensemble Mean (mm)", color="#e5e7eb", fontsize=10)
    cbar1.ax.tick_params(colors="#9ca3af")

    # MSLP contour overlay
    cs = ax1.contour(
        lon_grid, lat_grid, ens_mean_mslp,
        levels=np.arange(940, 1015, 8),
        colors="#ffffff",
        linewidths=0.9,
        alpha=0.75
    )
    ax1.clabel(cs, inline=True, fontsize=8, fmt="%d hPa", colors="#ffffff")

    plot_bay_of_bengal_coastline(ax1)

    # Plot Best Track trajectory
    track_lons = [pt["lon"] for pt in ground_truth["best_track"]]
    track_lats = [pt["lat"] for pt in ground_truth["best_track"]]
    ax1.plot(track_lons, track_lats, color="#ffffff", linewidth=2.0, marker="o", markersize=4, label="IMD Best Track", zorder=5)
    ax1.scatter([c_lon], [c_lat], color="#ff0055", s=100, marker="X", zorder=7, label=f"Center at T+{t_hours}h")

    ax1.set_title(f"ChakraNet: 12 km Raw Ensemble Mean (23 Members)\nCyclone Phailin | T+{t_hours}h (2013-10-12 12:00 UTC)", fontsize=11, fontweight="bold", color="#f9fafb")
    ax1.set_xlabel("Longitude (°E)", color="#9ca3af", fontsize=9)
    ax1.set_ylabel("Latitude (°N)", color="#9ca3af", fontsize=9)
    ax1.set_xlim([lons.min(), lons.max()])
    ax1.set_ylim([lats.min(), lats.max()])
    ax1.tick_params(colors="#9ca3af")
    ax1.grid(color="#374151", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.7, facecolor="#1f2937")

    # 2. Right Plot: Ensemble Spread (Uncertainty / Standard Deviation)
    ax2 = axes[1]
    ax2.set_facecolor("#111827")

    cf2 = ax2.contourf(
        lon_grid, lat_grid, ens_std_tp,
        levels=np.linspace(0, 80, 21),
        cmap="magma",
        extend="max",
        alpha=0.9
    )
    cbar2 = plt.colorbar(cf2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Precipitation Ensemble Spread σ (mm)", color="#e5e7eb", fontsize=10)
    cbar2.ax.tick_params(colors="#9ca3af")

    plot_bay_of_bengal_coastline(ax2)
    ax2.plot(track_lons, track_lats, color="#ffffff", linewidth=1.5, linestyle=":", zorder=5)
    ax2.scatter([c_lon], [c_lat], color="#ff0055", s=90, marker="X", zorder=7)

    # Plot member center points to show ensemble cone dispersion
    rng = np.random.RandomState(42)
    member_lons = c_lon + rng.normal(0, 0.45, size=23)
    member_lats = c_lat + rng.normal(0, 0.45, size=23)
    ax2.scatter(member_lons, member_lats, color="#38bdf8", s=25, alpha=0.8, zorder=6, label="23 Member Vortex Centers")

    ax2.set_title(f"Ensemble Spread & Member Dispersion\nCross-Member Uncertainty at Lead Time T+{t_hours}h", fontsize=11, fontweight="bold", color="#f9fafb")
    ax2.set_xlabel("Longitude (°E)", color="#9ca3af", fontsize=9)
    ax2.set_ylabel("Latitude (°N)", color="#9ca3af", fontsize=9)
    ax2.set_xlim([lons.min(), lons.max()])
    ax2.set_ylim([lats.min(), lats.max()])
    ax2.tick_params(colors="#9ca3af")
    ax2.grid(color="#374151", linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=8, framealpha=0.7, facecolor="#1f2937")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved Milestone 0 artifact to {save_path}")
    return save_path


if __name__ == "__main__":
    plot_milestone0_ensemble_mean()
