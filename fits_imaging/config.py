"""Configuration for FITS imaging analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ImagingConfig:
    """User-adjustable configuration values."""

    # Change this on each computer.
    data_root: Path = Path("~/Dropbox/CuRIOS/Software/DataDirectories").expanduser()

    # Photometry
    zero_mag_counts: float = 115000000
    pixel_arcsec: float = 1.24

    # Peak finding
    threshold_sigma: float = 5.0
    max_peaks: int = 1000
    peak_separation: float = 10.0
    peak_sharp: float = 0.2
    fit_method: str = "gaussian"

    # Display
    contrast_method: str = "sigma"
    contrast_percentiles: tuple = (1.0, 99.8)
    contrast_sigma_low: float = 1.0
    contrast_sigma_high: float = 5.0
    stretch: str = "linear"

    show_grid = True
    show_peak_labels = True
    show_center_cross = True
    show_peak_circles = False

    colormap: str = "viridis"      # Default Matplotlib colormap
    invert_colormap: bool = False
    invert_gray: bool = False
    histogram_max_points: int = 2_000_000
    histogram_bins: int = 200
    
    # Statistics region
    stats_region: str = "center_fraction"   # "full", "center_fraction", or "center_pixels"
    stats_center_fraction: float = 0.5      # central 50% in x and y
    stats_center_pixels: int = 2000         # used only for center_pixels
    
