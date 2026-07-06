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

    # Display
    contrast_method: str = "zscale"      # "zscale", "percentile", "sigma", "manual"
    contrast_percentiles: tuple = (1.0, 99.7)
    contrast_sigma: float = 5.0
    manual_vmin: Optional[float] = None
    manual_vmax: Optional[float] = None
