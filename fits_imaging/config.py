"""Configuration for FITS imaging analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class ImagingConfig:
    """User-adjustable configuration values."""

    # Change this on each computer.
    data_root: Optional[Path] = None
    #data_root: Path = Path("~/Dropbox/CuRIOS/Software/DataDirectories").expanduser()

    # Photometry
    zero_mag_counts: float = 115000000
    pixel_arcsec: float = 1.55

    # Peak finding
    threshold_sigma: float = 10.0
    max_peaks: int = 100
    peak_separation: float = 10.0
    peak_sharp: float = 0.2
    fit_method: str = "gaussian"

    # Display
    contrast_method: str = "sigma"
    contrast_percentiles: tuple = (1.0, 99.8)
    contrast_sigma_low: float = 1.0
    contrast_sigma_high: float = 5.0
    stretch: str = "linear"
    manual_vmin: Optional[float] = None
    manual_vmax: Optional[float] = None

    # Cutout display (independent of the full-image display settings).
    # Supported methods are "sigma", "percentile", "zscale", and "manual".
    cutout_contrast_method: str = "sigma"
    cutout_contrast_percentiles: tuple = (1.0, 99.5)
    cutout_contrast_sigma_low: float = 1.0
    cutout_contrast_sigma_high: float = 5.0
    cutout_stretch: str = "linear"
    cutout_colormap: Optional[str] = None
    cutout_vmin: Optional[float] = None
    cutout_vmax: Optional[float] = None

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
    
    # Source diagnostics region
    source_region: str = "center_fraction"   # "full" or "center_fraction"
    source_region_fraction: float = 0.5
    
    def __post_init__(self):
        if self.data_root is not None:
            self.data_root = Path(self.data_root).expanduser().resolve()

    def use_standard_main_display(self):
        """Use balanced display settings for the full image."""
        self.contrast_method = "sigma"
        self.contrast_sigma_low = 1.0
        self.contrast_sigma_high = 5.0
        self.stretch = "linear"
        self.manual_vmin = None
        self.manual_vmax = None

    def use_bahtinov_main_display(self):
        """Saturate the full-image core to emphasize faint mask arms."""
        self.contrast_method = "sigma"
        self.contrast_sigma_low = 0.5
        self.contrast_sigma_high = 2.0
        self.stretch = "sqrt"
        self.manual_vmin = None
        self.manual_vmax = None

    def use_percentile_main_display(self, low=1.0, high=99.5, stretch="linear"):
        """Scale the full image from the requested percentile range."""
        self.contrast_method = "percentile"
        self.contrast_percentiles = (float(low), float(high))
        self.stretch = stretch
        self.manual_vmin = None
        self.manual_vmax = None

    def use_manual_main_display(self, vmin, vmax, stretch="linear"):
        """Use fixed data limits for the full image."""
        if vmax <= vmin:
            raise ValueError("main-image vmax must be greater than vmin")
        self.contrast_method = "manual"
        self.manual_vmin = float(vmin)
        self.manual_vmax = float(vmax)
        self.stretch = stretch

    def use_standard_cutout_display(self):
        """Use balanced display settings for ordinary stellar cutouts."""
        self.cutout_contrast_method = "sigma"
        self.cutout_contrast_sigma_low = 1.0
        self.cutout_contrast_sigma_high = 5.0
        self.cutout_stretch = "linear"
        self.cutout_vmin = None
        self.cutout_vmax = None

    def use_bahtinov_display(self):
        """Saturate bright cores to emphasize faint Bahtinov-mask arms."""
        self.cutout_contrast_method = "sigma"
        self.cutout_contrast_sigma_low = 0.5
        self.cutout_contrast_sigma_high = 2.0
        self.cutout_stretch = "sqrt"
        self.cutout_vmin = None
        self.cutout_vmax = None

    def use_percentile_cutout_display(self, low=1.0, high=99.5, stretch="linear"):
        """Scale each cutout from the requested percentile range."""
        self.cutout_contrast_method = "percentile"
        self.cutout_contrast_percentiles = (float(low), float(high))
        self.cutout_stretch = stretch
        self.cutout_vmin = None
        self.cutout_vmax = None

    def use_manual_cutout_display(self, vmin, vmax, stretch="linear"):
        """Use fixed data limits for every cutout."""
        if vmax <= vmin:
            raise ValueError("cutout vmax must be greater than vmin")
        self.cutout_contrast_method = "manual"
        self.cutout_vmin = float(vmin)
        self.cutout_vmax = float(vmax)
        self.cutout_stretch = stretch
