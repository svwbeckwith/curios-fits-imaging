"""Data models for FITS imaging analysis."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ImagingResults:
    """Container for one analyzed FITS image."""

    image_path: Path
    image: np.ndarray
    record: Any
    peaks: np.ndarray
    stats: Dict[str, Any] = field(default_factory=dict)
    convolved_image: Optional[np.ndarray] = None
    header: Optional[Any] = None

    @property
    def object_name(self) -> str:
        return getattr(self.record, "object_name", "")

    @property
    def exposure_sec(self) -> float:
        return float(getattr(self.record, "exposure_sec", 0.0))

    @property
    def camera(self) -> str:
        return getattr(self.record, "camera", "")

    @property
    def title(self) -> str:
        return f"{self.object_name}   {self.camera}  {self.exposure_sec:.3f} sec"
        
#    @property
#    def nstars(self) -> int:
#        return 0 if self.peaks is None else len(self.peaks)

#    @property
#    def brightest_peak(self):
#        if self.peaks is None or len(self.peaks) == 0:
#            return None
#        peaks = np.asarray(self.peaks)
#        return peaks[np.argmax(peaks[:, 5])]

#    @property
#    def background_mean(self) -> float:
#        return float(self.stats.get("mean", np.nan))

#    @property
#    def background_sigma(self) -> float:
#        return float(self.stats.get("std", self.stats.get("sigma", np.nan)))

#    @property
#    def median_background(self) -> float:
3        return float(self.stats.get("median", np.nan))

    @property
    def peak_array(self):
        return np.asarray(self.peaks) if self.peaks is not None else np.empty((0, 6))

    @property
    def nstars(self) -> int:
        return len(self.peak_array)

    @property
    def brightest_peak(self):
        peaks = self.peak_array
        if len(peaks) == 0 or peaks.shape[1] < 6:
            return None
        return peaks[np.argmax(peaks[:, 5])]

    @property
    def brightest_flux(self) -> float:
        peak = self.brightest_peak
        return float("nan") if peak is None else float(peak[5])

    @property
    def median_background(self) -> float:
        return float(self.stats.get("median", np.nan))

    @property
    def background_mean(self) -> float:
        return float(self.stats.get("mean", np.nan))

    @property
    def background_sigma(self) -> float:
        return float(self.stats.get("std", self.stats.get("sigma", np.nan)))

    @property
    def median_sigma_pixels(self) -> float:
        peaks = self.peak_array
        if len(peaks) == 0 or peaks.shape[1] < 5:
            return float("nan")
        return float(np.nanmedian(0.5 * (peaks[:, 3] + peaks[:, 4])))

    @property
    def median_fwhm_pixels(self) -> float:
        return 2.3548 * self.median_sigma_pixels
