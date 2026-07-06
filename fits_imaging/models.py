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
        
    @property
    def nstars(self) -> int:
        return 0 if self.peaks is None else len(self.peaks)

    @property
    def brightest_peak(self):
        if self.peaks is None or len(self.peaks) == 0:
            return None
        peaks = np.asarray(self.peaks)
        return peaks[np.argmax(peaks[:, 5])]

    @property
    def background_mean(self) -> float:
        return float(self.stats.get("mean", np.nan))

    @property
    def background_sigma(self) -> float:
        return float(self.stats.get("std", self.stats.get("sigma", np.nan)))

    @property
    def median_background(self) -> float:
        return float(self.stats.get("median", np.nan))
