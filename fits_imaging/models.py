"""Data models for FITS imaging analysis."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ImagingResults:
    """Container and convenience interface for one analyzed FITS image."""

    image_path: Path
    image: np.ndarray
    record: Any
    peaks: np.ndarray
    stats: Dict[str, Any] = field(default_factory=dict)
    convolved_image: Optional[np.ndarray] = None
    header: Optional[Any] = None

    @property
    def object_name(self) -> str:
        return str(getattr(self.record, "object_name", ""))

    @property
    def camera(self) -> str:
        return str(getattr(self.record, "camera", ""))

    @property
    def exposure_sec(self) -> float:
        return float(getattr(self.record, "exposure_sec", np.nan))

    @property
    def softname(self) -> str:
        return str(getattr(self.record, "softname", ""))

    @property
    def title(self) -> str:
        return f"{self.object_name}   {self.camera}  {self.exposure_sec:.3f} sec"

    @property
    def display_image(self) -> np.ndarray:
        """Preferred image for display, using convolved image if available."""
        return self.convolved_image if self.convolved_image is not None else self.image

    @property
    def peak_array(self) -> np.ndarray:
        """Return peaks as a 2-D NumPy array.

        Expected columns are:
            x, y, area, sx, sy, flux
        """
        if self.peaks is None:
            return np.empty((0, 6), dtype=float)

        arr = np.asarray(self.peaks)

        if arr.size == 0:
            return np.empty((0, 6), dtype=float)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        return arr

    @property
    def nstars(self) -> int:
        return int(len(self.peak_array))

    @property
    def brightest_peak(self):
        peaks = self.peak_array

        if len(peaks) == 0 or peaks.shape[1] < 6:
            return None

        flux = peaks[:, 5]
        good = np.isfinite(flux)

        if not np.any(good):
            return None

        good_indices = np.where(good)[0]
        best_index = good_indices[np.argmax(flux[good])]

        return peaks[best_index]

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
    def image_min(self) -> float:
        return float(np.nanmin(self.image))

    @property
    def image_max(self) -> float:
        return float(np.nanmax(self.image))

    @property
    def image_shape(self):
        return self.image.shape

    @property
    def median_sigma_pixels(self) -> float:
        peaks = self.peak_array

        if len(peaks) == 0 or peaks.shape[1] < 5:
            return float("nan")

        sigma_mean = 0.5 * (peaks[:, 3] + peaks[:, 4])
        return float(np.nanmedian(sigma_mean))

    @property
    def median_fwhm_pixels(self) -> float:
        return 2.3548 * self.median_sigma_pixels

    @property
    def brightest_xy(self):
        peak = self.brightest_peak

        if peak is None or len(peak) < 2:
            return None

        return float(peak[0]), float(peak[1])
        
    def plot_image(self, config=None, **kwargs):
        from .plotting import plot_image_with_peaks
        return plot_image_with_peaks(
            self.display_image,
            self.peaks,
            title=self.title,
            config=config,
            **kwargs,
        )

    def plot_cutouts(self, config=None, **kwargs):
        from .plotting import plot_peak_cutouts
        return plot_peak_cutouts(
            self.display_image,
            self.peaks,
            config=config,
            **kwargs,
        )

    def plot_histogram(self, config=None, **kwargs):
        from .plotting import plot_histogram
        return plot_histogram(
            self.display_image,
            config=config,
            **kwargs,
        )

    def print_summary(self):
        from .report import print_image_summary
        return print_image_summary(self.record, self.display_image, self.stats)

    @property
    def analysis_mode(self) -> str:
        return str(self.stats.get("analysis_mode", "source_detection"))

    def print_peak_table(self, config=None, **kwargs):
        from .report import print_peak_table

        zero_mag_counts = getattr(config, "zero_mag_counts", 1.0)
        swarpfac = -1.0 if self.softname == "SWarp" else 1.0

        return print_peak_table(
            self.peaks,
            exposure_sec=self.exposure_sec,
            zero_mag_counts=zero_mag_counts,
            pa0=getattr(self.record, "pa_deg", 0.0),
            swarpfac=swarpfac,
            **kwargs,
        )
        
    @property
    def fwhm_pixels(self):
        from .diagnostics import peak_fwhm_pixels
        return peak_fwhm_pixels(self.peaks)

    def radial_profile(self, peak_index=0, rmax=20, binsize=1.0):
        from .diagnostics import radial_profile

        peaks = self.peak_array
        if len(peaks) == 0:
            raise ValueError("No peaks available")

        x, y = peaks[peak_index, 0], peaks[peak_index, 1]
        return radial_profile(self.display_image, x, y, rmax=rmax, binsize=binsize)

    def source_statistics(self):
        from .diagnostics import source_statistics
        return source_statistics(self.peaks)

    def export_csv(self, path, config=None):
        from .export import export_peak_csv

        zero_mag_counts = getattr(config, "zero_mag_counts", 1.0)
        pixel_arcsec = getattr(config, "pixel_arcsec", None)

        return export_peak_csv(
            path,
            self.peaks,
            exposure_sec=self.exposure_sec,
            zero_mag_counts=zero_mag_counts,
            pixel_arcsec=pixel_arcsec,
        )
        
    def report(
        self,
        config=None,
        max_rows=50,
        n_cutouts=25,
        spansize=24,
        show_summary=True,
        show_image=True,
        show_table=True,
        show_cutouts=True,
        show_histogram=False,
        show_statistics=True,
    ):
        """Run the standard quick-look report."""
        import matplotlib.pyplot as plt

        if show_summary:
            self.print_summary()

        if show_statistics and self.analysis_mode == "source_detection":
            print("\nSource statistics:")
            print(self.source_statistics())

            if config is not None:
                print("\nSelected-region source statistics:")
                print(self.selected_source_statistics(config=config))
        elif show_statistics:
            print(f"\nAnalysis mode: {self.analysis_mode}")

        if show_image:
            fig, ax = self.plot_image(config=config, figsize=(16, 16))
            plt.show()

        if show_table:
            self.print_peak_table(config=config, max_rows=max_rows)

        if show_cutouts and self.nstars > 0:
            fig, axs = self.plot_cutouts(
                config=config,
                nmax=n_cutouts,
                spansize=spansize,
                pixel_arcsec=getattr(config, "pixel_arcsec", None),
                exposure_sec=self.exposure_sec,
                zero_mag_counts=getattr(config, "zero_mag_counts", 1.0),
            )
            plt.show()

        if show_histogram:
            fig, ax = self.plot_histogram(config=config)
            plt.show()
            
    def psf_profile(self, peak_index=0, rmax=30, binsize=1.0):
        from .psf import radial_profile
        peaks = self.peak_array
        x, y = peaks[peak_index, 0], peaks[peak_index, 1]
        return radial_profile(self.display_image, x, y, rmax=rmax, binsize=binsize)

    def encircled_energy(self, peak_index=0, rmax=30, binsize=1.0):
        from .psf import encircled_energy
        peaks = self.peak_array
        x, y = peaks[peak_index, 0], peaks[peak_index, 1]
        return encircled_energy(self.display_image, x, y, rmax=rmax, binsize=binsize)

    def selected_peaks(self, config=None):
        from .regions import select_peaks_region
        return select_peaks_region(
            self.peaks,
            self.display_image.shape,
            config=config,
        )

    def selected_source_statistics(self, config=None):
        from .diagnostics import source_statistics
        return source_statistics(self.selected_peaks(config=config))
