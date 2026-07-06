"""Export utilities for FITS imaging results."""

from pathlib import Path
import csv
import numpy as np

from .diagnostics import peak_fwhm_pixels
from .photometry import counts_to_mag


def export_peak_csv(
    path,
    peaks_array,
    exposure_sec,
    zero_mag_counts,
    pixel_arcsec=None,
):
    """Export peak table to CSV."""
    path = Path(path)
    peaks = np.asarray(peaks_array)

    if peaks.size == 0:
        peaks = np.empty((0, 6))

    if peaks.ndim == 1:
        peaks = peaks.reshape(1, -1)

    fwhm = peak_fwhm_pixels(peaks)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)

        header = [
            "index",
            "x",
            "y",
            "area",
            "sigma_x",
            "sigma_y",
            "flux",
            "magnitude",
            "fwhm_pixels",
        ]

        if pixel_arcsec is not None:
            header.append("fwhm_arcsec")

        writer.writerow(header)

        for i, row in enumerate(peaks):
            x, y, area, sx, sy, flux = row[:6]

            mag = counts_to_mag(
                flux,
                exposure=exposure_sec,
                zero_mag_counts=zero_mag_counts,
            )

            out = [
                i,
                float(x),
                float(y),
                float(area),
                float(sx),
                float(sy),
                float(flux),
                float(mag),
                float(fwhm[i]) if i < len(fwhm) else np.nan,
            ]

            if pixel_arcsec is not None:
                out.append(float(fwhm[i] * pixel_arcsec) if i < len(fwhm) else np.nan)

            writer.writerow(out)

    return path
