"""Scientific diagnostics for FITS imaging results."""

import numpy as np

def peak_fwhm_pixels(peaks_array):
    """Return FWHM values from peak array columns [x, y, area, sx, sy, flux]."""
    peaks = np.asarray(peaks_array)

    if peaks.size == 0:
        return np.array([])

    if peaks.ndim == 1:
        peaks = peaks.reshape(1, -1)

    if peaks.shape[1] < 5:
        return np.array([])

    sigma_mean = 0.5 * (peaks[:, 3] + peaks[:, 4])
    return 2.3548 * sigma_mean


def radial_profile(image, x, y, rmax=20, binsize=1.0):
    """Compute mean radial profile around x, y."""
    image = np.asarray(image)

    ny, nx = image.shape
    x0 = max(int(np.floor(x - rmax)), 0)
    x1 = min(int(np.ceil(x + rmax + 1)), nx)
    y0 = max(int(np.floor(y - rmax)), 0)
    y1 = min(int(np.ceil(y + rmax + 1)), ny)

    cutout = image[y0:y1, x0:x1]

    yy, xx = np.indices(cutout.shape)
    rr = np.sqrt((xx + x0 - x) ** 2 + (yy + y0 - y) ** 2)

    bins = np.arange(0, rmax + binsize, binsize)
    radius = 0.5 * (bins[:-1] + bins[1:])
    profile = np.full_like(radius, np.nan, dtype=float)

    for i in range(len(radius)):
        mask = (rr >= bins[i]) & (rr < bins[i + 1])
        if np.any(mask):
            profile[i] = np.nanmean(cutout[mask])

    return radius, profile


def source_statistics(peaks_array):
    """Return simple statistics for detected sources."""
    peaks = np.asarray(peaks_array)

    if peaks.size == 0:
        return {
            "nstars": 0,
            "median_fwhm": np.nan,
            "median_flux": np.nan,
            "brightest_flux": np.nan,
        }

    if peaks.ndim == 1:
        peaks = peaks.reshape(1, -1)

    fwhm = peak_fwhm_pixels(peaks)

    flux = peaks[:, 5] if peaks.shape[1] >= 6 else np.array([])

    return {
        "nstars": len(peaks),
        "median_fwhm": float(np.nanmedian(fwhm)) if len(fwhm) else np.nan,
        "median_flux": float(np.nanmedian(flux)) if len(flux) else np.nan,
        "brightest_flux": float(np.nanmax(flux)) if len(flux) else np.nan,
    }
