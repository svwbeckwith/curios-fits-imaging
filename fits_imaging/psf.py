"""PSF and source-profile analysis."""

import numpy as np


def radial_profile(image, x, y, rmax=30, binsize=1.0, subtract_background=True):
    image = np.asarray(image)
    ny, nx = image.shape

    x0 = max(int(np.floor(x - rmax)), 0)
    x1 = min(int(np.ceil(x + rmax + 1)), nx)
    y0 = max(int(np.floor(y - rmax)), 0)
    y1 = min(int(np.ceil(y + rmax + 1)), ny)

    cutout = image[y0:y1, x0:x1]
    yy, xx = np.indices(cutout.shape)
    rr = np.sqrt((xx + x0 - x) ** 2 + (yy + y0 - y) ** 2)

    if subtract_background:
        outer = rr > 0.75 * rmax
        background = np.nanmedian(cutout[outer]) if np.any(outer) else 0.0
    else:
        background = 0.0

    values = cutout - background

    bins = np.arange(0, rmax + binsize, binsize)
    radius = 0.5 * (bins[:-1] + bins[1:])
    profile = np.full_like(radius, np.nan, dtype=float)

    for i in range(len(radius)):
        mask = (rr >= bins[i]) & (rr < bins[i + 1])
        if np.any(mask):
            profile[i] = np.nanmean(values[mask])

    return radius, profile


def encircled_energy(image, x, y, rmax=30, binsize=1.0, subtract_background=True):
    image = np.asarray(image)
    ny, nx = image.shape

    x0 = max(int(np.floor(x - rmax)), 0)
    x1 = min(int(np.ceil(x + rmax + 1)), nx)
    y0 = max(int(np.floor(y - rmax)), 0)
    y1 = min(int(np.ceil(y + rmax + 1)), ny)

    cutout = image[y0:y1, x0:x1]
    yy, xx = np.indices(cutout.shape)
    rr = np.sqrt((xx + x0 - x) ** 2 + (yy + y0 - y) ** 2)

    if subtract_background:
        outer = rr > 0.75 * rmax
        background = np.nanmedian(cutout[outer]) if np.any(outer) else 0.0
    else:
        background = 0.0

    values = cutout - background

    radius = np.arange(binsize, rmax + binsize, binsize)
    energy = np.full_like(radius, np.nan, dtype=float)

    for i, r in enumerate(radius):
        mask = rr <= r
        energy[i] = np.nansum(values[mask])

    if np.nanmax(energy) > 0:
        energy = energy / np.nanmax(energy)

    return radius, energy
