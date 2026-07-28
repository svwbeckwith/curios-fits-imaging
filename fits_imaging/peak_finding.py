from dataclasses import dataclass
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import math
import os
import numpy as np
from scipy.ndimage import maximum_filter, convolve
from scipy.optimize import curve_fit

BORDER = 20
GAUSS_SIGMA = 1.27


def create_peak_kernel() -> np.ndarray:
    k = np.zeros((5, 5), dtype=np.float32)
    sumg = np.float32(0.0)
    sumg2 = np.float32(0.0)
    for i in range(-2, 3):
        for j in range(-2, 3):
            g = np.float32(np.exp(-(i * i + j * j) / (2.0 * GAUSS_SIGMA * GAUSS_SIGMA)))
            k[i + 2, j + 2] = g
            sumg += g
            sumg2 += g * g
    n = np.float32(25.0)
    return ((k - sumg / n) / (sumg2 - sumg * sumg / n)).astype(np.float32)


W5x5 = create_peak_kernel()
W5x5factor = np.float32(np.sum(W5x5 * W5x5))


@dataclass
class FitPeak:
    x: float
    y: float
    area: float
    sx: float
    sy: float
    flux: float


def image_stats(image: np.ndarray) -> Tuple[np.float32, np.float32]:
    h, w = image.shape
    stats_size = min(500, h, w)
    y0 = (h - stats_size) // 2
    x0 = (w - stats_size) // 2
    sub = image[y0:y0 + stats_size, x0:x0 + stats_size]
    median = np.median(sub)
    mad = np.median(np.abs(sub.astype(np.float32) - median))
    sigma = max(1.4826 * mad, 3.0)
    return np.float32(median), np.float32(sigma)


def sharpness_map(image: np.ndarray, hpix: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    neighbors = (
        np.roll(img, 1, axis=0) + np.roll(img, -1, axis=0)
        + np.roll(img, 1, axis=1) + np.roll(img, -1, axis=1)
        + np.roll(np.roll(img, 1, axis=0), 1, axis=1)
        + np.roll(np.roll(img, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(img, -1, axis=0), 1, axis=1)
        + np.roll(np.roll(img, -1, axis=0), -1, axis=1)
    ) / 8.0
    return (img - neighbors) / (hpix + 1.0e-6)


def find_candidates(image: np.ndarray, median: float, sigma: float, peak_sharp: float,
                    threshold_sigma: float = 10.0) -> np.ndarray:
    threshold = median + threshold_sigma * sigma
    localmax = image == maximum_filter(image, size=5, mode="nearest")
    bright = image > threshold
    hpix = convolve(image.astype(np.float32), W5x5, mode="nearest")
    sharp = sharpness_map(image, hpix)
    hthresh = sigma / W5x5factor
    mask = localmax & bright & (hpix > hthresh) & (sharp > peak_sharp) & (sharp < 1)
    mask[:BORDER, :] = False
    mask[-BORDER:, :] = False
    mask[:, :BORDER] = False
    mask[:, -BORDER:] = False
    y, x = np.where(mask)
    flux = image[y, x].astype(np.float32)
    return np.column_stack((x.astype(np.float32), y.astype(np.float32), flux))


def gaussian2d(coords, A, x0, y0, sx, sy):
    x, y = coords
    return (A * np.exp(-((x - x0) ** 2) / (2.0 * sx * sx)
                       - ((y - y0) ** 2) / (2.0 * sy * sy))).ravel()


def fit_one_peak_gauss(args) -> Optional[FitPeak]:
    image, peak_x, peak_y, median = args
    isize = jsize = 10
    i1 = int(round(peak_y)) - isize // 2
    j1 = int(round(peak_x)) - jsize // 2
    if i1 < 0 or j1 < 0 or i1 + isize >= image.shape[0] or j1 + jsize >= image.shape[1]:
        return None
    sub = image[i1:i1 + isize, j1:j1 + jsize].astype(np.float32) - median
    total_flux = float(np.sum(sub))
    X, Y = np.meshgrid(np.arange(jsize, dtype=np.float32), np.arange(isize, dtype=np.float32))
    try:
        p0 = (float(np.max(sub)), jsize / 2, isize / 2, 1.5, 1.5)
        popt, _ = curve_fit(gaussian2d, (X, Y), sub.ravel(), p0=p0, maxfev=200)
        A, x0, y0, sx, sy = popt
        sx = abs(float(sx)); sy = abs(float(sy))
        area = math.pi * float(A) * (sx * sx + sy * sy)
        return FitPeak(float(j1 + x0), float(i1 + y0), float(area), sx, sy, total_flux)
    except Exception:
        return None


def fit_one_peak_moments(args) -> Optional[FitPeak]:
    image, xpeak, ypeak, median = args
    box = 10
    half = box // 2
    x0 = int(round(xpeak)); y0 = int(round(ypeak))
    if x0 - half < 0 or y0 - half < 0 or x0 + half >= image.shape[1] or y0 + half >= image.shape[0]:
        return None
    sub = image[y0 - half:y0 + half, x0 - half:x0 + half].astype(np.float32) - median
    flux = float(np.sum(sub))
    if flux <= 0:
        return None
    ny, nx = sub.shape
    X, Y = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32))
    xc = float(np.sum(sub * X) / flux)
    yc = float(np.sum(sub * Y) / flux)
    sx = float(np.sqrt(max(np.sum(sub * (X - xc) ** 2) / flux, 0.01)))
    sy = float(np.sqrt(max(np.sum(sub * (Y - yc) ** 2) / flux, 0.01)))
    amplitude = float(np.max(sub))
    area = math.pi * amplitude * (sx * sx + sy * sy)
    return FitPeak(float(x0 - half + xc), float(y0 - half + yc), float(area), sx, sy, flux)


def peak_finder(
    image: np.ndarray,
    peak_separation: float = 10.0,
    peak_sharp: float = 0.2,
    maxsources: int = 400,
    fit_method: str = "gaussian",
    threshold_sigma: float = 10.0,
) -> List[FitPeak]:
    """Find point-like sources and return a flux-sorted list of fitted peaks."""
    if image.ndim != 2:
        raise ValueError("peak_finder expects a 2-D image")
    median, sigma = image_stats(image)
    candidates = find_candidates(image, median, sigma, peak_sharp, threshold_sigma=threshold_sigma)
    if len(candidates) == 0:
        return []
    candidates = candidates[np.argsort(candidates[:, 2])[::-1]][:maxsources * 3]
    fit_args = [(image, p[0], p[1], median) for p in candidates]
    fitter = fit_one_peak_moments if fit_method == "moments" else fit_one_peak_gauss
    workers = max(1, min(os.cpu_count() or 1, len(fit_args)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fitpeaks = list(pool.map(fitter, fit_args))
    fitpeaks = [p for p in fitpeaks if p is not None and p.area > 100.0]
    fitpeaks.sort(key=lambda p: p.flux, reverse=True)
    final = []  # type: List[FitPeak]
    r2max = peak_separation * peak_separation
    for peak in fitpeaks:
        if all((peak.x - old.x) ** 2 + (peak.y - old.y) ** 2 >= r2max for old in final):
            final.append(peak)
        if len(final) >= maxsources:
            break
    return final
