"""Image display contrast scaling utilities."""

import numpy as np


def finite_sample(image, max_points=1_000_000):
    """Return finite image values, sampled if the image is very large."""
    arr = np.asarray(image).ravel()
    arr = arr[np.isfinite(arr)]

    if arr.size > max_points:
        rng = np.random.default_rng(12345)
        arr = arr[rng.choice(arr.size, size=max_points, replace=False)]

    return arr


def percentile_limits(image, percentiles=(1.0, 99.7), max_points=1_000_000):
    arr = finite_sample(image, max_points=max_points)
    return tuple(np.percentile(arr, percentiles))


def sigma_limits(image, nsigma=5.0, nsigma_low=None, nsigma_high=None, max_points=1_000_000):
    """Return median-based sigma display limits.

    Default symmetric form:
        median ± nsigma*sigma

    Asymmetric form:
        median - nsigma_low*sigma
        median + nsigma_high*sigma
    """
    arr = finite_sample(image, max_points=max_points)

    med = np.median(arr)
    sig = 1.4826 * np.median(np.abs(arr - med))

    if nsigma_low is None:
        nsigma_low = nsigma

    if nsigma_high is None:
        nsigma_high = nsigma

    return med - nsigma_low * sig, med + nsigma_high * sig


def zscale_limits(image, contrast=0.25, max_points=600_000):
    """Approximate DS9-style zscale display limits."""
    arr = finite_sample(image, max_points=max_points)

    if arr.size < 10:
        return float(np.min(arr)), float(np.max(arr))

    arr = np.sort(arr)
    n = arr.size
    x = np.arange(n)

    # Fit only central 80% to reduce effect of stars/cosmic rays.
    lo = int(0.10 * n)
    hi = int(0.90 * n)

    coeff = np.polyfit(x[lo:hi], arr[lo:hi], 1)
    slope, intercept = coeff

    center_index = n // 2
    center_value = arr[center_index]

    z1 = center_value + (0 - center_index) * slope / contrast
    z2 = center_value + (n - 1 - center_index) * slope / contrast

    amin = float(np.percentile(arr, 0.1))
    amax = float(np.percentile(arr, 99.9))

    return max(float(z1), amin), min(float(z2), amax)


def display_limits(image, config=None, method=None):
    """Return vmin, vmax using config or explicit method."""
    method = method or getattr(config, "contrast_method", "zscale")

    if method == "zscale":
        return zscale_limits(image)

    if method == "percentile":
        percentiles = getattr(config, "contrast_percentiles", (1.0, 99.7))
        return percentile_limits(image, percentiles)

    if method == "sigma":
        nsigma = getattr(config, "contrast_sigma", 5.0)
        nsigma_low = getattr(config, "contrast_sigma_low", nsigma)
        nsigma_high = getattr(config, "contrast_sigma_high", nsigma)

        return sigma_limits(
            image,
            nsigma=nsigma,
            nsigma_low=nsigma_low,
            nsigma_high=nsigma_high,
        )

    if method == "manual":
        vmin = getattr(config, "manual_vmin", None)
        vmax = getattr(config, "manual_vmax", None)
        if vmin is None or vmax is None:
            raise ValueError("manual contrast requires manual_vmin and manual_vmax")
        return vmin, vmax

    raise ValueError(f"Unknown contrast method: {method}")

def apply_stretch(image, vmin=None, vmax=None, stretch="linear"):
    """Return display-scaled image after optional linear/sqrt/log stretch."""
    arr = np.asarray(image, dtype=float)

    if vmin is None:
        vmin = np.nanmin(arr)
    if vmax is None:
        vmax = np.nanmax(arr)

    scaled = (arr - vmin) / max(vmax - vmin, 1e-12)
    scaled = np.clip(scaled, 0.0, 1.0)

    if stretch == "linear":
        return scaled
    if stretch == "sqrt":
        return np.sqrt(scaled)
    if stretch == "log":
        return np.log1p(1000.0 * scaled) / np.log1p(1000.0)

    raise ValueError(f"Unknown stretch: {stretch}")
