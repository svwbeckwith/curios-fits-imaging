"""Plotting utilities for FITS image analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import plot_style as style

from .contrast import display_limits, apply_stretch

def _peak_xy(peak):
    """Return x, y from either a FitPeak object or a numeric peak row."""
    if hasattr(peak, "x") and hasattr(peak, "y"):
        return float(peak.x), float(peak.y)

    arr = np.asarray(peak)
    if arr.size < 2:
        raise ValueError("Peak must have at least x and y values")

    return float(arr[0]), float(arr[1])


def plot_image_with_peaks(
    image,
    peaks_array=None,
    title="",
    save_path=None,
    flagcirc=False,
    figsize=(16, 16),
    config=None,
):
    """Plot full image with optional peak labels.

    peaks_array may be either:
        - a list of FitPeak objects with .x and .y attributes
        - a NumPy array with columns [x, y, area, sx, sy, flux]
    """
    fig, ax = plt.subplots(figsize=figsize)

    vmin, vmax = display_limits(image, config=config)
    stretch = getattr(config, "stretch", "linear")
    display_image = apply_stretch(image, vmin=vmin, vmax=vmax, stretch=stretch)

    cmap = getattr(config, "colormap", "viridis")

    if getattr(config, "invert_colormap", False):
        if not cmap.endswith("_r"):
            cmap += "_r"
            
    ax.imshow(display_image, origin="lower", vmin=0, vmax=1)
    
    ax.set_title(title, fontsize=style.MAIN_TITLE_SIZE)
    ax.set_xlabel("X pixel", fontsize=style.MAIN_LABEL_SIZE)
    ax.set_ylabel("Y pixel", fontsize=style.MAIN_LABEL_SIZE)
    ax.tick_params(labelsize=style.MAIN_TICK_SIZE)

    ax.invert_yaxis()
    
    ax.minorticks_on()
    ax.grid(
        True,
        color=style.MAIN_GRID_COLOR,
        linewidth=style.MAIN_GRID_WIDTH,
        alpha=style.MAIN_GRID_ALPHA,
    )

    # Mark image center
    ax.text(
        image.shape[1] / 2,
        image.shape[0] / 2,
        "+",
        ha="center",
        va="center",
        fontsize=style.MAIN_CENTER_SIZE,
        color=style.MAIN_CENTER_COLOR,
    )

    if peaks_array is not None and len(peaks_array) > 0:
        for i, peak in enumerate(peaks_array):
            x, y = _peak_xy(peak)

            if flagcirc:
                ax.text(
                    x,
                    y,
                    "o",
                    ha="center",
                    va="center",
                    fontsize=style.CIRCLE_MARK_SIZE,
                    color=style.CIRCLE_MARK_COLOR,
                )

            ax.text(
                x,
                y,
                str(i),
                color=style.MAIN_PEAK_LABEL_COLOR,
                fontsize=style.MAIN_PEAK_LABEL_SIZE,
                ha="left",
                va="bottom",
            )

    if save_path:
        fig.savefig(Path(save_path), bbox_inches="tight", dpi=150)

    return fig, ax


def plot_peak_cutouts(
    image,
    peaks_array,
    half_size=15,
    max_peaks=25,
    figsize=(12, 12),
    nmax=None,
    spansize=None,
    pixel_arcsec=None,
    exposure_sec=None,
    zero_mag_counts=None,
    config=None,
):
    """Plot small cutouts centered on detected peaks.

    Supports both the cleaned API and the original notebook API.
    """
    if nmax is not None:
        max_peaks = nmax

    if spansize is not None:
        half_size = int(spansize // 2)

    if peaks_array is None or len(peaks_array) == 0:
        raise ValueError("No peaks provided")

    nplot = min(len(peaks_array), max_peaks)
    ncols = int(np.ceil(np.sqrt(nplot)))
    nrows = int(np.ceil(nplot / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    ny, nx = image.shape

    for i in range(nplot):
        ax = axes[i]
        x, y = _peak_xy(peaks_array[i])

        ix = int(round(x))
        iy = int(round(y))

        x0 = max(ix - half_size, 0)
        x1 = min(ix + half_size + 1, nx)
        y0 = max(iy - half_size, 0)
        y1 = min(iy + half_size + 1, ny)

        cutout = image[y0:y1, x0:x1]

        ax.imshow(cutout, origin="lower") #, cmap="gray")
        ax.set_title(f"Peak {i}", fontsize=style.CUTOUT_TITLE_SIZE)
        ax.tick_params(labelsize=style.CUTOUT_TICK_SIZE)

        cx = x - x0
        cy = y - y0

        ax.plot(
            cx,
            cy,
            marker="+",
            color=style.CUTOUT_CENTER_COLOR,
            markersize=style.CUTOUT_CENTER_SIZE,
            markeredgewidth=style.CUTOUT_CENTER_WIDTH,
        )

    for j in range(nplot, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    return fig, axes


def plot_histogram(
    values,
    bins=100,
    title="Histogram",
    xlabel="Value",
    ylabel="Count",
    figsize=(8, 6),
    max_points=5_000_000,
    percentile_clip=(0.1, 99.9),
    config=None,
):
    """Plot a histogram, sampling large images to avoid slow notebook plots."""
    arr = np.asarray(values).ravel()
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        raise ValueError("No finite values available for histogram")

    # Randomly sample very large images.
    if arr.size > max_points:
        rng = np.random.default_rng(12345)
        idx = rng.choice(arr.size, size=max_points, replace=False)
        arr = arr[idx]

    # Clip extreme outliers for a useful display range.
    if percentile_clip is not None:
        lo, hi = np.percentile(arr, percentile_clip)
        arr = arr[(arr >= lo) & (arr <= hi)]

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(arr, bins=bins)
    ax.set_title(title, fontsize=style.HIST_TITLE_SIZE)
    ax.set_xlabel(xlabel, fontsize=style.HIST_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=style.HIST_LABEL_SIZE)
    ax.tick_params(labelsize=style.HIST_TICK_SIZE)

    ax.grid(
        True,
        color=style.HIST_GRID_COLOR,
        linewidth=style.HIST_GRID_WIDTH,
        alpha=style.HIST_GRID_ALPHA,
    )

    fig.tight_layout()
    return fig, ax
