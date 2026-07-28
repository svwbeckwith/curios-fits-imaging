import time
from pathlib import Path

import numpy as np
from .fits_io import read_fits_image
from .photometry import peaks_to_array
from .peak_finding import peak_finder
from .coordinates import parallactic_pa_from_altaz
from .regions import statistics_image
from .models import ImagingResults


SOURCE_DETECTION_MODE = "source_detection"
STATISTICS_MODE = "statistics"
FOCUS_MODE = "focus"
ROTATION_BLUR_MODE = "rotation_blur"

ANALYSIS_MODES = (
    SOURCE_DETECTION_MODE,
    STATISTICS_MODE,
    FOCUS_MODE,
    ROTATION_BLUR_MODE,
)


MODE_ALIASES = {
    "source": SOURCE_DETECTION_MODE,
    "normal": SOURCE_DETECTION_MODE,
    "normal_source_detection": SOURCE_DETECTION_MODE,
    "source_detection": SOURCE_DETECTION_MODE,
    "statistics": STATISTICS_MODE,
    "stats": STATISTICS_MODE,
    "image_statistics": STATISTICS_MODE,
    "focus": FOCUS_MODE,
    "bahtinov": FOCUS_MODE,
    "focus_bahtinov": FOCUS_MODE,
    "rotation": ROTATION_BLUR_MODE,
    "blur": ROTATION_BLUR_MODE,
    "rotation_blur": ROTATION_BLUR_MODE,
    "rotation_blur_diagnostics": ROTATION_BLUR_MODE,
}


def normalize_analysis_mode(mode=None):
    """Return the canonical analysis mode name."""
    if mode is None:
        return SOURCE_DETECTION_MODE

    key = str(mode).strip().lower().replace("-", "_").replace(" ", "_")

    try:
        return MODE_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(ANALYSIS_MODES)
        raise ValueError(f"Unknown analysis mode {mode!r}. Valid modes: {valid}") from exc


def _image2d(record):
    image = record.data[0] if record.data.ndim == 3 else record.data
    return np.asarray(image)


def _basic_stats(image):
    finite = image[np.isfinite(image)]

    if finite.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _region_stats(image, config):
    stats_img = statistics_image(image, config)
    finite = stats_img[np.isfinite(stats_img)]

    if finite.size == 0:
        return {
            "stats_region": getattr(config, "stats_region", "full"),
            "region_mean": np.nan,
            "region_median": np.nan,
            "region_std": np.nan,
        }

    return {
        "stats_region": getattr(config, "stats_region", "full"),
        "region_mean": float(np.mean(finite)),
        "region_median": float(np.median(finite)),
        "region_std": float(np.std(finite)),
    }


def _pa_calc(record):
    return parallactic_pa_from_altaz(record.alt_deg, record.az_deg) if record.alt_deg or record.az_deg else 0.0


def _empty_peaks():
    return np.empty((0, 6), dtype=float)


def _focus_metrics(image):
    image = np.asarray(image, dtype=float)
    finite = image[np.isfinite(image)]

    if finite.size == 0:
        return {
            "focus_peak_value": np.nan,
            "focus_peak_x": np.nan,
            "focus_peak_y": np.nan,
            "focus_sharpness": np.nan,
        }

    yy, xx = np.indices(image.shape)
    safe = np.where(np.isfinite(image), image, np.nan)
    peak_index = np.nanargmax(safe)
    peak_y, peak_x = np.unravel_index(peak_index, image.shape)
    gy, gx = np.gradient(np.nan_to_num(image, nan=float(np.nanmedian(finite))))
    sharpness = float(np.mean(gx * gx + gy * gy))

    return {
        "focus_peak_value": float(image[peak_y, peak_x]),
        "focus_peak_x": float(xx[peak_y, peak_x]),
        "focus_peak_y": float(yy[peak_y, peak_x]),
        "focus_sharpness": sharpness,
    }


def _rotation_blur_metrics(image):
    image = np.asarray(image, dtype=float)
    finite = image[np.isfinite(image)]

    if finite.size == 0:
        return {
            "gradient_anisotropy": np.nan,
            "gradient_angle_deg": np.nan,
            "gradient_strength": np.nan,
        }

    filled = np.nan_to_num(image, nan=float(np.nanmedian(finite)))
    gy, gx = np.gradient(filled)
    gxx = float(np.mean(gx * gx))
    gyy = float(np.mean(gy * gy))
    gxy = float(np.mean(gx * gy))
    trace = gxx + gyy
    delta = np.sqrt((gxx - gyy) ** 2 + 4.0 * gxy * gxy)
    major = 0.5 * (trace + delta)
    minor = 0.5 * (trace - delta)
    anisotropy = np.nan if major <= 0 else 1.0 - max(minor, 0.0) / major
    angle = 0.5 * np.degrees(np.arctan2(2.0 * gxy, gxx - gyy))

    return {
        "gradient_anisotropy": float(anisotropy),
        "gradient_angle_deg": float(angle),
        "gradient_strength": float(np.sqrt(trace)),
    }


def _stats_result(image_path, config, mode, extra_stats=None):
    t0 = time.time()
    record = read_fits_image(image_path)
    image = _image2d(record)
    elapsed = time.time() - t0

    stats = _basic_stats(image)
    stats.update(_region_stats(image, config))
    stats.update(
        {
            "analysis_mode": mode,
            "elapsed_sec": elapsed,
            "npeaks": 0,
            "peak_find_time_sec": 0.0,
            "pa_calc_deg": _pa_calc(record),
        }
    )

    if extra_stats is not None:
        stats.update(extra_stats(image))

    return ImagingResults(
        image_path=Path(image_path),
        image=image,
        convolved_image=image,
        record=record,
        peaks=_empty_peaks(),
        stats=stats,
        header=getattr(record, "header", None),
    )

def analyze_fits_file(path, peak_separation=10.0, peak_sharp=0.2, maxsources=400, fit_method="gaussian"):
    """Read one FITS file and run peak finding. Returns (record, image2d, peaks_array)."""
    record = read_fits_image(path)
    image = record.data[0] if record.data.ndim == 3 else record.data
    t0 = time.time()
    peaks = peak_finder(image, peak_separation=peak_separation, peak_sharp=peak_sharp,
                        maxsources=maxsources, fit_method=fit_method)
    elapsed = time.time() - t0
    peaks_array = peaks_to_array(peaks)
    stats = {
        "mean": float(np.average(image)),
        "median": float(np.median(image)),
        "std": float(np.std(image)),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "elapsed_sec": elapsed,
        "npeaks": int(len(peaks_array)),
        "pa_calc_deg": parallactic_pa_from_altaz(record.alt_deg, record.az_deg) if record.alt_deg or record.az_deg else 0.0,
    }
    return record, image, peaks_array, stats


def analyze_image(image_path, config, mode=None):
    """Analyze one FITS image and return an ImagingResults object."""
    mode = normalize_analysis_mode(mode)

    if mode == STATISTICS_MODE:
        return _stats_result(image_path, config, mode)

    if mode == FOCUS_MODE:
        return _stats_result(image_path, config, mode, extra_stats=_focus_metrics)

    if mode == ROTATION_BLUR_MODE:
        return _stats_result(image_path, config, mode, extra_stats=_rotation_blur_metrics)

    t0 = time.time()

    result = analyze_fits_file(
        image_path,
        peak_separation=config.peak_separation,
        peak_sharp=getattr(config, "peak_sharp", 0.0),
        maxsources=config.max_peaks,
    )

    elapsed = time.time() - t0

    if len(result) == 5:
        record, image, imagec, peaksarray, stats = result
    elif len(result) == 4:
        record, image, peaksarray, stats = result
        imagec = image
    else:
        raise ValueError(f"analyze_fits_file returned {len(result)} values; expected 4 or 5")

    stats = dict(stats)
    stats["analysis_mode"] = mode
    stats["peak_find_time_sec"] = elapsed
    stats["npeaks"] = len(peaksarray)
    stats.update(_region_stats(imagec, config))

    return ImagingResults(
        image_path=Path(image_path),
        image=image,
        convolved_image=imagec,
        record=record,
        peaks=peaksarray,
        stats=stats,
        header=getattr(record, "header", None),
    )
