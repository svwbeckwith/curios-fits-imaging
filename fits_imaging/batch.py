"""Batch analysis tools for observing folders."""

from pathlib import Path
import pandas as pd

from .analysis import analyze_image


def analyze_folder(folder, config, extensions=(".fits", ".fit"), max_files=None):
    """Analyze all FITS files in a folder and return results plus summary table."""
    folder = Path(folder).expanduser()

    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if max_files is not None:
        files = files[:max_files]

    results = []
    rows = []

    for i, path in enumerate(files):
        print(f"[{i+1}/{len(files)}] {path.name}")

        try:
            result = analyze_image(path, config)
            results.append(result)

            rows.append({
                "file": path.name,
                "object": result.object_name,
                "camera": result.camera,
                "exposure_sec": result.exposure_sec,
                "nstars": result.nstars,
                "median_fwhm_pixels": result.median_fwhm_pixels,
                "background_median": result.median_background,
                "background_sigma": result.background_sigma,
                "peak_find_time_sec": result.stats.get("peak_find_time_sec"),
            })

        except Exception as e:
            rows.append({
                "file": path.name,
                "error": str(e),
            })

    summary = pd.DataFrame(rows)
    return results, summary
