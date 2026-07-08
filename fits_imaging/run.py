"""ImagingRun: analysis object for one observing folder."""

from pathlib import Path

import pandas as pd

from .analysis import analyze_image


class ImagingRun:
    """Represent one folder of FITS images."""

    def __init__(self, folder, extensions=(".fits", ".fit")):
        self.folder = Path(folder).expanduser()
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.files = self._find_files()
        self.results = []
        self.summary = pd.DataFrame()

    def _find_files(self):
        return sorted(
            p for p in self.folder.iterdir()
            if p.is_file() and p.suffix.lower() in self.extensions
        )

    def refresh(self):
        """Refresh file list from disk."""
        self.files = self._find_files()
        return self.files

    def analyze_all(self, config, max_files=None):
        """Analyze all files in this run."""
        files = self.files if max_files is None else self.files[:max_files]

        self.results = []
        rows = []

        for i, path in enumerate(files):
            print(f"[{i + 1}/{len(files)}] {path.name}")

            try:
                result = analyze_image(path, config)
                self.results.append(result)
                rows.append(self._summary_row(result, path))

            except Exception as e:
                rows.append({
                    "file": path.name,
                    "path": str(path),
                    "error": str(e),
                })

        self.summary = pd.DataFrame(rows)
        return self.summary

    def _summary_row(self, result, path):
        stats = result.stats

        return {
            "file": path.name,
            "path": str(path),
            "object": result.object_name,
            "camera": result.camera,
            "exposure_sec": result.exposure_sec,
            "nstars": result.nstars,
            "median_fwhm_pixels": result.median_fwhm_pixels,
            "median_background": result.median_background,
            "background_sigma": result.background_sigma,
            "brightest_flux": result.brightest_flux,
            "peak_find_time_sec": stats.get("peak_find_time_sec"),
            "stats_region": stats.get("stats_region"),
        }

    def export_summary(self, path=None):
        """Export summary table to CSV."""
        if path is None:
            path = self.folder / "imaging_run_summary.csv"

        path = Path(path)
        self.summary.to_csv(path, index=False)
        return path

    def show(self, index, config=None, **report_kwargs):
        """Show report for analyzed result by index."""
        result = self.results[index]
        return result.report(config=config, **report_kwargs)

    def best_by_fwhm(self, n=10):
        """Return rows with smallest median FWHM."""
        if self.summary.empty:
            return self.summary

        if "median_fwhm_pixels" not in self.summary:
            return self.summary

        return self.summary.sort_values("median_fwhm_pixels").head(n)

    def best_by_stars(self, n=10):
        """Return rows with most detected stars."""
        if self.summary.empty:
            return self.summary

        if "nstars" not in self.summary:
            return self.summary

        return self.summary.sort_values("nstars", ascending=False).head(n)
