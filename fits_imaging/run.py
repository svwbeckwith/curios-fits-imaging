"""ImagingRun: analysis object for one observing folder."""

import os
from pathlib import Path

from astropy.io import fits
import pandas as pd

from .analysis import analyze_image
from .fits_io import _header_float, exposure_info, parse_sum_frame_count


class ImagingRun:
    """Represent one folder of FITS images."""

    def __init__(
        self,
        folder,
        extensions=(".fits", ".fit", ".fits.fz", ".fit.fz"),
        read_headers=False,
    ):
        self.folder = Path(folder).expanduser()
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.read_headers = read_headers
        self.files = self._find_files()
        self.file_table = self._build_file_table(read_headers=read_headers)
        self.results = []
        self.summary = pd.DataFrame()

    def _find_files(self):
        files = []
        with os.scandir(self.folder) as entries:
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                if entry.name.lower().endswith(self.extensions):
                    files.append(self.folder / entry.name)
        return sorted(files)

    def refresh(self):
        """Refresh file list from disk."""
        self.files = self._find_files()
        self.file_table = self._build_file_table(read_headers=self.read_headers)
        return self.files

    def _build_file_table(self, read_headers=False):
        rows = []
        for path in self.files:
            rows.append(self._file_row(path, read_headers=read_headers))
        return pd.DataFrame(rows)

    def _file_row(self, path, read_headers=False):
        filename_frame_count = parse_sum_frame_count(path)
        row = {
            "file": path.name,
            "path": str(path),
            "kind": classify_image_path(path),
            "include": True,
            "object": "",
            "camera": "",
            "exposure_sec": None,
            "frame_count": filename_frame_count or 1,
            "total_exposure_sec": None,
            "exposure_source": "filename_sum_count" if filename_frame_count else "single_frame",
            "dateobs": "",
            "metadata_loaded": False,
            "error": "",
        }

        if not read_headers:
            return row

        try:
            metadata = read_fits_metadata(path)
            row.update(
                {
                    "kind": classify_image_metadata(path, metadata),
                    "object": metadata["object"],
                    "camera": metadata["camera"],
                    "exposure_sec": metadata["exposure_sec"],
                    "frame_count": metadata["frame_count"],
                    "total_exposure_sec": metadata["total_exposure_sec"],
                    "exposure_source": metadata["exposure_source"],
                    "dateobs": metadata["dateobs"],
                    "metadata_loaded": True,
                }
            )
        except Exception as exc:
            row["error"] = str(exc)

        return row

    def load_file_metadata(self):
        """Read FITS headers and update file_table metadata columns."""
        rows = []
        for path in self.files:
            rows.append(self._file_row(path, read_headers=True))
        self.file_table = pd.DataFrame(rows)
        self.read_headers = True
        return self.file_table

    def kinds(self):
        """Return image kinds present in this run."""
        if self.file_table.empty or "kind" not in self.file_table:
            return []
        return sorted(self.file_table["kind"].dropna().unique().tolist())

    def files_for(self, kind=None, included_only=True):
        """Return files matching a kind.

        Use kind=None for all files. Use kind="science" to include both
        science_single and science_sum images.
        """
        table = self.file_table

        if table.empty:
            return []

        if included_only and "include" in table:
            table = table[table["include"]]

        if kind is not None:
            if kind == "science":
                table = table[table["kind"].isin(["science_single", "science_sum"])]
            else:
                table = table[table["kind"] == kind]

        return [Path(p) for p in table["path"].tolist()]

    def table_for(self, kind=None, included_only=True):
        """Return file_table rows matching a kind with a fresh selection index."""
        table = self.file_table

        if table.empty:
            return table

        if included_only and "include" in table:
            table = table[table["include"]]

        if kind is not None:
            if kind == "science":
                table = table[table["kind"].isin(["science_single", "science_sum"])]
            else:
                table = table[table["kind"] == kind]

        table = table.reset_index(drop=True).copy()
        table.insert(0, "index", table.index)
        return table

    def preview(self, kind=None, n=25):
        """Return a compact notebook-friendly preview of the file table."""
        columns = ["index", "file", "kind", "frame_count", "total_exposure_sec", "include"]
        return self.table_for(kind=kind).reindex(columns=columns).head(n)

    def analyze_all(self, config, kind=None, mode=None, max_files=None):
        """Analyze files in this run, optionally filtered by image kind."""
        files = self.files_for(kind=kind)
        if max_files is not None:
            files = files[:max_files]

        self.results = []
        rows = []

        for i, path in enumerate(files):
            print(f"[{i + 1}/{len(files)}] {path.name}")

            try:
                analysis_mode = mode or analysis_mode_for_kind(self._kind_for_path(path))
                result = analyze_image(path, config, mode=analysis_mode)
                self.results.append(result)
                rows.append(self._summary_row(result, path))

            except Exception as e:
                rows.append({
                    "file": path.name,
                    "path": str(path),
                    "kind": self._kind_for_path(path),
                    "error": str(e),
                })

        self.summary = pd.DataFrame(rows)
        return self.summary

    def _summary_row(self, result, path):
        stats = result.stats

        return {
            "file": path.name,
            "path": str(path),
            "kind": self._kind_for_path(path),
            "analysis_mode": stats.get("analysis_mode"),
            "object": result.object_name,
            "camera": result.camera,
            "exposure_sec": result.exposure_sec,
            "frame_count": result.frame_count,
            "total_exposure_sec": result.total_exposure_sec,
            "photometry_exposure_sec": result.photometry_exposure_sec,
            "exposure_source": result.exposure_source,
            "nstars": result.nstars,
            "median_fwhm_pixels": result.median_fwhm_pixels,
            "median_background": result.median_background,
            "background_sigma": result.background_sigma,
            "brightest_flux": result.brightest_flux,
            "peak_find_time_sec": stats.get("peak_find_time_sec"),
            "stats_region": stats.get("stats_region"),
            "focus_sharpness": stats.get("focus_sharpness"),
            "gradient_anisotropy": stats.get("gradient_anisotropy"),
            "gradient_angle_deg": stats.get("gradient_angle_deg"),
        }

    def _kind_for_path(self, path):
        if self.file_table.empty:
            return classify_image_path(path)

        matches = self.file_table[self.file_table["path"] == str(path)]
        if matches.empty:
            return classify_image_path(path)

        return matches.iloc[0]["kind"]

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


def read_fits_metadata(path):
    """Read lightweight FITS metadata without loading image data."""
    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header.copy()
        for hdu in hdul:
            if hdu.header.get("NAXIS", 0) > 0:
                header.extend(hdu.header, update=True)
                break

    exposure_sec = _header_float(header, ("EXPTIME", "IEXP", "EXPOSURE"), 0.0)
    exposure = exposure_info(path, header, exposure_sec)

    return {
        "object": str(header.get("ID", header.get("OBJECT", ""))),
        "camera": str(header.get("INSTRUME", header.get("CAMERA", ""))),
        "exposure_sec": exposure_sec,
        "frame_count": exposure["frame_count"],
        "total_exposure_sec": exposure["total_exposure_sec"],
        "exposure_source": exposure["exposure_source"],
        "dateobs": str(header.get("DATEOBS", header.get("DATE-OBS", header.get("DATE", "")))),
        "header": header,
    }


def classify_image_metadata(path, metadata):
    """Classify a FITS image using common header values and filename hints."""
    header = metadata.get("header", {})
    values = [
        Path(path).name,
        metadata.get("object", ""),
        header.get("IMAGETYP", ""),
        header.get("IMAGETYPE", ""),
        header.get("FRAME", ""),
        header.get("FRAMETYP", ""),
        header.get("CALSTAT", ""),
    ]
    return classify_image_text(" ".join(str(value) for value in values if value is not None))


def classify_image_path(path):
    """Classify a FITS image from its path when headers are not available."""
    return classify_image_text(Path(path).name)


def classify_image_text(text):
    """Return a conservative image-kind label from filename/header text."""
    normalized = text.lower().replace("-", "_")

    if any(token in normalized for token in ("bahtinov", "focus", "focusing")):
        return "focus"

    if any(token in normalized for token in ("dark", "bias")):
        return "dark"

    if "flat" in normalized:
        return "flat"

    if any(token in normalized for token in ("rotation", "rotating", "blur", "drift", "trail")):
        return "rotation_blur"

    if "_sum" in normalized or "stack" in normalized or "stacked" in normalized:
        return "science_sum"

    return "science_single"


def analysis_mode_for_kind(kind):
    """Return the default analysis mode for an image kind."""
    if kind in ("science_single", "science_sum"):
        return "source_detection"

    if kind == "focus":
        return "focus"

    if kind == "rotation_blur":
        return "rotation_blur"

    if kind in ("dark", "flat"):
        return "statistics"

    return "source_detection"
