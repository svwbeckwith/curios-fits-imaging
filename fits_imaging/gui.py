"""PySide6 desktop workbench for FITS imaging."""

from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

from .analysis import analyze_image, normalize_analysis_mode
from .config import ImagingConfig
from .contrast import apply_stretch, display_limits, percentile_limits
from .fits_io import read_fits_image
from .photometry import counts_to_mag
from .run import ImagingRun


ANALYSIS_MODE_LABELS = {
    "Auto": None,
    "Normal source detection": "source_detection",
    "Image statistics only": "statistics",
    "Focus/Bahtinov analysis": "focus",
    "Rotation/blur diagnostics": "rotation_blur",
}

MAX_DISPLAY_PIXELS = 2_000_000
DEFAULT_CUTOUT_SIZE = 81
MIN_CUTOUT_SIZE = 49
MAX_CUTOUT_SIZE = 1001
IMAGE_ORIGIN = "upper"


class MissingGuiDependencyError(RuntimeError):
    """Raised when GUI dependencies are not installed."""


def _import_qt():
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSpinBox,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise MissingGuiDependencyError(
            "PySide6 is required for the GUI. Install it with: "
            "python -m pip install -e '.[gui]'"
        ) from exc

    return {
        "Qt": Qt,
        "QApplication": QApplication,
        "QCheckBox": QCheckBox,
        "QComboBox": QComboBox,
        "QDoubleSpinBox": QDoubleSpinBox,
        "QFileDialog": QFileDialog,
        "QGridLayout": QGridLayout,
        "QHBoxLayout": QHBoxLayout,
        "QLabel": QLabel,
        "QLineEdit": QLineEdit,
        "QMainWindow": QMainWindow,
        "QMessageBox": QMessageBox,
        "QPushButton": QPushButton,
        "QSpinBox": QSpinBox,
        "QTableWidget": QTableWidget,
        "QTableWidgetItem": QTableWidgetItem,
        "QVBoxLayout": QVBoxLayout,
        "QWidget": QWidget,
        "FigureCanvasQTAgg": FigureCanvasQTAgg,
        "Figure": Figure,
    }


def _table_item(qt, value):
    item = qt["QTableWidgetItem"]("" if pd.isna(value) else str(value))
    item.setFlags(item.flags() & ~qt["Qt"].ItemFlag.ItemIsEditable)
    return item


def _display_sample(image, max_pixels=MAX_DISPLAY_PIXELS):
    """Return a strided image view for responsive GUI display."""
    image = np.asarray(image)
    pixels = image.shape[0] * image.shape[1]

    if pixels <= max_pixels:
        return image, 1

    stride = int(np.ceil(np.sqrt(pixels / max_pixels)))
    return image[::stride, ::stride], stride


def peak_rows(result, config=None):
    """Return GUI table rows for detected peaks."""
    zero_mag_counts = getattr(config, "zero_mag_counts", 1.0)
    rows = []

    for index, peak in enumerate(result.peak_array):
        flux = float(peak[5]) if len(peak) > 5 else np.nan
        mag = counts_to_mag(
            flux,
            exposure=result.photometry_exposure_sec,
            zero_mag_counts=zero_mag_counts,
        )
        rows.append(
            {
                "peak": index,
                "x": float(peak[0]),
                "y": float(peak[1]),
                "flux": flux,
                "mag": mag,
                "sx": float(peak[3]) if len(peak) > 3 else np.nan,
                "sy": float(peak[4]) if len(peak) > 4 else np.nan,
                "fwhm": 2.3548 * 0.5 * (float(peak[3]) + float(peak[4])) if len(peak) > 4 else np.nan,
            }
        )

    return rows


def cutout_half_size(cutout_size=DEFAULT_CUTOUT_SIZE):
    """Return a centered half-size from a requested full cutout size."""
    cutout_size = int(cutout_size)
    if cutout_size < MIN_CUTOUT_SIZE:
        cutout_size = MIN_CUTOUT_SIZE
    return cutout_size // 2


def peak_cutout(image, peak, half_size=None, cutout_size=DEFAULT_CUTOUT_SIZE):
    """Return cutout bounds and local peak coordinates."""
    image = np.asarray(image)
    x = float(peak[0])
    y = float(peak[1])
    ny, nx = image.shape
    ix = int(round(x))
    iy = int(round(y))
    if half_size is None:
        half_size = cutout_half_size(cutout_size)
    x0 = max(ix - half_size, 0)
    x1 = min(ix + half_size + 1, nx)
    y0 = max(iy - half_size, 0)
    y1 = min(iy + half_size + 1, ny)

    return {
        "image": image[y0:y1, x0:x1],
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "local_x": x - x0,
        "local_y": y - y0,
    }


def peak_display_limits(cutout_image, percentiles=(0.5, 99.5)):
    """Return local display limits for source cutouts."""
    image = np.asarray(cutout_image, dtype=float)
    finite = image[np.isfinite(image)]

    if finite.size == 0:
        return 0.0, 1.0

    vmin, vmax = percentile_limits(finite, percentiles=percentiles, max_points=finite.size)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        center = float(np.nanmedian(finite))
        return center - 1.0, center + 1.0

    return float(vmin), float(vmax)


def cutout_extent(cutout):
    """Return imshow extent for absolute image coordinates with row index increasing downward."""
    return (cutout["x0"] - 0.5, cutout["x1"] - 0.5, cutout["y1"] - 0.5, cutout["y0"] - 0.5)


def create_app_class(qt):
    Qt = qt["Qt"]
    QCheckBox = qt["QCheckBox"]
    QComboBox = qt["QComboBox"]
    QDoubleSpinBox = qt["QDoubleSpinBox"]
    QFileDialog = qt["QFileDialog"]
    QGridLayout = qt["QGridLayout"]
    QHBoxLayout = qt["QHBoxLayout"]
    QLabel = qt["QLabel"]
    QLineEdit = qt["QLineEdit"]
    QMainWindow = qt["QMainWindow"]
    QMessageBox = qt["QMessageBox"]
    QPushButton = qt["QPushButton"]
    QSpinBox = qt["QSpinBox"]
    QTableWidget = qt["QTableWidget"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]
    FigureCanvasQTAgg = qt["FigureCanvasQTAgg"]
    Figure = qt["Figure"]

    class PeakWindow(QMainWindow):
        """Separate source-inspection window."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Peak Inspector")
            self.resize(640, 640)
            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)
            self.figure = Figure(figsize=(6, 6))
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.details = QLabel("Select a peak after running source detection.")
            self.details.setWordWrap(True)
            layout.addWidget(self.canvas, 1)
            layout.addWidget(self.details)

        def show_peak(self, result, peak_index, config, cutout_size=DEFAULT_CUTOUT_SIZE):
            peaks = result.peak_array
            if peak_index < 0 or peak_index >= len(peaks):
                return

            peak = peaks[peak_index]
            cutout = peak_cutout(result.display_image, peak, cutout_size=cutout_size)
            rows = peak_rows(result, config=config)
            row = rows[peak_index]

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            vmin, vmax = peak_display_limits(cutout["image"])
            display_image = apply_stretch(
                cutout["image"],
                vmin=vmin,
                vmax=vmax,
                stretch=getattr(config, "stretch", "linear"),
            )
            ax.imshow(
                display_image,
                origin=IMAGE_ORIGIN,
                vmin=0,
                vmax=1,
                cmap=getattr(config, "colormap", "viridis"),
                extent=cutout_extent(cutout),
            )
            ax.plot(
                row["x"],
                row["y"],
                marker="+",
                markersize=14,
                markeredgewidth=2,
                color="yellow",
            )
            ax.set_title(f"Peak {peak_index}: x={row['x']:.2f}, y={row['y']:.2f}")
            ax.set_xlim(cutout["x0"] - 0.5, cutout["x1"] - 0.5)
            ax.set_ylim(cutout["y1"] - 0.5, cutout["y0"] - 0.5)
            ax.set_xlabel("X pixel")
            ax.set_ylabel("Y pixel")
            ax.text(
                0.02,
                0.98,
                "flux={flux:.1f}\nmag={mag:.3f}\nsx={sx:.2f}, sy={sy:.2f}\nfwhm={fwhm:.2f}".format(**row),
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none"},
            )
            self.figure.tight_layout()
            self.canvas.draw_idle()
            self.details.setText(
                "Peak {peak}: x={x:.2f}, y={y:.2f}, flux={flux:.1f}, "
                "mag={mag:.3f}, sx={sx:.2f}, sy={sy:.2f}, fwhm={fwhm:.2f}".format(**row)
            )
            self.show()
            self.raise_()

    class ImagingWorkbench(QMainWindow):
        """Small desktop workbench for browsing and analyzing one observing folder."""

        def __init__(self):
            super().__init__()
            self.setWindowTitle("CuRIOS FITS Imaging")
            self.resize(1400, 900)

            self.folder = None
            self.run = None
            self.current_image_path = None
            self.current_result = None
            self.peak_window = None

            self.config = ImagingConfig()
            self._build_ui()

        def _build_ui(self):
            root = QWidget()
            self.setCentralWidget(root)
            main = QVBoxLayout(root)

            top = QHBoxLayout()
            self.folder_text = QLineEdit()
            self.folder_text.setReadOnly(True)
            choose_button = QPushButton("Select observing folder")
            choose_button.clicked.connect(self.choose_folder)
            top.addWidget(QLabel("Folder"))
            top.addWidget(self.folder_text, 1)
            top.addWidget(choose_button)
            main.addLayout(top)

            content = QHBoxLayout()
            main.addLayout(content, 1)

            left = QVBoxLayout()
            self.file_table = QTableWidget()
            self.file_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.file_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            self.file_table.itemSelectionChanged.connect(self.on_file_selection_changed)
            left.addWidget(QLabel("Files"))
            left.addWidget(self.file_table, 1)
            content.addLayout(left, 2)

            center = QVBoxLayout()
            self.figure = Figure(figsize=(7, 7))
            self.canvas = FigureCanvasQTAgg(self.figure)
            center.addWidget(self.canvas, 1)
            content.addLayout(center, 4)

            right = QVBoxLayout()
            right.addWidget(QLabel("Analysis controls"))

            self.kind_combo = QComboBox()
            self.kind_combo.addItems(["science", "science_single", "science_sum", "focus", "dark", "flat", "rotation_blur"])
            self.kind_combo.currentTextChanged.connect(self.populate_files)
            right.addWidget(QLabel("Image type"))
            right.addWidget(self.kind_combo)

            self.mode_combo = QComboBox()
            self.mode_combo.addItems(list(ANALYSIS_MODE_LABELS.keys()))
            right.addWidget(QLabel("Analysis mode"))
            right.addWidget(self.mode_combo)

            self.threshold_sigma = QDoubleSpinBox()
            self.threshold_sigma.setRange(0.1, 100.0)
            self.threshold_sigma.setSingleStep(0.5)
            self.threshold_sigma.setValue(self.config.threshold_sigma)
            right.addWidget(QLabel("Detection threshold"))
            right.addWidget(self.threshold_sigma)

            self.max_peaks = QSpinBox()
            self.max_peaks.setRange(1, 100000)
            self.max_peaks.setValue(self.config.max_peaks)
            right.addWidget(QLabel("Max peaks"))
            right.addWidget(self.max_peaks)

            self.peak_separation = QDoubleSpinBox()
            self.peak_separation.setRange(0.0, 500.0)
            self.peak_separation.setSingleStep(1.0)
            self.peak_separation.setValue(self.config.peak_separation)
            right.addWidget(QLabel("Peak separation"))
            right.addWidget(self.peak_separation)

            self.peak_sharp = QDoubleSpinBox()
            self.peak_sharp.setRange(-10.0, 10.0)
            self.peak_sharp.setSingleStep(0.05)
            self.peak_sharp.setValue(self.config.peak_sharp)
            right.addWidget(QLabel("Peak sharpness"))
            right.addWidget(self.peak_sharp)

            self.fit_method = QComboBox()
            self.fit_method.addItems(["gaussian", "moments"])
            self.fit_method.setCurrentText(self.config.fit_method)
            right.addWidget(QLabel("Fit method"))
            right.addWidget(self.fit_method)

            self.cutout_size = QSpinBox()
            self.cutout_size.setRange(MIN_CUTOUT_SIZE, MAX_CUTOUT_SIZE)
            self.cutout_size.setSingleStep(20)
            self.cutout_size.setValue(DEFAULT_CUTOUT_SIZE)
            self.cutout_size.setSuffix(" px")
            self.cutout_size.valueChanged.connect(self.refresh_selected_peak)
            right.addWidget(QLabel("Peak cutout size"))
            right.addWidget(self.cutout_size)

            self.show_peaks = QCheckBox("Show detected peaks")
            self.show_peaks.setChecked(True)
            right.addWidget(self.show_peaks)

            run_button = QPushButton("Run selected analysis")
            run_button.clicked.connect(self.run_selected_analysis)
            right.addWidget(run_button)
            right.addStretch(1)
            content.addLayout(right, 2)

            bottom = QVBoxLayout()
            bottom_header = QHBoxLayout()
            bottom_header.addWidget(QLabel("Results"))
            export_button = QPushButton("Export summary CSV")
            export_button.clicked.connect(self.export_summary)
            bottom_header.addWidget(export_button)
            bottom.addLayout(bottom_header)

            bottom_tables = QHBoxLayout()
            self.results_table = QTableWidget()
            bottom_tables.addWidget(self.results_table, 1)
            self.peaks_table = QTableWidget()
            self.peaks_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.peaks_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            self.peaks_table.itemSelectionChanged.connect(self.on_peak_selection_changed)
            bottom_tables.addWidget(self.peaks_table, 2)
            bottom.addLayout(bottom_tables)
            self.status = QLabel("Select an observing folder to begin.")
            bottom.addWidget(self.status)
            main.addLayout(bottom, 1)

        def choose_folder(self):
            folder = QFileDialog.getExistingDirectory(self, "Select observing folder")
            if not folder:
                return

            self.folder = Path(folder)
            self.folder_text.setText(str(self.folder))
            self.config.data_root = self.folder.parent
            self.run = ImagingRun(self.folder)
            self.status.setText(f"Found {len(self.run.files)} FITS files.")
            self.populate_files()

        def populate_files(self, *_args):
            self.file_table.clear()
            self.current_image_path = None
            self.current_result = None
            self.clear_peak_table()
            self.figure.clear()
            self.canvas.draw_idle()

            if self.run is None:
                return

            kind = self.kind_combo.currentText()
            table = self.run.table_for(kind=kind)
            columns = ["index", "file", "kind", "frame_count", "total_exposure_sec", "include"]

            self.file_table.setRowCount(len(table))
            self.file_table.setColumnCount(len(columns))
            self.file_table.setHorizontalHeaderLabels(columns)

            for row_index, row in table.reset_index(drop=True).iterrows():
                for col_index, column in enumerate(columns):
                    self.file_table.setItem(row_index, col_index, _table_item(qt, row.get(column, "")))

            self.file_table.resizeColumnsToContents()
            self.status.setText(f"Showing {len(table)} {kind} file(s).")

        def selected_path(self):
            if self.run is None:
                return None

            selected = self.file_table.selectedItems()
            if not selected:
                return None

            row = selected[0].row()
            index_item = self.file_table.item(row, 0)
            if index_item is None:
                return None

            files = self.run.files_for(kind=self.kind_combo.currentText())
            index = int(index_item.text())
            if index < 0 or index >= len(files):
                return None
            return files[index]

        def on_file_selection_changed(self):
            path = self.selected_path()
            if path is None:
                return
            self.current_image_path = path
            self.preview_image(path)

        def preview_image(self, path):
            try:
                record = read_fits_image(path)
                image = record.data[0] if record.data.ndim == 3 else record.data
                self.draw_image(image, title=path.name)
                self.status.setText(f"Previewing {path.name}")
            except Exception as exc:
                self.show_error("Could not preview image", exc)

        def draw_image(self, image, title="", peaks=None):
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            vmin, vmax = display_limits(image, config=self.config)
            image_sample, stride = _display_sample(image)
            display_image = apply_stretch(image_sample, vmin=vmin, vmax=vmax, stretch=self.config.stretch)
            ny, nx = image.shape
            ax.imshow(
                display_image,
                origin=IMAGE_ORIGIN,
                vmin=0,
                vmax=1,
                cmap=self.config.colormap,
                extent=(0, nx, ny, 0),
            )
            ax.set_title(title)
            ax.set_xlabel("X pixel")
            ax.set_ylabel("Y pixel")
            if stride > 1:
                ax.text(
                    0.01,
                    0.01,
                    f"display stride {stride}",
                    transform=ax.transAxes,
                    color="white",
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )

            if peaks is not None and len(peaks) > 0 and self.show_peaks.isChecked():
                for i, peak in enumerate(peaks):
                    ax.text(float(peak[0]), float(peak[1]), str(i), color="yellow", fontsize=8)

            self.figure.tight_layout()
            self.canvas.draw_idle()

        def update_config_from_controls(self):
            self.config.threshold_sigma = self.threshold_sigma.value()
            self.config.max_peaks = self.max_peaks.value()
            self.config.peak_separation = self.peak_separation.value()
            self.config.peak_sharp = self.peak_sharp.value()
            self.config.fit_method = self.fit_method.currentText()

        def selected_mode(self):
            mode = ANALYSIS_MODE_LABELS[self.mode_combo.currentText()]
            return normalize_analysis_mode(mode) if mode else None

        def run_selected_analysis(self):
            path = self.current_image_path or self.selected_path()
            if path is None:
                self.status.setText("Select a file before running analysis.")
                return

            self.update_config_from_controls()
            mode = self.selected_mode()

            try:
                result = analyze_image(path, self.config, mode=mode)
                self.current_result = result
                self.draw_image(result.display_image, title=result.title, peaks=result.peak_array)
                self.show_result(result)
                self.populate_peak_table(result)
                self.status.setText(f"Analysis complete: {path.name}")
            except Exception as exc:
                self.show_error("Analysis failed", exc)

        def show_result(self, result):
            rows = [
                ("file", result.image_path.name),
                ("analysis_mode", result.analysis_mode),
                ("object", result.object_name),
                ("camera", result.camera),
                ("single_exposure_sec", result.exposure_sec),
                ("frame_count", result.frame_count),
                ("total_exposure_sec", result.total_exposure_sec),
                ("nstars", result.nstars),
                ("median_fwhm_pixels", result.median_fwhm_pixels),
                ("background_median", result.median_background),
                ("background_sigma", result.background_sigma),
                ("brightest_flux", result.brightest_flux),
            ]

            self.results_table.clear()
            self.results_table.setRowCount(len(rows))
            self.results_table.setColumnCount(2)
            self.results_table.setHorizontalHeaderLabels(["Metric", "Value"])

            for row_index, (key, value) in enumerate(rows):
                self.results_table.setItem(row_index, 0, _table_item(qt, key))
                self.results_table.setItem(row_index, 1, _table_item(qt, value))

            self.results_table.resizeColumnsToContents()

        def clear_peak_table(self):
            self.peaks_table.clear()
            self.peaks_table.setRowCount(0)
            self.peaks_table.setColumnCount(0)

        def populate_peak_table(self, result):
            rows = peak_rows(result, config=self.config)
            columns = ["peak", "x", "y", "flux", "mag", "sx", "sy", "fwhm"]

            self.peaks_table.clear()
            self.peaks_table.setRowCount(len(rows))
            self.peaks_table.setColumnCount(len(columns))
            self.peaks_table.setHorizontalHeaderLabels(columns)

            for row_index, row in enumerate(rows):
                for col_index, column in enumerate(columns):
                    value = row[column]
                    if isinstance(value, float):
                        value = f"{value:.3f}"
                    self.peaks_table.setItem(row_index, col_index, _table_item(qt, value))

            self.peaks_table.resizeColumnsToContents()

        def on_peak_selection_changed(self):
            if self.current_result is None:
                return

            selected = self.peaks_table.selectedItems()
            if not selected:
                return

            peak_index = selected[0].row()
            self.show_peak_window(peak_index)

        def refresh_selected_peak(self):
            if self.current_result is None:
                return

            selected = self.peaks_table.selectedItems()
            if not selected:
                return

            self.show_peak_window(selected[0].row())

        def show_peak_window(self, peak_index):
            if self.current_result is None:
                return

            if self.peak_window is None:
                self.peak_window = PeakWindow(self)

            self.peak_window.show_peak(
                self.current_result,
                peak_index,
                self.config,
                cutout_size=self.cutout_size.value(),
            )

        def export_summary(self):
            if self.current_result is None:
                self.status.setText("Run an analysis before exporting.")
                return

            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export summary CSV",
                str((self.folder or Path.home()) / "selected_image_summary.csv"),
                "CSV files (*.csv)",
            )
            if not path:
                return

            rows = []
            for row in range(self.results_table.rowCount()):
                key = self.results_table.item(row, 0).text()
                value = self.results_table.item(row, 1).text()
                rows.append({"metric": key, "value": value})

            pd.DataFrame(rows).to_csv(path, index=False)
            self.status.setText(f"Summary written to {path}")

        def show_error(self, title, exc):
            self.status.setText(f"{title}: {exc}")
            QMessageBox.critical(self, title, f"{exc}\n\n{traceback.format_exc()}")

    return ImagingWorkbench


def main(argv=None):
    """Launch the PySide6 workbench."""
    try:
        qt = _import_qt()
    except MissingGuiDependencyError as exc:
        print(exc, file=sys.stderr)
        return 1

    QApplication = qt["QApplication"]
    app = QApplication(list(sys.argv if argv is None else argv))
    window_class = create_app_class(qt)
    window = window_class()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
