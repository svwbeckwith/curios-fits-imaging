"""PySide6 desktop workbench for FITS imaging."""

from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

from .analysis import analyze_image, normalize_analysis_mode
from .config import ImagingConfig
from .contrast import apply_stretch, cutout_display_limits, display_limits, percentile_limits
from .fits_io import read_fits_headers, read_fits_image
from .photometry import counts_to_mag
from .run import ImagingRun


ANALYSIS_MODE_LABELS = {
    "Auto": None,
    "Normal source detection": "source_detection",
    "Image statistics only": "statistics",
    "Focus/Bahtinov analysis": "focus",
    "Rotation/blur diagnostics": "rotation_blur",
}

DISPLAY_PRESETS = ("Standard", "Bahtinov", "Percentile", "Manual")
DISPLAY_STRETCHES = ("linear", "sqrt", "log")

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


def configure_cutout_display(
    config,
    preset,
    *,
    stretch="linear",
    percentile_low=1.0,
    percentile_high=99.5,
    manual_vmin=0.0,
    manual_vmax=65535.0,
):
    """Apply GUI cutout controls to an ImagingConfig instance."""
    if preset == "Standard":
        config.use_standard_cutout_display()
    elif preset == "Bahtinov":
        config.use_bahtinov_display()
    elif preset == "Percentile":
        if not 0.0 <= percentile_low < percentile_high <= 100.0:
            raise ValueError("percentile low must be less than high (0 to 100)")
        config.use_percentile_cutout_display(percentile_low, percentile_high, stretch=stretch)
    elif preset == "Manual":
        config.use_manual_cutout_display(manual_vmin, manual_vmax, stretch=stretch)
    else:
        raise ValueError(f"Unknown cutout display preset: {preset}")

    # Standard and Bahtinov set their recommended stretch when selected. The
    # GUI may subsequently override it through the independent stretch menu.
    if preset in {"Standard", "Bahtinov"} and stretch is not None:
        config.cutout_stretch = stretch


def configure_main_display(
    config,
    preset,
    *,
    stretch="linear",
    percentile_low=1.0,
    percentile_high=99.5,
    manual_vmin=0.0,
    manual_vmax=65535.0,
):
    """Apply GUI main-image controls independently of cutout controls."""
    if preset == "Standard":
        config.use_standard_main_display()
    elif preset == "Bahtinov":
        config.use_bahtinov_main_display()
    elif preset == "Percentile":
        if not 0.0 <= percentile_low < percentile_high <= 100.0:
            raise ValueError("percentile low must be less than high (0 to 100)")
        config.use_percentile_main_display(percentile_low, percentile_high, stretch=stretch)
    elif preset == "Manual":
        config.use_manual_main_display(manual_vmin, manual_vmax, stretch=stretch)
    else:
        raise ValueError(f"Unknown main-image display preset: {preset}")

    if preset in {"Standard", "Bahtinov"} and stretch is not None:
        config.stretch = stretch


def create_app_class(qt):
    QApplication = qt["QApplication"]
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

    class HeaderWindow(QMainWindow):
        """Searchable, copyable display of all headers in one FITS file."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("FITS Header")
            self.resize(1000, 650)
            self.headers = []

            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)

            self.filename = QLabel("No FITS file selected.")
            self.filename.setWordWrap(True)
            layout.addWidget(self.filename)

            controls = QHBoxLayout()
            controls.addWidget(QLabel("Header unit"))
            self.hdu_combo = QComboBox()
            self.hdu_combo.currentIndexChanged.connect(self.populate_cards)
            controls.addWidget(self.hdu_combo, 1)
            controls.addWidget(QLabel("Find"))
            self.search_text = QLineEdit()
            self.search_text.setPlaceholderText("keyword, value, or comment")
            self.search_text.textChanged.connect(self.filter_cards)
            controls.addWidget(self.search_text, 1)
            copy_button = QPushButton("Copy selected rows")
            copy_button.clicked.connect(self.copy_selected_rows)
            controls.addWidget(copy_button)
            layout.addLayout(controls)

            self.table = QTableWidget()
            self.table.setColumnCount(4)
            self.table.setHorizontalHeaderLabels(["#", "Keyword", "Value", "Comment"])
            self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
            layout.addWidget(self.table, 1)

        def show_file(self, path):
            path = Path(path)
            self.headers = read_fits_headers(path)
            self.filename.setText(str(path))
            self.setWindowTitle(f"FITS Header — {path.name}")

            self.hdu_combo.blockSignals(True)
            self.hdu_combo.clear()
            for header in self.headers:
                name = header["name"] or "unnamed"
                self.hdu_combo.addItem(
                    f"HDU {header['index']}: {name} ({header['type']}, "
                    f"{len(header['cards'])} cards)"
                )
            self.hdu_combo.blockSignals(False)
            self.hdu_combo.setCurrentIndex(0 if self.headers else -1)
            self.populate_cards()
            self.show()
            self.raise_()

        def populate_cards(self, *_args):
            self.table.setRowCount(0)
            index = self.hdu_combo.currentIndex()
            if index < 0 or index >= len(self.headers):
                return

            cards = self.headers[index]["cards"]
            self.table.setRowCount(len(cards))
            for row, card in enumerate(cards):
                values = (card["position"], card["keyword"], card["value"], card["comment"])
                for column, value in enumerate(values):
                    self.table.setItem(row, column, _table_item(qt, value))
            self.table.resizeColumnsToContents()
            self.filter_cards()

        def filter_cards(self, *_args):
            query = self.search_text.text().strip().casefold()
            for row in range(self.table.rowCount()):
                searchable = " ".join(
                    self.table.item(row, column).text()
                    for column in range(1, self.table.columnCount())
                    if self.table.item(row, column) is not None
                ).casefold()
                self.table.setRowHidden(row, bool(query and query not in searchable))

        def copy_selected_rows(self):
            rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
            lines = []
            for row in rows:
                lines.append(
                    "\t".join(
                        self.table.item(row, column).text()
                        if self.table.item(row, column) is not None else ""
                        for column in range(self.table.columnCount())
                    )
                )
            if lines:
                QApplication.clipboard().setText("\n".join(lines))

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
            vmin, vmax = cutout_display_limits(cutout["image"], config=config)
            display_image = apply_stretch(
                cutout["image"],
                vmin=vmin,
                vmax=vmax,
                stretch=getattr(config, "cutout_stretch", "linear"),
            )
            cmap = getattr(config, "cutout_colormap", None) or getattr(config, "colormap", "viridis")
            if getattr(config, "invert_colormap", False) and not cmap.endswith("_r"):
                cmap += "_r"
            ax.imshow(
                display_image,
                origin=IMAGE_ORIGIN,
                vmin=0,
                vmax=1,
                cmap=cmap,
                extent=cutout_extent(cutout),
            )
            ax.plot(
                row["x"],
                row["y"],
                marker="+",
                markersize=22,
                markeredgewidth=4,
                color="black",
                linestyle="none",
            )
            ax.plot(
                row["x"],
                row["y"],
                marker="+",
                markersize=18,
                markeredgewidth=2.5,
                color="red",
                linestyle="none",
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

        def show_location(self, image, x, y, config, cutout_size=DEFAULT_CUTOUT_SIZE, title=""):
            """Inspect an arbitrary image location without a detected peak."""
            image = np.asarray(image)
            if image.ndim != 2 or image.size == 0:
                return

            x = float(np.clip(x, 0, image.shape[1] - 1))
            y = float(np.clip(y, 0, image.shape[0] - 1))
            cutout = peak_cutout(image, (x, y), cutout_size=cutout_size)

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            vmin, vmax = cutout_display_limits(cutout["image"], config=config)
            shown = apply_stretch(
                cutout["image"],
                vmin=vmin,
                vmax=vmax,
                stretch=getattr(config, "cutout_stretch", "linear"),
            )
            cmap = getattr(config, "cutout_colormap", None) or getattr(config, "colormap", "viridis")
            if getattr(config, "invert_colormap", False) and not cmap.endswith("_r"):
                cmap += "_r"
            ax.imshow(
                shown,
                origin=IMAGE_ORIGIN,
                vmin=0,
                vmax=1,
                cmap=cmap,
                extent=cutout_extent(cutout),
            )
            ax.plot(x, y, marker="+", markersize=22, markeredgewidth=4, color="black", linestyle="none")
            ax.plot(x, y, marker="+", markersize=18, markeredgewidth=2.5, color="cyan", linestyle="none")
            label = f"Location: x={x:.2f}, y={y:.2f}"
            ax.set_title(f"{title} — {label}" if title else label)
            ax.set_xlim(cutout["x0"] - 0.5, cutout["x1"] - 0.5)
            ax.set_ylim(cutout["y1"] - 0.5, cutout["y0"] - 0.5)
            ax.set_xlabel("X pixel")
            ax.set_ylabel("Y pixel")
            self.figure.tight_layout()
            self.canvas.draw_idle()
            self.details.setText(
                f"Manual inspection at x={x:.2f}, y={y:.2f}; "
                f"cutout {cutout['image'].shape[1]} x {cutout['image'].shape[0]} pixels."
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
            self.current_record = None
            self.current_raw_image = None
            self.current_result = None
            self.peak_window = None
            self.header_window = None
            self.current_display_image = None
            self.current_display_title = ""
            self.current_display_peaks = None
            self.manual_inspection_location = None

            self.config = ImagingConfig()
            self._build_ui()
            self.canvas.mpl_connect("button_press_event", self.on_main_image_click)

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
            file_header = QHBoxLayout()
            file_header.addWidget(QLabel("Files"))
            header_button = QPushButton("Show FITS header")
            header_button.clicked.connect(self.show_fits_header)
            file_header.addWidget(header_button)
            left.addLayout(file_header)
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

            inspect_row = QHBoxLayout()
            self.inspect_x = QDoubleSpinBox()
            self.inspect_x.setRange(0.0, 1.0e9)
            self.inspect_x.setDecimals(2)
            self.inspect_y = QDoubleSpinBox()
            self.inspect_y.setRange(0.0, 1.0e9)
            self.inspect_y.setDecimals(2)
            inspect_button = QPushButton("Inspect X/Y")
            inspect_button.clicked.connect(self.inspect_coordinates)
            inspect_row.addWidget(QLabel("X"))
            inspect_row.addWidget(self.inspect_x)
            inspect_row.addWidget(QLabel("Y"))
            inspect_row.addWidget(self.inspect_y)
            right.addLayout(inspect_row)
            right.addWidget(inspect_button)

            self.inspect_clicks = QCheckBox("Click main image to inspect")
            self.inspect_clicks.setChecked(False)
            right.addWidget(self.inspect_clicks)

            right.addWidget(QLabel("Main image brightness"))
            self.main_preset = QComboBox()
            self.main_preset.addItems(DISPLAY_PRESETS)
            right.addWidget(self.main_preset)

            main_display_row = QHBoxLayout()
            self.main_stretch = QComboBox()
            self.main_stretch.addItems(DISPLAY_STRETCHES)
            self.main_percentile_low = QDoubleSpinBox()
            self.main_percentile_low.setRange(0.0, 99.99)
            self.main_percentile_low.setDecimals(2)
            self.main_percentile_low.setValue(self.config.contrast_percentiles[0])
            self.main_percentile_high = QDoubleSpinBox()
            self.main_percentile_high.setRange(0.01, 100.0)
            self.main_percentile_high.setDecimals(2)
            self.main_percentile_high.setValue(self.config.contrast_percentiles[1])
            main_display_row.addWidget(QLabel("Stretch / percentiles"))
            main_display_row.addWidget(self.main_stretch)
            main_display_row.addWidget(self.main_percentile_low)
            main_display_row.addWidget(self.main_percentile_high)
            right.addLayout(main_display_row)

            main_manual_row = QHBoxLayout()
            self.main_manual_vmin = QDoubleSpinBox()
            self.main_manual_vmin.setRange(-1.0e12, 1.0e12)
            self.main_manual_vmin.setDecimals(3)
            self.main_manual_vmax = QDoubleSpinBox()
            self.main_manual_vmax.setRange(-1.0e12, 1.0e12)
            self.main_manual_vmax.setDecimals(3)
            self.main_manual_vmax.setValue(65535.0)
            main_manual_row.addWidget(QLabel("Main min/max"))
            main_manual_row.addWidget(self.main_manual_vmin)
            main_manual_row.addWidget(self.main_manual_vmax)
            right.addLayout(main_manual_row)

            self.main_preset.currentTextChanged.connect(self.on_main_preset_changed)
            self.main_stretch.currentTextChanged.connect(self.update_main_display_from_controls)
            self.main_percentile_low.valueChanged.connect(self.update_main_display_from_controls)
            self.main_percentile_high.valueChanged.connect(self.update_main_display_from_controls)
            self.main_manual_vmin.valueChanged.connect(self.update_main_display_from_controls)
            self.main_manual_vmax.valueChanged.connect(self.update_main_display_from_controls)
            self.update_main_control_state()

            right.addWidget(QLabel("Cutout brightness"))
            self.cutout_preset = QComboBox()
            self.cutout_preset.addItems(DISPLAY_PRESETS)
            right.addWidget(self.cutout_preset)

            right.addWidget(QLabel("Cutout stretch"))
            self.cutout_stretch = QComboBox()
            self.cutout_stretch.addItems(DISPLAY_STRETCHES)
            self.cutout_stretch.setCurrentText(self.config.cutout_stretch)
            right.addWidget(self.cutout_stretch)

            percentile_row = QHBoxLayout()
            self.cutout_percentile_low = QDoubleSpinBox()
            self.cutout_percentile_low.setRange(0.0, 99.99)
            self.cutout_percentile_low.setDecimals(2)
            self.cutout_percentile_low.setValue(self.config.cutout_contrast_percentiles[0])
            self.cutout_percentile_high = QDoubleSpinBox()
            self.cutout_percentile_high.setRange(0.01, 100.0)
            self.cutout_percentile_high.setDecimals(2)
            self.cutout_percentile_high.setValue(self.config.cutout_contrast_percentiles[1])
            percentile_row.addWidget(QLabel("Percentiles"))
            percentile_row.addWidget(self.cutout_percentile_low)
            percentile_row.addWidget(self.cutout_percentile_high)
            right.addLayout(percentile_row)

            manual_row = QHBoxLayout()
            self.cutout_manual_vmin = QDoubleSpinBox()
            self.cutout_manual_vmin.setRange(-1.0e12, 1.0e12)
            self.cutout_manual_vmin.setDecimals(3)
            self.cutout_manual_vmin.setValue(0.0)
            self.cutout_manual_vmax = QDoubleSpinBox()
            self.cutout_manual_vmax.setRange(-1.0e12, 1.0e12)
            self.cutout_manual_vmax.setDecimals(3)
            self.cutout_manual_vmax.setValue(65535.0)
            manual_row.addWidget(QLabel("Manual min/max"))
            manual_row.addWidget(self.cutout_manual_vmin)
            manual_row.addWidget(self.cutout_manual_vmax)
            right.addLayout(manual_row)

            self.cutout_preset.currentTextChanged.connect(self.on_cutout_preset_changed)
            self.cutout_stretch.currentTextChanged.connect(self.update_cutout_display_from_controls)
            self.cutout_percentile_low.valueChanged.connect(self.update_cutout_display_from_controls)
            self.cutout_percentile_high.valueChanged.connect(self.update_cutout_display_from_controls)
            self.cutout_manual_vmin.valueChanged.connect(self.update_cutout_display_from_controls)
            self.cutout_manual_vmax.valueChanged.connect(self.update_cutout_display_from_controls)
            self.update_cutout_control_state()

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
            self.current_record = None
            self.current_raw_image = None
            self.current_result = None
            self.current_display_image = None
            self.current_display_title = ""
            self.current_display_peaks = None
            self.manual_inspection_location = None
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

        def show_fits_header(self):
            """Open the selected file's FITS metadata in a separate window."""
            path = self.current_image_path or self.selected_path()
            if path is None:
                self.status.setText("Select a FITS file before opening its header.")
                return

            try:
                if self.header_window is None:
                    self.header_window = HeaderWindow(self)
                self.header_window.show_file(path)
            except Exception as exc:
                self.show_error("Could not read FITS header", exc)

        def on_file_selection_changed(self):
            path = self.selected_path()
            if path is None:
                return
            self.current_image_path = path
            self.preview_image(path)

        def preview_image(self, path):
            self.current_record = None
            self.current_raw_image = None
            try:
                record = read_fits_image(path)
                image = record.data[0] if record.data.ndim == 3 else record.data
                self.current_record = record
                self.current_raw_image = image
                self.draw_image(image, title=path.name)
                self.status.setText(f"Previewing {path.name}")
            except Exception as exc:
                self.show_error("Could not preview image", exc)

        def draw_image(self, image, title="", peaks=None):
            self.current_display_image = image
            self.current_display_title = title
            self.current_display_peaks = peaks
            ny, nx = image.shape
            self.inspect_x.setMaximum(max(float(nx - 1), 0.0))
            self.inspect_y.setMaximum(max(float(ny - 1), 0.0))
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            vmin, vmax = display_limits(image, config=self.config)
            image_sample, stride = _display_sample(image)
            display_image = apply_stretch(image_sample, vmin=vmin, vmax=vmax, stretch=self.config.stretch)
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

        def update_cutout_control_state(self):
            """Enable only the controls used by the selected preset."""
            preset = self.cutout_preset.currentText()
            percentile_enabled = preset == "Percentile"
            manual_enabled = preset == "Manual"
            self.cutout_percentile_low.setEnabled(percentile_enabled)
            self.cutout_percentile_high.setEnabled(percentile_enabled)
            self.cutout_manual_vmin.setEnabled(manual_enabled)
            self.cutout_manual_vmax.setEnabled(manual_enabled)

        def update_main_control_state(self):
            """Enable only the main-image fields used by the selected preset."""
            preset = self.main_preset.currentText()
            percentile_enabled = preset == "Percentile"
            manual_enabled = preset == "Manual"
            self.main_percentile_low.setEnabled(percentile_enabled)
            self.main_percentile_high.setEnabled(percentile_enabled)
            self.main_manual_vmin.setEnabled(manual_enabled)
            self.main_manual_vmax.setEnabled(manual_enabled)

        def on_main_preset_changed(self, preset):
            """Load main-image preset defaults and redraw the image."""
            if preset == "Standard":
                stretch = "linear"
            elif preset == "Bahtinov":
                stretch = "sqrt"
            else:
                stretch = self.main_stretch.currentText()

            self.main_stretch.blockSignals(True)
            self.main_stretch.setCurrentText(stretch)
            self.main_stretch.blockSignals(False)
            self.update_main_control_state()
            self.update_main_display_from_controls()

        def update_main_display_from_controls(self, *_args):
            """Apply main-image settings and redraw without reanalysis."""
            try:
                configure_main_display(
                    self.config,
                    self.main_preset.currentText(),
                    stretch=self.main_stretch.currentText(),
                    percentile_low=self.main_percentile_low.value(),
                    percentile_high=self.main_percentile_high.value(),
                    manual_vmin=self.main_manual_vmin.value(),
                    manual_vmax=self.main_manual_vmax.value(),
                )
            except ValueError as exc:
                self.status.setText(f"Main image display settings: {exc}")
                return

            if self.current_display_image is not None:
                self.draw_image(
                    self.current_display_image,
                    title=self.current_display_title,
                    peaks=self.current_display_peaks,
                )

        def on_cutout_preset_changed(self, preset):
            """Load preset defaults and refresh the selected cutout."""
            if preset == "Standard":
                stretch = "linear"
            elif preset == "Bahtinov":
                stretch = "sqrt"
            else:
                stretch = self.cutout_stretch.currentText()

            self.cutout_stretch.blockSignals(True)
            self.cutout_stretch.setCurrentText(stretch)
            self.cutout_stretch.blockSignals(False)
            self.update_cutout_control_state()
            self.update_cutout_display_from_controls()

        def update_cutout_display_from_controls(self, *_args):
            """Apply brightness controls and redraw an open Peak Inspector."""
            try:
                configure_cutout_display(
                    self.config,
                    self.cutout_preset.currentText(),
                    stretch=self.cutout_stretch.currentText(),
                    percentile_low=self.cutout_percentile_low.value(),
                    percentile_high=self.cutout_percentile_high.value(),
                    manual_vmin=self.cutout_manual_vmin.value(),
                    manual_vmax=self.cutout_manual_vmax.value(),
                )
            except ValueError as exc:
                self.status.setText(f"Cutout display settings: {exc}")
                return

            self.refresh_selected_peak()

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
                cached_record = self.current_record if path == self.current_image_path else None
                cached_image = self.current_raw_image if cached_record is not None else None
                result = analyze_image(
                    path,
                    self.config,
                    mode=mode,
                    record=cached_record,
                    image=cached_image,
                )
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
            if self.manual_inspection_location is not None:
                self.show_inspection_location(*self.manual_inspection_location)
                return

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

            self.manual_inspection_location = None
            self.peak_window.show_peak(
                self.current_result,
                peak_index,
                self.config,
                cutout_size=self.cutout_size.value(),
            )

        def inspect_coordinates(self):
            """Open Peak Inspector at the typed main-image coordinates."""
            self.show_inspection_location(self.inspect_x.value(), self.inspect_y.value())

        def on_main_image_click(self, event):
            """Inspect a clicked main-image location when click mode is enabled."""
            if not self.inspect_clicks.isChecked() or event.inaxes is None:
                return
            if event.xdata is None or event.ydata is None:
                return
            self.inspect_x.setValue(event.xdata)
            self.inspect_y.setValue(event.ydata)
            self.show_inspection_location(event.xdata, event.ydata)

        def show_inspection_location(self, x, y):
            """Show an arbitrary location from the current preview or result."""
            image = self.current_display_image
            if image is None:
                self.status.setText("Select an image before inspecting coordinates.")
                return

            ny, nx = image.shape
            x = float(np.clip(x, 0, nx - 1))
            y = float(np.clip(y, 0, ny - 1))
            self.inspect_x.setValue(x)
            self.inspect_y.setValue(y)
            self.manual_inspection_location = (x, y)
            if self.peak_window is None:
                self.peak_window = PeakWindow(self)
            self.peak_window.show_location(
                image,
                x,
                y,
                self.config,
                cutout_size=self.cutout_size.value(),
                title=self.current_display_title,
            )
            self.status.setText(f"Inspecting x={x:.2f}, y={y:.2f}")

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
