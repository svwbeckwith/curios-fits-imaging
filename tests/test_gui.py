import importlib.util

import numpy as np
from types import SimpleNamespace

from fits_imaging import gui


def test_gui_module_imports_without_pyside6():
    assert gui.ANALYSIS_MODE_LABELS["Auto"] is None


def test_gui_main_reports_missing_dependency_when_pyside6_absent():
    if importlib.util.find_spec("PySide6") is not None:
        return

    assert gui.main([]) == 1


def test_display_sample_downsamples_large_images():
    image = np.zeros((100, 100))
    sample, stride = gui._display_sample(image, max_pixels=1000)

    assert stride > 1
    assert sample.shape[0] < image.shape[0]
    assert sample.shape[1] < image.shape[1]


def test_peak_rows_include_flux_magnitude_and_widths():
    result = SimpleNamespace(
        peak_array=np.array([[10.0, 12.0, 100.0, 1.0, 2.0, 5000.0]]),
        photometry_exposure_sec=5.0,
    )
    config = SimpleNamespace(zero_mag_counts=1000.0)

    rows = gui.peak_rows(result, config=config)

    assert rows[0]["peak"] == 0
    assert rows[0]["x"] == 10.0
    assert rows[0]["y"] == 12.0
    assert rows[0]["flux"] == 5000.0
    assert rows[0]["sx"] == 1.0
    assert rows[0]["sy"] == 2.0
    assert rows[0]["fwhm"] == 2.3548 * 1.5


def test_peak_cutout_returns_local_peak_coordinates():
    image = np.zeros((100, 100))
    peak = np.array([10.0, 12.0, 100.0, 1.0, 2.0, 5000.0])

    cutout = gui.peak_cutout(image, peak, half_size=5)

    assert cutout["image"].shape == (11, 11)
    assert cutout["local_x"] == 5.0
    assert cutout["local_y"] == 5.0


def test_peak_display_limits_use_local_cutout_range():
    image = np.array([[1.0, 2.0], [3.0, 1000.0]])

    vmin, vmax = gui.peak_display_limits(image)

    assert vmin >= 1.0
    assert vmax <= 1000.0
    assert vmax > vmin


def test_peak_display_limits_handles_flat_cutout():
    image = np.full((5, 5), 7.0)

    vmin, vmax = gui.peak_display_limits(image)

    assert vmin == 6.0
    assert vmax == 8.0
