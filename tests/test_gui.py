import importlib.util

import numpy as np
import pytest
from types import SimpleNamespace

from fits_imaging import gui
from fits_imaging.config import ImagingConfig


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


def test_peak_cutout_default_is_larger_than_48_pixels():
    image = np.zeros((200, 200))
    peak = np.array([100.0, 100.0, 100.0, 1.0, 2.0, 5000.0])

    cutout = gui.peak_cutout(image, peak)

    assert cutout["image"].shape[0] > 48
    assert cutout["image"].shape[1] > 48


def test_peak_cutout_accepts_larger_requested_sizes():
    image = np.zeros((300, 300))
    peak = np.array([150.0, 150.0, 100.0, 1.0, 2.0, 5000.0])

    cutout = gui.peak_cutout(image, peak, cutout_size=151)

    assert cutout["image"].shape == (151, 151)


def test_cutout_extent_uses_absolute_coordinates_with_y_increasing_downward():
    cutout = {"x0": 10, "x1": 21, "y0": 30, "y1": 41}

    extent = gui.cutout_extent(cutout)

    assert extent == (9.5, 20.5, 40.5, 29.5)


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


@pytest.mark.parametrize(
    ("preset", "expected_method", "expected_stretch"),
    [
        ("Standard", "sigma", "linear"),
        ("Bahtinov", "sigma", "sqrt"),
        ("Percentile", "percentile", "log"),
        ("Manual", "manual", "log"),
    ],
)
def test_configure_cutout_display_presets(preset, expected_method, expected_stretch):
    config = ImagingConfig()

    gui.configure_cutout_display(
        config,
        preset,
        stretch="log" if preset in {"Percentile", "Manual"} else expected_stretch,
        percentile_low=2.0,
        percentile_high=98.0,
        manual_vmin=100.0,
        manual_vmax=2000.0,
    )

    assert config.cutout_contrast_method == expected_method
    assert config.cutout_stretch == expected_stretch
    if preset == "Percentile":
        assert config.cutout_contrast_percentiles == (2.0, 98.0)
    if preset == "Manual":
        assert (config.cutout_vmin, config.cutout_vmax) == (100.0, 2000.0)


def test_configure_cutout_display_rejects_invalid_percentiles():
    with pytest.raises(ValueError, match="percentile low"):
        gui.configure_cutout_display(
            ImagingConfig(),
            "Percentile",
            percentile_low=99.0,
            percentile_high=1.0,
        )


@pytest.mark.parametrize(
    ("preset", "expected_method", "expected_stretch"),
    [
        ("Standard", "sigma", "linear"),
        ("Bahtinov", "sigma", "sqrt"),
        ("Percentile", "percentile", "log"),
        ("Manual", "manual", "log"),
    ],
)
def test_configure_main_display_is_independent_of_cutouts(preset, expected_method, expected_stretch):
    config = ImagingConfig()
    config.use_bahtinov_display()
    original_cutout = (
        config.cutout_contrast_method,
        config.cutout_contrast_sigma_high,
        config.cutout_stretch,
    )

    gui.configure_main_display(
        config,
        preset,
        stretch="log" if preset in {"Percentile", "Manual"} else expected_stretch,
        percentile_low=2.0,
        percentile_high=98.0,
        manual_vmin=100.0,
        manual_vmax=2000.0,
    )

    assert config.contrast_method == expected_method
    assert config.stretch == expected_stretch
    assert (
        config.cutout_contrast_method,
        config.cutout_contrast_sigma_high,
        config.cutout_stretch,
    ) == original_cutout
    if preset == "Percentile":
        assert config.contrast_percentiles == (2.0, 98.0)
    if preset == "Manual":
        assert (config.manual_vmin, config.manual_vmax) == (100.0, 2000.0)
