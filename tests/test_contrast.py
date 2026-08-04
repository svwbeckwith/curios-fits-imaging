import numpy as np

import pytest

from fits_imaging.config import ImagingConfig
from fits_imaging.contrast import (
    apply_stretch,
    cutout_display_limits,
    percentile_limits,
    sigma_limits,
)


def test_percentile_limits():
    image = np.arange(1000, dtype=float)
    vmin, vmax = percentile_limits(image, percentiles=(1, 99))
    assert vmin < vmax


def test_sigma_limits():
    image = np.random.default_rng(1).normal(100, 5, size=10000)
    vmin, vmax = sigma_limits(image, nsigma=5)
    assert vmin < 100 < vmax


def test_apply_stretch_linear():
    image = np.array([0, 5, 10], dtype=float)
    out = apply_stretch(image, vmin=0, vmax=10, stretch="linear")
    assert np.allclose(out, [0, 0.5, 1])


def test_standard_and_bahtinov_cutout_presets():
    config = ImagingConfig()
    config.use_bahtinov_display()
    assert config.cutout_contrast_method == "sigma"
    assert config.cutout_contrast_sigma_high == 2.0
    assert config.cutout_stretch == "sqrt"

    config.use_standard_cutout_display()
    assert config.cutout_contrast_sigma_high == 5.0
    assert config.cutout_stretch == "linear"


def test_percentile_cutout_limits():
    image = np.arange(1000, dtype=float)
    config = ImagingConfig()
    config.use_percentile_cutout_display(10, 90)
    vmin, vmax = cutout_display_limits(image, config=config)
    assert (vmin, vmax) == pytest.approx((99.9, 899.1))


def test_manual_cutout_limits():
    config = ImagingConfig()
    config.use_manual_cutout_display(12, 34, stretch="log")
    assert cutout_display_limits(np.arange(10), config=config) == (12.0, 34.0)
    assert config.cutout_stretch == "log"


def test_invalid_manual_cutout_limits():
    with pytest.raises(ValueError, match="greater"):
        ImagingConfig().use_manual_cutout_display(10, 10)
