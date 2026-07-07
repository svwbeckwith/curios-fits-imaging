import numpy as np

from fits_imaging.contrast import percentile_limits, sigma_limits, apply_stretch


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