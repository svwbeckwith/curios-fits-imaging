import numpy as np

from fits_imaging.peak_finding import find_candidates


def test_find_candidates_uses_threshold_sigma():
    image = np.full((80, 80), 100.0, dtype=float)
    y, x = np.indices(image.shape)
    image += 60.0 * np.exp(-((x - 40.0) ** 2 + (y - 40.0) ** 2) / (2.0 * 1.5 * 1.5))
    median = 100.0
    sigma = 5.0

    low_threshold = find_candidates(
        image,
        median=median,
        sigma=sigma,
        peak_sharp=0.0,
        threshold_sigma=5.0,
    )
    high_threshold = find_candidates(
        image,
        median=median,
        sigma=sigma,
        peak_sharp=0.0,
        threshold_sigma=20.0,
    )

    assert len(low_threshold) == 1
    assert len(high_threshold) == 0
