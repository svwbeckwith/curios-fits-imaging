import matplotlib
import numpy as np

matplotlib.use("Agg")

from fits_imaging.config import ImagingConfig
from fits_imaging.plotting import plot_peak_cutouts


def test_plot_peak_cutouts_applies_manual_display_settings():
    image = np.arange(121, dtype=float).reshape(11, 11)
    config = ImagingConfig(cutout_colormap="gray")
    config.use_manual_cutout_display(20, 80, stretch="sqrt")

    fig, axes = plot_peak_cutouts(image, [[5, 5]], half_size=5, config=config)
    shown = axes[0].images[0]

    assert shown.get_cmap().name == "gray"
    assert shown.get_clim() == (0.0, 1.0)
    assert np.asarray(shown.get_array())[5, 5] == np.sqrt((60.0 - 20.0) / 60.0)
    fig.clear()
