"""Examples of independent peak-cutout brightness controls."""

from fits_imaging import ImagingConfig


config = ImagingConfig()

# Balanced stellar cutouts (the default).
config.use_standard_cutout_display()

# Deliberately saturate the core and reveal faint diffraction arms.
config.use_bahtinov_display()

# Or choose per-cutout percentile scaling.
config.use_percentile_cutout_display(low=0.5, high=98.0, stretch="sqrt")

# Or use the same detector-value range for every cutout.
config.use_manual_cutout_display(vmin=100.0, vmax=2500.0, stretch="log")

# Pass config to the existing API:
# result.plot_cutouts(config=config)
