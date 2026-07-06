# FITS Imaging Version 2.2

Cleaned Jupyter-based intermediate package for CuRIOS FITS image analysis.

## Changes in 2.2

- Added `fits_imaging/plot_style.py` so plot colors and sizes are centralized.
- Full-image grid lines are now light gray.
- Peak labels are red.
- Peak-cutout center crosses are red.
- Histogram and convolved-image grids use the same light gray style.
- Added `fits_imaging/report.py` for reusable printed image summaries.
- Preserved the photometry exposure floor: `max(exposure, 0.001)` seconds.
- Preserved Python 3.9-compatible type hints.
- Preserved automatic `DataDirectories -> run subdirectory -> FITS file` selection.

## Install/use

Copy the `fits_imaging` folder into your Python working directory or onto your Python path, open
`Fits_Imaging_clean.ipynb`, restart the Jupyter kernel, and run from the top.
