# Changelog

All notable changes to this project will be documented in this file.

---

## Version 2.8

Major redesign of the package architecture.

### Added

- ImagingSession class
- ImagingResults API
- Quick-look report
- Region-aware image statistics
- Persistent session state
- CSV export
- Source diagnostics
- PSF radial profiles
- Encircled-energy calculations
- Configurable plotting styles

### Changed

- Simplified notebook workflow
- Plotting functions integrated into ImagingResults
- Configuration consolidated into ImagingConfig
- Session management separated from configuration
- Improved image contrast controls
- Improved histogram generation

### Fixed

- Python 3.12 compatibility
- Contrast scaling
- Peak table interface
- Cutout plotting
- Histogram performance
- Numerous import and API inconsistencies

---

## Version 2.7

### Added

- ImagingResults class
- Diagnostics module
- Export module
- Peak table improvements
- Plot styling module

### Changed

- Refactored plotting code
- Improved package organization

---

## Version 2.6

### Added

- Configurable plotting styles
- Improved image display
- Viridis color map support
- Adjustable contrast scaling
- Better notebook organization

---

## Earlier Versions

Initial development of the FITS imaging package including

- FITS image reading
- Peak detection
- Gaussian fitting
- Photometry
- Plotting
- Notebook workflow
