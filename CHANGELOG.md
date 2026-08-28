# Changelog

## Unreleased

- Add an optional, searchable FITS-header window to the GUI with multi-HDU
  navigation and clipboard copying, without decoding compressed image data.
- Read all three CuRIOS 32-bit JPEG-LS container revisions (historical 2-byte
  and 4-byte headers plus the current length-and-baseline header), using JPEG-LS
  marker validation to reject corrupt or ambiguous tiles.
- Reuse the GUI's decoded preview during analysis so compressed images are not
  decompressed twice.
- Add JPEG-LS compressed FITS reading through the pinned CuRIOS Astropy fork
  and `imagecodecs`, while retaining plain, Rice, GZIP, and other existing FITS
  modes.
- Move the supported runtime to NumPy 2 and remove the unused Photutils
  dependency.

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
