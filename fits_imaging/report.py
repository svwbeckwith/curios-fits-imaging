"""Text reporting helpers for FITS imaging analyses."""

import numpy as np
from .photometry import counts_to_mag

def print_image_summary(record, image, stats):
    """Print a compact image and telescope metadata summary."""
    mode = stats.get("analysis_mode", "source_detection")
    print("{}  {} x {}  {}  {:.3f} sec".format(
        record.object_name, image.shape[1], image.shape[0], record.camera, record.exposure_sec
    ))
    print(f"Analysis mode: {mode}")
    print(
        "RAH: {:7.3f}  Dec: {:7.3f}  PA: {:6.2f}  PA2: {:6.2f}  Alt: {:7.1f}  Az: {:7.1f}".format(
            record.ra_hours,
            record.dec_deg,
            record.pa_deg,
            stats.get("pa_calc_deg", 0.0),
            record.alt_deg,
            record.az_deg,
        )
    )
    print(
        "Mean: {:.1f}  Median: {:.1f}  Std: {:.1f}  Min: {:.1f}  Max: {:.1f}".format(
            stats.get("mean", 0.0),
            stats.get("median", 0.0),
            stats.get("std", 0.0),
            stats.get("min", 0.0),
            stats.get("max", 0.0),
        )
    )
    if mode == "source_detection":
        print("Found {} peaks in {:.2f} sec".format(stats.get("npeaks", 0), stats.get("elapsed_sec", 0.0)))
    else:
        print("Completed in {:.2f} sec".format(stats.get("elapsed_sec", 0.0)))

def print_peak_table(
    peaks_array,
    exposure_sec,
    zero_mag_counts,
    pa0=0.0,
    swarpfac=1.0,
    max_rows=200,
):
    """Print detected peak table.

    peaks_array columns are expected to be:
        x, y, area, sx, sy, flux
    """
    if peaks_array is None or len(peaks_array) == 0:
        print("No peaks found.")
        return

    peaks = np.asarray(peaks_array)

    print()
    print("Peak#        X        Y        Area       sx      sy        Flux      Mag")
    print("-" * 78)

    for i, row in enumerate(peaks[:max_rows]):
        x, y, area, sx, sy, flux = row[:6]
        mag = counts_to_mag(
            flux,
            exposure=exposure_sec,
            zero_mag_counts=zero_mag_counts,
        )

        print(
            f"{i:5d} "
            f"{x:9.2f} "
            f"{y:9.2f} "
            f"{area:11.1f} "
            f"{sx:7.2f} "
            f"{sy:7.2f} "
            f"{flux:11.1f} "
            f"{mag:8.3f}"
        )
