import time
import numpy as np
from .fits_io import read_fits_image
from .photometry import peaks_to_array
from .peak_finding import peak_finder
from .coordinates import parallactic_pa_from_altaz


def analyze_fits_file(path, peak_separation=10.0, peak_sharp=0.2, maxsources=400, fit_method="gaussian"):
    """Read one FITS file and run peak finding. Returns (record, image2d, peaks_array)."""
    record = read_fits_image(path)
    image = record.data[0] if record.data.ndim == 3 else record.data
    t0 = time.time()
    peaks = peak_finder(image, peak_separation=peak_separation, peak_sharp=peak_sharp,
                        maxsources=maxsources, fit_method=fit_method)
    elapsed = time.time() - t0
    peaks_array = peaks_to_array(peaks)
    stats = {
        "mean": float(np.average(image)),
        "median": float(np.median(image)),
        "std": float(np.std(image)),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "elapsed_sec": elapsed,
        "npeaks": int(len(peaks_array)),
        "pa_calc_deg": parallactic_pa_from_altaz(record.alt_deg, record.az_deg) if record.alt_deg or record.az_deg else 0.0,
    }
    return record, image, peaks_array, stats
