import numpy as np


def counts_to_mag(counts: float, exposure: float, zero_mag_counts: float = 115_000_000.0) -> float:
    """Convert measured counts to an instrumental magnitude.

    Parameters
    ----------
    counts : float
        Integrated source counts.
    exposure : float
        Exposure time in seconds. Values <= 0 are treated as 0.001 second.
    zero_mag_counts : float
        Counts expected from a zero-magnitude source in one second.
    """
    counts = max(abs(float(counts)), 1.0)
    exposure = max(float(exposure), 0.001)
    return float(2.5 * np.log10(zero_mag_counts) - 2.5 * np.log10(counts / exposure))


def peaks_to_array(peaks):
    """Return an ``N x 6`` float32 array: x, y, area, sx, sy, flux."""
    return np.asarray([[p.x, p.y, p.area, p.sx, p.sy, p.flux] for p in peaks], dtype=np.float32)
