"""Image-region helpers."""

import numpy as np


def center_crop(image, fraction=None, size_pixels=None):
    """Return centered crop of an image.

    Use either fraction, e.g. 0.5 for central half in x and y,
    or size_pixels, e.g. 2000 for a 2000 x 2000 center crop.
    """
    image = np.asarray(image)
    ny, nx = image.shape

    if size_pixels is not None:
        sx = min(int(size_pixels), nx)
        sy = min(int(size_pixels), ny)
    else:
        if fraction is None:
            fraction = 1.0
        sx = max(1, int(nx * fraction))
        sy = max(1, int(ny * fraction))

    x0 = (nx - sx) // 2
    y0 = (ny - sy) // 2

    return image[y0:y0 + sy, x0:x0 + sx]


def statistics_image(image, config=None):
    """Return image region to use for global statistics."""
    region = getattr(config, "stats_region", "full")

    if region == "full":
        return image

    if region == "center_fraction":
        fraction = getattr(config, "stats_center_fraction", 0.5)
        return center_crop(image, fraction=fraction)

    if region == "center_pixels":
        size_pixels = getattr(config, "stats_center_pixels", 2000)
        return center_crop(image, size_pixels=size_pixels)

    raise ValueError(f"Unknown stats_region: {region}")
    
def point_in_center_region(x, y, image_shape, fraction=0.5):
    ny, nx = image_shape
    sx = nx * fraction
    sy = ny * fraction
    x0 = 0.5 * (nx - sx)
    x1 = 0.5 * (nx + sx)
    y0 = 0.5 * (ny - sy)
    y1 = 0.5 * (ny + sy)
    return (x >= x0) and (x <= x1) and (y >= y0) and (y <= y1)


def select_peaks_region(peaks_array, image_shape, config=None):
    peaks = np.asarray(peaks_array)

    if peaks.size == 0:
        return peaks

    if peaks.ndim == 1:
        peaks = peaks.reshape(1, -1)

    region = getattr(config, "source_region", "full")

    if region == "full":
        return peaks

    if region == "center_fraction":
        fraction = getattr(config, "source_region_fraction", 0.5)
        mask = [
            point_in_center_region(row[0], row[1], image_shape, fraction=fraction)
            for row in peaks
        ]
        return peaks[np.asarray(mask)]

    raise ValueError(f"Unknown source_region: {region}")
