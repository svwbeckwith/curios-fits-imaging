import numpy as np
from astropy.io import fits

from fits_imaging.fits_io import list_image_files, read_fits_image
from fits_imaging.run import ImagingRun


def test_reader_preserves_standard_and_jpegls_compressed_images(tmp_path):
    """The application reader supports JPEG-LS without regressing FITS modes."""
    data = (np.arange(96 * 128).reshape(96, 128) % 4096).astype(np.uint16)
    cases = {
        "plain.fits": fits.PrimaryHDU(data),
        "rice.fits.fz": fits.CompImageHDU(data, compression_type="RICE_1"),
        "gzip.fits.fz": fits.CompImageHDU(data, compression_type="GZIP_1"),
        "jpegls.fits.fz": fits.CompImageHDU(data, compression_type="JPEGLS"),
    }

    for filename, hdu in cases.items():
        path = tmp_path / filename
        hdu.writeto(path)

        record = read_fits_image(path)

        assert record.data.dtype == data.dtype
        assert np.array_equal(record.data, data), filename

    discovered = [path.name for path in list_image_files(tmp_path)]
    assert discovered == sorted(cases)

    run = ImagingRun(tmp_path)
    assert [path.name for path in run.files] == sorted(cases)
