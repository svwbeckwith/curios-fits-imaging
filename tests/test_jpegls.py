import numpy as np
from astropy.io import fits

from fits_imaging.fits_io import list_image_files, read_fits_image
from fits_imaging.jpegls_compat import _split_container_layout, install_jpegls_compatibility
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


def test_32bit_decoder_accepts_current_and_historical_containers():
    from astropy.io.fits.hdu.compressed import _tiled_compression

    assert install_jpegls_compatibility()
    codec = _tiled_compression.ALGORITHMS["JPEGLS"](bitpix=32, max_err=0)
    data = np.array(
        [np.iinfo(np.int32).min, -123456, 0, 123456, np.iinfo(np.int32).max],
        dtype=np.int32,
    ).reshape(1, -1)
    current = codec.encode(data)
    upper_len = int.from_bytes(current[:4], "big")
    assert int.from_bytes(current[4:8], "big") == 0
    assert upper_len < 65536

    legacy_4 = current[:4] + current[8:]
    legacy_2 = upper_len.to_bytes(2, "big") + current[8:]

    assert _split_container_layout(current)[0] == "current"
    assert _split_container_layout(legacy_4)[0] == "legacy-4-byte"
    assert _split_container_layout(legacy_2)[0] == "legacy-2-byte"
    for encoded in (current, legacy_4, legacy_2):
        assert np.array_equal(codec.decode(encoded).reshape(data.shape), data)
