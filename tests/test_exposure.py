import numpy as np
from astropy.io import fits

from fits_imaging.analysis import analyze_image
from fits_imaging.config import ImagingConfig
from fits_imaging.fits_io import parse_sum_frame_count, read_fits_image
from fits_imaging.run import ImagingRun


def test_parse_sum_frame_count():
    assert parse_sum_frame_count("M15_2026-06-29_CuED_10_sum.fits") == 10
    assert parse_sum_frame_count("Vega_2026-06-29_CuED-0.fits") is None


def test_read_fits_image_derives_total_exposure_from_sum_filename(tmp_path):
    path = tmp_path / "M15_2026-06-29_CuED_5_sum.fits"
    hdu = fits.PrimaryHDU(data=np.ones((3, 3)))
    hdu.header["EXPTIME"] = 0.02
    hdu.writeto(path)

    record = read_fits_image(path)

    assert record.exposure_sec == 0.02
    assert record.frame_count == 5
    assert record.total_exposure_sec == 0.1
    assert record.exposure_source == "filename_sum_count"


def test_header_total_exposure_overrides_filename_count(tmp_path):
    path = tmp_path / "M15_2026-06-29_CuED_5_sum.fits"
    hdu = fits.PrimaryHDU(data=np.ones((3, 3)))
    hdu.header["EXPTIME"] = 0.02
    hdu.header["EXPTOTAL"] = 0.2
    hdu.writeto(path)

    record = read_fits_image(path)

    assert record.frame_count == 5
    assert record.total_exposure_sec == 0.2
    assert record.exposure_source == "header_total_exposure"


def test_run_metadata_includes_exposure_accounting(tmp_path):
    path = tmp_path / "M15_2026-06-29_CuED_5_sum.fits"
    hdu = fits.PrimaryHDU(data=np.ones((3, 3)))
    hdu.header["EXPTIME"] = 0.02
    hdu.writeto(path)

    run = ImagingRun(tmp_path, read_headers=True)

    assert run.file_table.loc[0, "exposure_sec"] == 0.02
    assert run.file_table.loc[0, "frame_count"] == 5
    assert run.file_table.loc[0, "total_exposure_sec"] == 0.1


def test_analysis_uses_total_exposure_for_photometry(tmp_path):
    path = tmp_path / "M15_2026-06-29_CuED_5_sum.fits"
    hdu = fits.PrimaryHDU(data=np.ones((3, 3)))
    hdu.header["EXPTIME"] = 0.02
    hdu.writeto(path)

    result = analyze_image(path, ImagingConfig(data_root=tmp_path), mode="statistics")

    assert result.exposure_sec == 0.02
    assert result.total_exposure_sec == 0.1
    assert result.photometry_exposure_sec == 0.1
