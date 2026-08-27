import numpy as np
from astropy.io import fits

from fits_imaging.analysis import analyze_image, normalize_analysis_mode
from fits_imaging.config import ImagingConfig
from fits_imaging.fits_io import read_fits_image


def write_test_fits(path, image):
    fits.PrimaryHDU(data=np.asarray(image, dtype=float)).writeto(path)


def test_normalize_analysis_mode_aliases():
    assert normalize_analysis_mode(None) == "source_detection"
    assert normalize_analysis_mode("source") == "source_detection"
    assert normalize_analysis_mode("stats") == "statistics"
    assert normalize_analysis_mode("bahtinov") == "focus"
    assert normalize_analysis_mode("blur") == "rotation_blur"


def test_statistics_mode_skips_peak_detection(tmp_path):
    path = tmp_path / "Dark_001.fits"
    write_test_fits(path, [[1, 2], [3, 4]])

    result = analyze_image(path, ImagingConfig(data_root=tmp_path), mode="statistics")

    assert result.analysis_mode == "statistics"
    assert result.nstars == 0
    assert result.stats["median"] == 2.5
    assert result.stats["peak_find_time_sec"] == 0.0


def test_analysis_can_reuse_decoded_preview(tmp_path, monkeypatch):
    path = tmp_path / "cached.fits"
    write_test_fits(path, [[1, 2], [3, 4]])
    record = read_fits_image(path)
    image = record.data

    def unexpected_read(_path):
        raise AssertionError("cached analysis decoded the FITS file again")

    monkeypatch.setattr("fits_imaging.analysis.read_fits_image", unexpected_read)
    result = analyze_image(
        path,
        ImagingConfig(data_root=tmp_path),
        mode="statistics",
        record=record,
        image=image,
    )

    assert result.stats["median"] == 2.5


def test_focus_mode_adds_focus_metrics(tmp_path):
    path = tmp_path / "Vega-Bahtinov_001.fits"
    write_test_fits(path, [[1, 2, 1], [2, 10, 2], [1, 2, 1]])

    result = analyze_image(path, ImagingConfig(data_root=tmp_path), mode="focus")

    assert result.analysis_mode == "focus"
    assert result.nstars == 0
    assert result.stats["focus_peak_x"] == 1.0
    assert result.stats["focus_peak_y"] == 1.0
    assert result.stats["focus_sharpness"] > 0.0


def test_rotation_blur_mode_adds_gradient_metrics(tmp_path):
    path = tmp_path / "Vega_rotating_field.fits"
    image = np.tile(np.arange(6, dtype=float), (6, 1))
    write_test_fits(path, image)

    result = analyze_image(path, ImagingConfig(data_root=tmp_path), mode="rotation_blur")

    assert result.analysis_mode == "rotation_blur"
    assert result.nstars == 0
    assert result.stats["gradient_strength"] > 0.0
    assert 0.0 <= result.stats["gradient_anisotropy"] <= 1.0
