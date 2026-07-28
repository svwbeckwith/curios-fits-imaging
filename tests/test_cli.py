from pathlib import Path

from fits_imaging.cli import build_parser, config_from_args


def test_config_from_args_sets_analysis_parameters(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            str(tmp_path),
            "--threshold-sigma",
            "7.5",
            "--max-peaks",
            "25",
            "--peak-separation",
            "12",
            "--peak-sharp",
            "0.3",
            "--fit-method",
            "moments",
            "--zero-mag-counts",
            "123",
            "--pixel-arcsec",
            "1.7",
        ]
    )

    config = config_from_args(args, data_root=Path("/tmp"))

    assert config.threshold_sigma == 7.5
    assert config.max_peaks == 25
    assert config.peak_separation == 12
    assert config.peak_sharp == 0.3
    assert config.fit_method == "moments"
    assert config.zero_mag_counts == 123
    assert config.pixel_arcsec == 1.7
