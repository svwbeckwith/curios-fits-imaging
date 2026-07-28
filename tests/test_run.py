import pandas as pd

from fits_imaging.run import ImagingRun, analysis_mode_for_kind, classify_image_text


def test_classify_image_text():
    assert classify_image_text("Vega-Bahtinov_2026-06-29.fits") == "focus"
    assert classify_image_text("Dark_001.fits") == "dark"
    assert classify_image_text("sky_flat_001.fits") == "flat"
    assert classify_image_text("M15_2026-06-29_CuED_10_sum.fits") == "science_sum"
    assert classify_image_text("Vega_rotating_field.fits") == "rotation_blur"
    assert classify_image_text("Vega_2026-06-29_CuED-0.fits") == "science_single"


def test_analysis_mode_for_kind():
    assert analysis_mode_for_kind("science_single") == "source_detection"
    assert analysis_mode_for_kind("science_sum") == "source_detection"
    assert analysis_mode_for_kind("focus") == "focus"
    assert analysis_mode_for_kind("rotation_blur") == "rotation_blur"
    assert analysis_mode_for_kind("dark") == "statistics"
    assert analysis_mode_for_kind("flat") == "statistics"


def test_files_for_filters_science_alias(tmp_path):
    files = [
        tmp_path / "Vega_2026-06-29_CuED-0.fits",
        tmp_path / "M15_2026-06-29_CuED_10_sum.fits",
        tmp_path / "Dark_001.fits",
    ]
    for path in files:
        path.touch()

    run = ImagingRun(tmp_path)
    run.file_table = pd.DataFrame(
        [
            {
                "file": path.name,
                "path": str(path),
                "kind": classify_image_text(path.name),
                "include": True,
            }
            for path in files
        ]
    )

    assert run.files_for(kind="science") == [files[0], files[1]]
    assert run.files_for(kind="dark") == [files[2]]

    table = run.table_for(kind="science")
    assert table["index"].tolist() == [0, 1]
    assert table["file"].tolist() == [files[0].name, files[1].name]

    preview = run.preview(kind="science", n=1)
    assert preview.columns.tolist() == ["index", "file", "kind", "include"]
    assert preview["file"].tolist() == [files[0].name]


def test_file_table_is_filename_based_by_default(tmp_path):
    path = tmp_path / "Dark_001.fits"
    path.touch()

    run = ImagingRun(tmp_path)

    assert run.file_table.loc[0, "file"] == "Dark_001.fits"
    assert run.file_table.loc[0, "kind"] == "dark"
    assert not run.file_table.loc[0, "metadata_loaded"]
    assert run.file_table.loc[0, "error"] == ""


def test_files_for_respects_include_flag(tmp_path):
    first = tmp_path / "Vega_2026-06-29_CuED-0.fits"
    second = tmp_path / "M15_2026-06-29_CuED_10_sum.fits"
    first.touch()
    second.touch()

    run = ImagingRun(tmp_path)
    run.file_table = pd.DataFrame(
        [
            {
                "file": first.name,
                "path": str(first),
                "kind": "science_single",
                "include": False,
            },
            {
                "file": second.name,
                "path": str(second),
                "kind": "science_sum",
                "include": True,
            },
        ]
    )

    assert run.files_for(kind="science") == [second]
    assert run.files_for(kind="science", included_only=False) == [first, second]
