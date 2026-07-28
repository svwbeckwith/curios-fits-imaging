import json

from fits_imaging.session import ImagingSession


def test_explicit_data_root_does_not_restore_saved_folder(tmp_path):
    data_root = tmp_path / "data"
    saved_root = tmp_path / "saved"
    saved_folder = saved_root / "old-folder"
    data_root.mkdir()
    saved_folder.mkdir(parents=True)

    state_path = tmp_path / "session.json"
    state_path.write_text(
        json.dumps(
            {
                "data_root": str(saved_root),
                "current_folder": str(saved_folder),
                "last_image": None,
            }
        )
    )

    session = ImagingSession(data_root=data_root, state_path=state_path)

    assert session.data_root == data_root.resolve()
    assert session.current_folder is None
    assert session.last_image is None


def test_folder_table_and_set_folder_by_index(tmp_path):
    data_root = tmp_path / "data"
    first = data_root / "Patio-001"
    second = data_root / "Patio-002"
    first.mkdir(parents=True)
    second.mkdir()

    session = ImagingSession(data_root=data_root, state_path=tmp_path / "session.json")

    table = session.folder_table()
    assert table["folder"].tolist() == ["Patio-001", "Patio-002"]

    selected = session.set_folder_by_index(1)
    assert selected == second
    assert session.current_folder == second


def test_set_image_by_index_uses_filtered_kind(tmp_path):
    data_root = tmp_path / "data"
    folder = data_root / "Patio-001"
    folder.mkdir(parents=True)
    dark = folder / "Dark_001.fits"
    science = folder / "Vega_001_CuED-0.fits"
    dark.touch()
    science.touch()

    session = ImagingSession(data_root=data_root, state_path=tmp_path / "session.json")
    session.set_folder_by_index(0)

    selected = session.set_image_by_index(0, kind="science")
    assert selected == science
    assert session.last_image == science
