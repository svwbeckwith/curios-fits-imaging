"""Session state for notebook/workbench workflows."""

from pathlib import Path

from .fits_io import choose_file


class ImagingSession:
    """Remember the current working data folder."""

    def __init__(self, data_root):
        self.data_root = Path(data_root).expanduser()
        self.current_folder = None

    def choose_folder(self):
        folders = sorted([p for p in self.data_root.iterdir() if p.is_dir()])

        if not folders:
            raise FileNotFoundError(f"No subfolders found in {self.data_root}")

        print("\nFolder#   Folder")
        print("-" * 60)
        for i, folder in enumerate(folders):
            print(f"{i:4d}     {folder.name}")

        choice = int(input("\nChoose folder number: "))
        self.current_folder = folders[choice]
        return self.current_folder

    def choose_image(self, extensions=(".fits", ".fit"), change_folder=False):
        if self.current_folder is None or change_folder:
            self.choose_folder()

        return choose_file(self.current_folder, extensions=extensions)
