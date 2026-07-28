"""Session state for notebook/workbench workflows."""

import json
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .fits_io import choose_file, list_data_directories


#DEFAULT_DATA_ROOT = "~/Dropbox/CuRIOS/Software/DataDirectories"
DEFAULT_DATA_ROOT = None
DEFAULT_STATE_PATH = "~/.curios_fits_imaging/session.json"
DEFAULT_DATA_ROOT_CANDIDATES = [
    "~/Library/CloudStorage/Dropbox/CuRIOS/Software/DataDirectories",
    "~/Dropbox/CuRIOS/Software/DataDirectories",
]

def default_data_root():
    env_path = os.environ.get("CURIOS_DATA_ROOT")
    if env_path:
        return Path(env_path).expanduser()

    for path in DEFAULT_DATA_ROOT_CANDIDATES:
        p = Path(path).expanduser()
        if p.exists():
            return p
    return Path(DEFAULT_DATA_ROOT_CANDIDATES[0]).expanduser()

class ImagingSession:
    """Remember the current working data folder and last-used image."""
    
    def __init__(
        self,
        data_root: Optional[Union[str, Path]] = None,
        state_path: Optional[Union[str, Path]] = None,
    ):
        # Location of the saved session-state file
        if state_path is None:
            self.state_path = (
                Path.home()
                / ".config"
                / "curios-fits-imaging"
                / "session.json"
            )
        else:
            self.state_path = Path(state_path).expanduser().resolve()

        # Session values expected by save(), load(), and choose_image()
        self.data_root = None
        self.current_folder = None
        self.last_image = None

        # An explicitly supplied data_root overrides the saved value
        if data_root is not None:
            self.data_root = Path(data_root).expanduser().resolve()
        else:
            self.load()
            if self.data_root is None:
                self.data_root = default_data_root()
            
    def load(self):
        """Load previous session state if available."""
        if not self.state_path.exists():
            return

        try:
            with self.state_path.open("r") as f:
                state = json.load(f)

            if "data_root" in state:
                self.data_root = Path(state["data_root"]).expanduser()

            if "current_folder" in state and state["current_folder"]:
                folder = Path(state["current_folder"]).expanduser()
                if folder.exists():
                    self.current_folder = folder

            if "last_image" in state and state["last_image"]:
                image = Path(state["last_image"]).expanduser()
                if image.exists():
                    self.last_image = image

        except Exception:
            # Bad session files should not prevent analysis.
            pass

    def save(self):
        """Save current session state."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "data_root": str(self.data_root),
            "current_folder": str(self.current_folder) if self.current_folder else None,
            "last_image": str(self.last_image) if self.last_image else None,
        }

        with self.state_path.open("w") as f:
            json.dump(state, f, indent=2)

    def set_data_root(self, data_root):
        """Set the top-level data directory."""
        self.data_root = Path(data_root).expanduser()
        self.current_folder = None
        self.last_image = None
        self.save()

    def folders(self):
        """Return selectable observing folders under data_root."""
        if self.data_root is None:
            raise ValueError("No data_root is set. Pass ImagingSession(data_root=...) or call set_data_root().")

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

        return list_data_directories(self.data_root)

    def folder_table(self):
        """Return a notebook-friendly table of observing folders."""
        rows = [
            {
                "index": i,
                "folder": folder.name,
                "path": str(folder),
            }
            for i, folder in enumerate(self.folders())
        ]
        return pd.DataFrame(rows)

    def set_folder(self, folder):
        """Set the current observing folder by path or folder name."""
        folder = Path(folder).expanduser()

        if not folder.is_absolute():
            folder = self.data_root / folder

        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder}")

        if not folder.is_dir():
            raise NotADirectoryError(f"Not a folder: {folder}")

        self.current_folder = folder
        self.last_image = None
        self.save()
        return self.current_folder

    def set_folder_by_index(self, index):
        """Set the current observing folder from folder_table index."""
        folders = self.folders()
        try:
            self.current_folder = folders[int(index)]
        except IndexError as exc:
            raise IndexError(f"Folder index {index} is out of range.") from exc
        self.last_image = None
        self.save()
        return self.current_folder

    def set_image(self, image_path):
        """Set the last-used image by path or filename in the current folder."""
        image_path = Path(image_path).expanduser()

        if not image_path.is_absolute():
            if self.current_folder is None:
                raise ValueError("No current folder is set.")
            image_path = self.current_folder / image_path

        if not image_path.exists():
            raise FileNotFoundError(f"Image does not exist: {image_path}")

        if not image_path.is_file():
            raise FileNotFoundError(f"Not an image file: {image_path}")

        self.last_image = image_path
        self.save()
        return self.last_image

    def set_image_by_index(self, index, kind=None):
        """Set the last-used image from the current run file table."""
        run = self.open_run()
        files = run.files_for(kind=kind)
        if not files:
            label = "all files" if kind is None else f"kind={kind}"
            raise FileNotFoundError(f"No files available for {label}.")

        try:
            self.last_image = files[int(index)]
        except IndexError as exc:
            raise IndexError(f"Image index {index} is out of range for kind={kind}.") from exc
        self.save()
        return self.last_image

    def choose_folder(self):
        """Choose a working folder under data_root."""
        if self.data_root is None:
            raise ValueError("No data_root is set. Pass ImagingSession(data_root=...) or call set_data_root().")

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

        folders = self.folders()

        if not folders:
            raise FileNotFoundError(f"No subfolders found in {self.data_root}")

        print("\nFolder#   Folder")
        print("-" * 60)
        for i, folder in enumerate(folders):
            print(f"{i:4d}     {folder.name}")

        choice = int(input("\nChoose folder number: "))
        self.current_folder = folders[choice]
        self.save()
        return self.current_folder

    def choose_image(self, extensions=(".fits", ".fit"), change_folder=False):
        """Choose an image from the current working folder.

        The folder is remembered between calls and across sessions.
        """
        if self.current_folder is None or change_folder:
            self.choose_folder()

        image_path = choose_file(self.current_folder, extensions=extensions)
        self.last_image = Path(image_path)
        self.save()
        return image_path

    def status(self):
        """Print current session state."""
        print(f"Data root:      {self.data_root}")
        print(f"Current folder: {self.current_folder}")
        print(f"Last image:     {self.last_image}")
        
    def analyze(self, config=None, change_folder=False, mode=None):
        """Choose an image from the session folder and analyze it."""
        from .analysis import analyze_image

        image_path = self.choose_image(change_folder=change_folder)
        return analyze_image(image_path, config, mode=mode)
        
    def analyze_current_folder(self, config, kind=None, mode=None, max_files=None):
        """Analyze FITS images in the current working folder."""
        from .run import ImagingRun

        if self.current_folder is None:
            self.choose_folder()

        run = ImagingRun(self.current_folder)
        summary = run.analyze_all(config=config, kind=kind, mode=mode, max_files=max_files)
        return run.results, summary
        
    def open_run(self, change_folder=False):
        """Open the current folder as an ImagingRun."""
        from .run import ImagingRun

        if self.current_folder is None or change_folder:
            self.choose_folder()

        return ImagingRun(self.current_folder)
