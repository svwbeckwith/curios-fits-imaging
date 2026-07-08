"""Session state for notebook/workbench workflows."""

import json
from pathlib import Path

from .fits_io import choose_file


DEFAULT_DATA_ROOT = "~/Dropbox/CuRIOS/Software/DataDirectories"
DEFAULT_STATE_PATH = "~/.curios_fits_imaging/session.json"
DEFAULT_DATA_ROOT_CANDIDATES = [
    "~/Library/CloudStorage/Dropbox/CuRIOS/Software/DataDirectories",
    "~/Dropbox/CuRIOS/Software/DataDirectories",
]

def default_data_root():
    for path in DEFAULT_DATA_ROOT_CANDIDATES:
        p = Path(path).expanduser()
        if p.exists():
            return p
    return Path(DEFAULT_DATA_ROOT_CANDIDATES[0]).expanduser()

class ImagingSession:
    """Remember the current working data folder and last-used image."""

    def __init__(self, data_root=None, state_path=None):
        self.state_path = Path(state_path or DEFAULT_STATE_PATH).expanduser()
        self.data_root = Path(data_root).expanduser() if data_root else default_data_root()
        self.current_folder = None
        self.last_image = None

        self.load()

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

    def choose_folder(self):
        """Choose a working folder under data_root."""
        folders = sorted([p for p in self.data_root.iterdir() if p.is_dir()])

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
        
    def analyze(self, config=None, change_folder=False):
        """Choose an image from the session folder and analyze it."""
        from .analysis import analyze_image

        image_path = self.choose_image(change_folder=change_folder)
        return analyze_image(image_path, config)
        
    def analyze_current_folder(self, config, max_files=None):
        """Analyze all FITS images in the current working folder."""
        from .batch import analyze_folder

        if self.current_folder is None:
            self.choose_folder()

        return analyze_folder(
            self.current_folder,
            config=config,
            max_files=max_files,
        )
        
    def open_run(self, change_folder=False):
        """Open the current folder as an ImagingRun."""
        from .run import ImagingRun

        if self.current_folder is None or change_folder:
            self.choose_folder()

        return ImagingRun(self.current_folder)
