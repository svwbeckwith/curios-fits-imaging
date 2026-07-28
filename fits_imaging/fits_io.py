from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List, Tuple, Union, Optional
import numpy as np
from astropy.io import fits


@dataclass
class ImageRecord:
    """Image data plus the most useful FITS metadata."""
    data: np.ndarray
    filename: str
    path: Path
    header: fits.Header
    object_name: str = "test"
    camera: str = "Camera"
    exposure_sec: float = 1.0
    frame_count: int = 1
    total_exposure_sec: float = 1.0
    exposure_source: str = "header"
    dateobs: str = ""
    ra_hours: float = 0.0
    dec_deg: float = 0.0
    alt_deg: float = 0.0
    az_deg: float = 0.0
    pa_deg: float = 0.0
    detector_temp_c: float = -20.0
    softname: str = ""


def _normalized_extensions(extensions: Iterable[str]) -> Tuple[str, ...]:
    return tuple(ext.lower() if ext.startswith(".") else "." + ext.lower() for ext in extensions)


def list_image_files(directory: Union[str, Path], extensions: Iterable[str] = (".fits", ".fit")) -> List[Path]:
    """Return sorted image files directly inside a directory."""
    directory = Path(directory).expanduser()
    if not directory.exists():
        raise FileNotFoundError("Directory does not exist: {}".format(directory))
    if not directory.is_dir():
        raise NotADirectoryError("Not a directory: {}".format(directory))
    extensions = _normalized_extensions(extensions)
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions)


def list_data_directories(directory: Union[str, Path]) -> List[Path]:
    """Return visible subdirectories, excluding common helper/output directories."""
    directory = Path(directory).expanduser()
    if not directory.exists():
        raise FileNotFoundError("Directory does not exist: {}".format(directory))
    excluded = {"__pycache__", "Figures", "Icons", "Logs", ".ipynb_checkpoints"}
    dirs = []
    for p in directory.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith(".") or p.name.startswith("_") or p.name in excluded:
            continue
        dirs.append(p)
    return sorted(dirs)


def _choose_from_list(items: List[Path], title: str, prompt: str) -> Path:
    if not items:
        raise FileNotFoundError("No selectable items found")
    print("\n{}".format(title))
    print("-" * 70)
    for i, path in enumerate(items):
        print("{:5d}       {}".format(i, path.name))
    print("-" * 70)
    while True:
        text = input(prompt).strip()
        try:
            index = int(text)
            return items[index]
        except (ValueError, IndexError):
            print("Please enter a number from 0 to {}.".format(len(items) - 1))


def choose_directory(directory: Union[str, Path]) -> Path:
    """Interactively choose a visible subdirectory."""
    dirs = list_data_directories(directory)
    return _choose_from_list(dirs, "Directory#   Directory", "Choose directory number: ")


def choose_file(directory: Union[str, Path], extensions: Iterable[str] = (".fits", ".fit"),
                allow_subdirectory_choice: bool = True) -> Path:
    """
    Interactively choose an image file.

    If *directory* contains FITS files, they are listed directly. If it contains
    no FITS files but does contain data subdirectories, the user is first asked
    to choose one subdirectory, then asked to choose a FITS file from it. This
    matches the original CuRIOS DataDirectories workflow.
    """
    directory = Path(directory).expanduser()
    files = list_image_files(directory, extensions)

    if not files and allow_subdirectory_choice:
        subdirs = list_data_directories(directory)
        if subdirs:
            directory = _choose_from_list(subdirs, "Directory#   Directory", "Choose directory number: ")
            files = list_image_files(directory, extensions)

    if not files:
        raise FileNotFoundError("No image files found in {}".format(directory))

    return _choose_from_list(files, "File#       File", "Enter file number to analyze: ")


def _first_header_with_data(hdul: fits.HDUList) -> Tuple[int, fits.Header]:
    for idx, hdu in enumerate(hdul):
        if hdu.data is not None:
            return idx, hdu.header
    return 0, hdul[0].header


def _header_float(header: fits.Header, keys: Iterable[str], default: float = 0.0) -> float:
    for key in keys:
        if key in header:
            try:
                return float(header.get(key))
            except (TypeError, ValueError):
                pass
    return float(default)


def _header_int(header: fits.Header, keys: Iterable[str], default: Optional[int] = None) -> Optional[int]:
    for key in keys:
        if key in header:
            try:
                return int(float(header.get(key)))
            except (TypeError, ValueError):
                pass
    return default


def parse_sum_frame_count(path: Union[str, Path]) -> Optional[int]:
    """Return frame count from common summed-image filename patterns."""
    name = Path(path).name.lower()

    for pattern in (
        r"_(\d+)_sum(?:\.|_)",
        r"_(\d+)sum(?:\.|_)",
        r"_sum(\d+)(?:\.|_)",
        r"_stack(\d+)(?:\.|_)",
        r"_(\d+)_stack(?:\.|_)",
    ):
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))

    return None


def exposure_info(path: Union[str, Path], header: fits.Header, exposure_sec: float):
    """Return frame-count and total-exposure metadata."""
    header_frame_count = _header_int(
        header,
        (
            "NCOMBINE",
            "NFRAMES",
            "NSTACK",
            "STACKCNT",
            "NUMEXP",
            "FRAMES",
        ),
        None,
    )
    filename_frame_count = parse_sum_frame_count(path)

    if header_frame_count is not None and header_frame_count > 0:
        frame_count = header_frame_count
        exposure_source = "header_frame_count"
    elif filename_frame_count is not None:
        frame_count = filename_frame_count
        exposure_source = "filename_sum_count"
    else:
        frame_count = 1
        exposure_source = "single_frame"

    header_total_exposure = _header_float(
        header,
        ("TELAPSE", "ONTIME", "LIVETIME", "EXPTOTAL", "TOTALEXP", "TOTEXP", "SUMEXP"),
        0.0,
    )

    if header_total_exposure > 0:
        total_exposure_sec = header_total_exposure
        exposure_source = "header_total_exposure"
    else:
        total_exposure_sec = max(float(exposure_sec), 0.0) * max(int(frame_count), 0)

    return {
        "frame_count": int(frame_count),
        "total_exposure_sec": float(total_exposure_sec),
        "exposure_source": exposure_source,
    }


def read_fits_image(path: Union[str, Path]) -> ImageRecord:
    """Read a FITS image or cube and return image data plus common metadata."""
    path = Path(path).expanduser()
    with fits.open(path) as hdul:
        ext, header = _first_header_with_data(hdul)
        data = np.asarray(hdul[ext].data)
        primary = hdul[0].header
        # Some files keep metadata in the primary header even when data are in ext=1.
        h = header if len(header) > 6 else primary
        merged = primary.copy()
        merged.extend(h, update=True)
        h = merged

    object_name = h.get("ID", h.get("OBJECT", "test"))
    exposure_sec = _header_float(h, ("EXPTIME", "IEXP", "EXPOSURE"), 1.0)
    exposure = exposure_info(path, h, exposure_sec)
    dateobs = h.get("DATEOBS", h.get("DATE-OBS", h.get("DATE", "")))
    camera = h.get("INSTRUME", h.get("CAMERA", "Camera"))
    softname = h.get("SOFTNAME", "")

    ra_hours = 0.0
    dec_deg = 0.0
    if "CRVAL1" in h:
        ra_hours = _header_float(h, ("CRVAL1",), 0.0) / 15.0
        dec_deg = _header_float(h, ("CRVAL2",), 0.0)
    if "RA_TARG" in h:
        ra_hours = _header_float(h, ("RA_TARG",), 0.0) / 15.0
        dec_deg = _header_float(h, ("DEC_TARG",), 0.0)

    return ImageRecord(
        data=data,
        filename=path.name,
        path=path,
        header=h,
        object_name=str(object_name),
        camera=str(camera),
        exposure_sec=exposure_sec,
        frame_count=exposure["frame_count"],
        total_exposure_sec=exposure["total_exposure_sec"],
        exposure_source=exposure["exposure_source"],
        dateobs=str(dateobs),
        ra_hours=ra_hours,
        dec_deg=dec_deg,
        alt_deg=_header_float(h, ("ALT_TARG", "ALT", "OBJCTALT"), 0.0),
        az_deg=_header_float(h, ("AZ_TARG", "AZ", "OBJCTAZ"), 0.0),
        pa_deg=_header_float(h, ("PA", "PA_TARG", "POSANGLE"), 0.0),
        detector_temp_c=_header_float(h, ("TEMP_C", "CCD-TEMP", "SET-TEMP"), -20.0),
        softname=str(softname),
    )
