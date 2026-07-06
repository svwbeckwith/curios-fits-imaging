from dataclasses import dataclass
from pathlib import Path

@dataclass
class ImagingConfig:
    """Configuration values used by the cleaned FITS imaging notebook."""
    telescope: str = "90mm SVBony telescope"
    pixel_um: float = 3.76
    pixel_arcsec: float = 1.55
    xformat: int = 9576
    yformat: int = 6380
    maxcounts: int = 65535
    aperture_cm: float = 9.0
    focal_length_cm: float = 50.0
    zero_mag_counts: float = 115_000_000.0
    latitude_deg: float = 37.78111
    longitude_deg: float = -122.39139
    data_root: Path = Path.cwd() / "Data"

    @property
    def figures_dir(self) -> Path:
        return Path(self.data_root) / "Figures"
