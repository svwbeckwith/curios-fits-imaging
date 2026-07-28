"""Command-line interface for curios-fits-imaging."""

import argparse
from pathlib import Path

from .analysis import ANALYSIS_MODES, normalize_analysis_mode
from .config import ImagingConfig
from .run import ImagingRun


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze a folder of astronomical FITS images."
    )

    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing FITS images.",
    )

    parser.add_argument(
        "--kind",
        default="science",
        help=(
            "Image kind to analyze. Use science, science_single, science_sum, "
            "focus, dark, flat, or rotation_blur. Defaults to science."
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze every FITS file in the folder, regardless of image kind.",
    )

    parser.add_argument(
        "--mode",
        default=None,
        help=(
            "Analysis mode to run. Choices are source_detection, statistics, "
            "focus, and rotation_blur. Defaults from each file's image kind."
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List classified FITS files without analyzing them.",
    )

    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Read FITS headers when listing files to include exposure metadata.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to analyze.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV file for the summary.",
    )

    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=None,
        help="Peak detection threshold in background sigma units.",
    )

    parser.add_argument(
        "--max-peaks",
        type=int,
        default=None,
        help="Maximum number of detected peaks per image.",
    )

    parser.add_argument(
        "--peak-separation",
        type=float,
        default=None,
        help="Minimum separation between accepted peaks in pixels.",
    )

    parser.add_argument(
        "--peak-sharp",
        type=float,
        default=None,
        help="Minimum sharpness value for candidate peaks.",
    )

    parser.add_argument(
        "--fit-method",
        choices=("gaussian", "moments"),
        default=None,
        help="Peak fitting method.",
    )

    parser.add_argument(
        "--zero-mag-counts",
        type=float,
        default=None,
        help="Photometric zero point counts.",
    )

    parser.add_argument(
        "--pixel-arcsec",
        type=float,
        default=None,
        help="Pixel scale in arcseconds per pixel.",
    )

    return parser


def config_from_args(args, data_root):
    """Build ImagingConfig from command-line options."""
    config = ImagingConfig(data_root=data_root)

    for attr in (
        "threshold_sigma",
        "max_peaks",
        "peak_separation",
        "peak_sharp",
        "fit_method",
        "zero_mag_counts",
        "pixel_arcsec",
    ):
        value = getattr(args, attr)
        if value is not None:
            setattr(config, attr, value)

    return config


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    folder = args.folder.expanduser().resolve()

    if not folder.exists():
        parser.error(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        parser.error(f"Not a directory: {folder}")

    config = config_from_args(args, data_root=folder.parent)
    run = ImagingRun(folder, read_headers=args.metadata)
    kind = None if args.all else args.kind
    try:
        mode = normalize_analysis_mode(args.mode) if args.mode is not None else None
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Folder: {folder}")
    print(f"Found {len(run.files)} FITS files")
    print(f"Available image types: {', '.join(run.kinds()) or 'none'}")
    print(f"Analysis mode: {mode or 'auto'}")
    print(
        "Peak config: threshold_sigma={:.2f}, max_peaks={}, separation={:.1f}, sharp={:.2f}, fit={}".format(
            config.threshold_sigma,
            config.max_peaks,
            config.peak_separation,
            config.peak_sharp,
            config.fit_method,
        )
    )

    if args.list:
        print()
        columns = ["file", "kind", "frame_count", "exposure_sec", "total_exposure_sec", "include"]
        print(run.file_table.reindex(columns=columns))
        return 0

    selected_files = run.files_for(kind=kind)
    if not selected_files:
        label = "all files" if kind is None else f"kind={kind}"
        parser.error(f"No FITS files selected for {label}. Use --list to inspect file kinds.")

    summary = run.analyze_all(
        config,
        kind=kind,
        mode=mode,
        max_files=args.max_files,
    )

    output = args.output
    if output is None:
        output = folder / "imaging_run_summary.csv"

    summary.to_csv(output, index=False)

    print()
    print(summary)
    print()
    print(f"Summary written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
