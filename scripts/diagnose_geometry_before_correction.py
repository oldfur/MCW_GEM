#!/usr/bin/env python3
"""Diagnose raw generated geometry before ZBL/local-repulsion correction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mp20.geometry_diagnostics import (  # noqa: E402
    diagnose_geometry_records,
    records_from_npz,
    save_raw_geometry_npz,
)


def records_from_cif_directory(cif_dir: str):
    from pymatgen.core import Structure

    root = Path(cif_dir)
    paths = sorted(path for path in root.rglob("*.cif") if path.is_file())
    records = []
    failures = []
    for path in paths:
        try:
            structure = Structure.from_file(path)
            records.append(
                {
                    "sample_id": str(path.relative_to(root)),
                    "source": str(path),
                    # pymatgen stores lattice vectors as rows; the utility uses
                    # columns so cart = frac @ L.T.
                    "lattice": structure.lattice.matrix.T,
                    "frac_coords": structure.frac_coords,
                    "num_atoms": len(structure),
                    "atom_types": structure.atomic_numbers,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "sample_id": str(path.relative_to(root)),
                    "source": str(path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return records, failures, len(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PBC geometry diagnostics for structures before geometry correction."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--cif-dir", type=str, help="Recursively read .cif files.")
    inputs.add_argument("--npz", type=str, help="Read raw lattice/frac_coords/num_atoms arrays.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--train-csv",
        type=str,
        default="",
        help="Optional MP-20 CSV used to compute V/N conditional statistics.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip optional PNG plots.")
    parser.add_argument(
        "--save-normalized-npz",
        action="store_true",
        help="Also save parsed input using the padded canonical NPZ layout.",
    )
    return parser


def _print_core_summary(summary):
    keys = (
        "ratio_dmin_lt_0.7",
        "ratio_pairs_lt_0.7_ge_2",
        "median_volume_per_atom",
        "median_atom_number_density",
        "p05_d_min",
    )
    print("[PreCorrectionGeometry] summary")
    for key in keys:
        print(f"  {key}: {summary.get(key)}")


def main() -> None:
    args = build_parser().parse_args()
    if args.cif_dir:
        records, failures, total = records_from_cif_directory(args.cif_dir)
    else:
        records = records_from_npz(args.npz)
        failures = []
        total = len(records)

    _, summary, _ = diagnose_geometry_records(
        records=records,
        output_dir=args.output_dir,
        failures=failures,
        total_samples=total,
        train_csv=args.train_csv,
        make_plots=not args.no_plots,
    )
    if args.save_normalized_npz:
        save_raw_geometry_npz(
            records,
            str(Path(args.output_dir) / "geometry_pre_correction_raw.npz"),
        )
    _print_core_summary(summary)
    print(f"[PreCorrectionGeometry] outputs: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
