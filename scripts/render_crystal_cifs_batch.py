#!/usr/bin/env python
"""Batch-render selected generated crystal CIFs.

Example:
    python scripts/render_crystal_cifs_batch.py \
      --selected-csv outputs/visualization/generated_samples_mace_relaxed_100/selected/selected_samples.csv \
      --output-dir outputs/visualization/generated_samples_mace_relaxed_100/renders \
      --backend auto \
      --supercell auto \
      --image-size 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from pymatgen.core import Structure
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing pymatgen. Use the project environment before rendering. "
        f"Original error: {exc}"
    )


LOGGER = logging.getLogger("render_crystal_cifs_batch")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, default=json_default)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=json_default) + "\n")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def resolve_path(path_text: Any) -> Path:
    path = Path(str(path_text)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(strict=False)


def collect_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if args.selected_csv:
        for idx, row in enumerate(read_csv_rows(args.selected_csv.expanduser())):
            path_text = row.get("copied_cif_path") or row.get("source_cif_path") or row.get("cif_path")
            if not path_text:
                continue
            record = dict(row)
            record.setdefault("sample_id", f"sample_{idx:03d}")
            record["render_input_cif_path"] = str(resolve_path(path_text))
            records.append(record)
        return records

    if not args.cif_dir:
        raise SystemExit("Provide either --selected-csv or --cif-dir.")
    cif_paths = sorted(args.cif_dir.expanduser().rglob("*.cif"), key=lambda path: str(path))
    for idx, path in enumerate(cif_paths):
        records.append(
            {
                "sample_id": f"sample_{idx:03d}",
                "render_input_cif_path": str(path.resolve(strict=False)),
                "source_cif_path": str(path.resolve(strict=False)),
                "copied_cif_path": str(path.resolve(strict=False)),
            }
        )
    return records


def parse_supercell(value: str, num_sites: int) -> tuple[int, int, int]:
    value = value.strip().lower()
    if value == "auto":
        return (2, 2, 2) if num_sites <= 20 else (1, 1, 1)
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--supercell must be auto or a,b,c")
    dims = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("supercell dimensions must be positive")
    return dims  # type: ignore[return-value]


def ensure_ase_atoms(structure: Structure, supercell: tuple[int, int, int]):
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:
        raise RuntimeError(f"ASE bridge unavailable: {exc}") from exc
    atoms = AseAtomsAdaptor.get_atoms(structure)
    if supercell != (1, 1, 1):
        atoms = atoms.repeat(supercell)
    atoms.wrap()
    return atoms


def render_with_povray(atoms: Any, out_png: Path, args: argparse.Namespace) -> str:
    if shutil.which("povray") is None:
        raise RuntimeError("POV-Ray executable not found")
    try:
        from ase.io import write
    except ImportError as exc:
        raise RuntimeError(f"ASE writer unavailable: {exc}") from exc

    out_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crystal_pov_") as tmp:
        pov_path = Path(tmp) / (out_png.stem + ".pov")
        kwargs = {
            "rotation": args.rotation,
            "show_unit_cell": 2,
            "radii": args.radius_scale,
            "canvas_width": args.image_size,
            "canvas_height": args.image_size,
            "transparent": False,
            "run_povray": True,
        }
        write(str(pov_path), atoms, format="pov", **kwargs)
        generated_png = pov_path.with_suffix(".png")
        if not generated_png.exists():
            raise RuntimeError(f"POV-Ray did not create {generated_png}")
        shutil.copy2(generated_png, out_png)
    return "ase_povray"


def render_with_matplotlib(atoms: Any, out_png: Path, args: argparse.Namespace) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ase.visualize.plot import plot_atoms
    except ImportError as exc:
        raise RuntimeError(f"ASE matplotlib rendering unavailable: {exc}") from exc

    out_png.parent.mkdir(parents=True, exist_ok=True)
    dpi = 200
    size_inches = args.image_size / dpi
    fig, ax = plt.subplots(figsize=(size_inches, size_inches), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plot_atoms(
        atoms,
        ax,
        rotation=args.rotation,
        show_unit_cell=2,
        radii=args.radius_scale,
    )
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_png, dpi=dpi, facecolor="white", pad_inches=0)
    plt.close(fig)
    return "ase_matplotlib"


def render_with_simple_projection(structure: Structure, out_png: Path, args: argparse.Namespace) -> str:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(f"PIL fallback unavailable: {exc}") from exc

    out_png.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (args.image_size, args.image_size), "white")
    draw = ImageDraw.Draw(image)
    coords = structure.frac_coords[:, :2] if len(structure) else []
    margin = int(args.image_size * 0.12)
    box = [margin, margin, args.image_size - margin, args.image_size - margin]
    draw.rectangle(box, outline=(40, 40, 40), width=max(2, args.image_size // 250))
    palette = [
        (67, 115, 191),
        (218, 82, 69),
        (78, 157, 92),
        (238, 174, 68),
        (139, 91, 177),
        (89, 177, 197),
        (180, 180, 180),
    ]
    elements = sorted({str(site.specie) for site in structure})
    color_by_el = {el: palette[idx % len(palette)] for idx, el in enumerate(elements)}
    radius = max(8, args.image_size // 55)
    span = args.image_size - 2 * margin
    for site, frac in zip(structure, coords):
        x = margin + int(float(frac[0] % 1.0) * span)
        y = margin + int(float(frac[1] % 1.0) * span)
        color = color_by_el[str(site.specie)]
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=(20, 20, 20), width=2)
    image.save(out_png)
    return "pymatgen_pil_projection"


def normalize_png_canvas(path: Path, image_size: int) -> None:
    """Place a rendered PNG on a square white canvas of the requested size."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return
    image = Image.open(path).convert("RGBA")
    if image.size == (image_size, image_size):
        image.convert("RGB").save(path)
        return
    image = ImageOps.contain(image, (image_size, image_size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), "white")
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image).convert("RGB")
    else:
        image = image.convert("RGB")
    canvas.paste(image, ((image_size - image.width) // 2, (image_size - image.height) // 2))
    canvas.save(path)


def render_one(record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    sample_id = str(record.get("sample_id") or "sample")
    cif_path = resolve_path(record["render_input_cif_path"])
    out_png = args.output_dir / f"{sample_id}.png"
    try:
        structure = Structure.from_file(str(cif_path))
        supercell = parse_supercell(args.supercell, len(structure))
        backend_used = ""
        if args.backend in {"auto", "povray"}:
            try:
                atoms = ensure_ase_atoms(structure, supercell)
                backend_used = render_with_povray(atoms, out_png, args)
            except Exception as exc:
                if args.backend == "povray":
                    raise
                LOGGER.warning("POV-Ray fallback for %s: %s", sample_id, exc)
        if not backend_used and args.backend in {"auto", "matplotlib"}:
            try:
                atoms = ensure_ase_atoms(structure, supercell)
                backend_used = render_with_matplotlib(atoms, out_png, args)
            except Exception as exc:
                if args.backend == "matplotlib":
                    raise
                LOGGER.warning("Matplotlib fallback for %s: %s", sample_id, exc)
        if not backend_used:
            backend_used = render_with_simple_projection(structure, out_png, args)
        normalize_png_canvas(out_png, args.image_size)

        meta = {
            "sample_id": sample_id,
            "reduced_formula": record.get("reduced_formula") or structure.composition.reduced_formula,
            "source_cif_path": record.get("source_cif_path", str(cif_path)),
            "copied_cif_path": record.get("copied_cif_path", str(cif_path)),
            "render_path": str(out_png.resolve(strict=False)),
            "render_backend": backend_used,
            "supercell": ",".join(str(v) for v in supercell),
            "struct_valid": record.get("struct_valid", ""),
            "comp_valid": record.get("comp_valid", ""),
            "unique": record.get("unique", ""),
            "novel": record.get("novel", ""),
            "mace_success": record.get("mace_success", ""),
            "num_sites": record.get("num_sites", len(structure)),
            "density": record.get("density", structure.density),
            "min_pbc_distance": record.get("min_pbc_distance", ""),
        }
        write_json(args.output_dir / f"{sample_id}_meta.json", meta)
        return meta, None
    except Exception as exc:
        return None, {
            "kind": "render_failure",
            "sample_id": sample_id,
            "cif_path": str(cif_path),
            "error_message": str(exc),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--selected-csv", type=Path, default=None)
    parser.add_argument("--cif-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualization/generated_samples_mace_relaxed_100/renders"))
    parser.add_argument("--backend", choices=["auto", "povray", "matplotlib", "simple"], default="auto")
    parser.add_argument("--supercell", default="auto")
    parser.add_argument("--image-size", type=int, default=1000)
    parser.add_argument("--rotation", default="15x,25y,0z")
    parser.add_argument("--radius-scale", type=float, default=0.65)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_inputs(args)
    if args.max_samples is not None:
        records = records[: max(0, args.max_samples)]
    LOGGER.info("Rendering %d CIFs to %s", len(records), args.output_dir)

    metas: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        meta, failure = render_one(record, args)
        if meta:
            metas.append(meta)
        if failure:
            failures.append(failure)
            LOGGER.error("Failed %s: %s", failure.get("sample_id"), failure.get("error_message"))

    fieldnames = [
        "sample_id",
        "reduced_formula",
        "source_cif_path",
        "copied_cif_path",
        "render_path",
        "render_backend",
        "supercell",
        "struct_valid",
        "comp_valid",
        "unique",
        "novel",
        "mace_success",
        "num_sites",
        "density",
        "min_pbc_distance",
    ]
    write_csv(args.output_dir / "render_metadata.csv", metas, fieldnames)
    write_jsonl(args.output_dir / "render_failures.jsonl", failures)
    summary = {
        "created_at": now_iso(),
        "input_count": len(records),
        "rendered_count": len(metas),
        "failure_count": len(failures),
        "render_backends": dict(sorted({backend: sum(m["render_backend"] == backend for m in metas) for backend in {m["render_backend"] for m in metas}}.items())),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "failures_jsonl": str((args.output_dir / "render_failures.jsonl").resolve(strict=False)),
    }
    write_json(args.output_dir / "render_summary.json", summary)
    if failures:
        raise SystemExit(f"Rendered {len(metas)} samples, but {len(failures)} failed. See render_failures.jsonl.")
    LOGGER.info("Rendered %d samples successfully.", len(metas))


if __name__ == "__main__":
    main()
