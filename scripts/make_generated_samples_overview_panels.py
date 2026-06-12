#!/usr/bin/env python
"""Create overview panels for generated sample renders.

Examples:
    python scripts/make_generated_samples_overview_panels.py \
      --render-metadata outputs/visualization/generated_samples_mace_relaxed_100/renders/render_metadata.csv \
      --output-dir outputs/visualization/generated_samples_mace_relaxed_100/panels \
      --panel-rows 4 \
      --panel-cols 5

    python scripts/make_generated_samples_overview_panels.py \
      --render-metadata outputs/visualization/generated_samples_mace_relaxed_100/renders/render_metadata.csv \
      --output-dir outputs/visualization/generated_samples_mace_relaxed_100/panels \
      --final-ids sample_003 sample_018 sample_021 sample_034 sample_047 sample_052 sample_071 sample_096 \
      --final-output-prefix ~/papers/stage_decoupled_crystal_aaai/figures/fig5_generated_samples_mace_relaxed
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Missing Pillow/PIL, required for panel assembly: {exc}")


LOGGER = logging.getLogger("make_generated_samples_overview_panels")
CHECK = "✓"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, default=str)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    width: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (20, 20, 20),
) -> None:
    text_w, _ = text_size(draw, text, font)
    x = xy[0] + max(0, (width - text_w) // 2)
    draw.text((x, xy[1]), text, fill=fill, font=font)


def overview_tags(row: dict[str, Any]) -> str:
    tags = []
    if parse_bool(row.get("struct_valid", "")):
        tags.append(f"Struct{CHECK}")
    if parse_bool(row.get("comp_valid", "")):
        tags.append(f"Comp{CHECK}")
    if parse_bool(row.get("unique", "")) and parse_bool(row.get("novel", "")):
        tags.append(f"UN{CHECK}")
    if parse_bool(row.get("mace_success", "")):
        tags.append(f"MACE{CHECK}")
    return " ".join(tags)


def final_tags(row: dict[str, Any]) -> str:
    valid = parse_bool(row.get("struct_valid", "")) and parse_bool(row.get("comp_valid", ""))
    un = parse_bool(row.get("unique", "")) and parse_bool(row.get("novel", ""))
    tags = []
    if valid:
        tags.append(f"Valid{CHECK}")
    if un:
        tags.append(f"UN{CHECK}")
    if parse_bool(row.get("mace_success", "")):
        tags.append(f"MACE{CHECK}")
    return " ".join(tags)


def paste_sample_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    row: dict[str, Any],
    *,
    x0: int,
    y0: int,
    tile_w: int,
    tile_h: int,
    image_h: int,
    formula_font: ImageFont.ImageFont,
    tag_font: ImageFont.ImageFont,
    final_style: bool,
) -> dict[str, Any] | None:
    render_path = Path(str(row.get("render_path", ""))).expanduser()
    if not render_path.exists():
        return {
            "kind": "missing_render",
            "sample_id": row.get("sample_id", ""),
            "render_path": str(render_path),
        }
    try:
        image = Image.open(render_path).convert("RGB")
        image = ImageOps.contain(image, (tile_w - 24, image_h - 16), method=Image.Resampling.LANCZOS)
        px = x0 + (tile_w - image.width) // 2
        py = y0 + 8
        canvas.paste(image, (px, py))
    except Exception as exc:
        return {
            "kind": "image_open_failure",
            "sample_id": row.get("sample_id", ""),
            "render_path": str(render_path),
            "error_message": str(exc),
        }

    formula = str(row.get("reduced_formula", "") or row.get("sample_id", ""))
    tags = final_tags(row) if final_style else overview_tags(row)
    label_y = y0 + image_h + 4
    draw_centered_text(draw, (x0, label_y), tile_w, formula, formula_font)
    draw_centered_text(draw, (x0, label_y + 30), tile_w, tags, tag_font, fill=(48, 48, 48))
    return None


def make_panel(
    rows: list[dict[str, Any]],
    output_prefix: Path,
    *,
    panel_rows: int,
    panel_cols: int,
    tile_width: int,
    tile_height: int,
    final_style: bool = False,
) -> list[dict[str, Any]]:
    margin = 30
    title_gap = 0
    image_h = tile_height - 72
    canvas_w = panel_cols * tile_width + 2 * margin
    canvas_h = panel_rows * tile_height + 2 * margin + title_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    formula_font = load_font(24 if final_style else 22, bold=True)
    tag_font = load_font(18 if final_style else 17, bold=False)

    failures: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        r = idx // panel_cols
        c = idx % panel_cols
        x0 = margin + c * tile_width
        y0 = margin + title_gap + r * tile_height
        failure = paste_sample_tile(
            canvas,
            draw,
            row,
            x0=x0,
            y0=y0,
            tile_w=tile_width,
            tile_h=tile_height,
            image_h=image_h,
            formula_font=formula_font,
            tag_font=tag_font,
            final_style=final_style,
        )
        if failure:
            failures.append(failure)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    canvas.save(png_path)
    canvas.save(pdf_path, "PDF", resolution=300.0)
    return failures


def chunk_rows(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[idx : idx + size] for idx in range(0, len(rows), size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--render-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualization/generated_samples_mace_relaxed_100/panels"))
    parser.add_argument("--panel-rows", type=int, default=4)
    parser.add_argument("--panel-cols", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=380)
    parser.add_argument("--final-ids", nargs="*", default=[])
    parser.add_argument("--final-output-prefix", type=Path, default=Path("~/papers/stage_decoupled_crystal_aaai/figures/fig5_generated_samples_mace_relaxed"))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(args.render_metadata.expanduser())
    rows_by_id = {str(row.get("sample_id", "")): row for row in rows}
    failures: list[dict[str, Any]] = []

    per_panel = args.panel_rows * args.panel_cols
    for panel_idx, panel_rows in enumerate(chunk_rows(rows, per_panel)):
        prefix = args.output_dir / f"panel_{panel_idx:02d}"
        failures.extend(
            make_panel(
                panel_rows,
                prefix,
                panel_rows=args.panel_rows,
                panel_cols=args.panel_cols,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                final_style=False,
            )
        )
        LOGGER.info("Wrote %s.png/.pdf", prefix)

    final_outputs: dict[str, str] = {}
    if args.final_ids:
        missing = [sample_id for sample_id in args.final_ids if sample_id not in rows_by_id]
        if missing:
            failures.append({"kind": "missing_final_ids", "sample_ids": missing})
            raise SystemExit(f"Final ids not found in metadata: {missing}")
        final_rows = [rows_by_id[sample_id] for sample_id in args.final_ids]
        final_prefix = args.final_output_prefix.expanduser()
        final_cols = 4
        final_rows_count = max(1, math.ceil(len(final_rows) / final_cols))
        failures.extend(
            make_panel(
                final_rows,
                final_prefix,
                panel_rows=final_rows_count,
                panel_cols=final_cols,
                tile_width=args.tile_width + 40,
                tile_height=args.tile_height,
                final_style=True,
            )
        )
        final_outputs = {
            "png": str(final_prefix.with_suffix(".png").resolve(strict=False)),
            "pdf": str(final_prefix.with_suffix(".pdf").resolve(strict=False)),
        }
        LOGGER.info("Wrote final Figure 5 panel to %s", final_prefix)

    failures_path = args.output_dir / "panel_failures.jsonl"
    write_jsonl(failures_path, failures)
    summary = {
        "created_at": now_iso(),
        "metadata_path": str(args.render_metadata.expanduser().resolve(strict=False)),
        "input_render_count": len(rows),
        "overview_panel_count": len(chunk_rows(rows, per_panel)),
        "panel_rows": args.panel_rows,
        "panel_cols": args.panel_cols,
        "final_ids": args.final_ids,
        "final_outputs": final_outputs,
        "failure_count": len(failures),
        "failures_jsonl": str(failures_path.resolve(strict=False)),
    }
    write_json(args.output_dir / "panel_summary.json", summary)
    if failures:
        LOGGER.warning("Panel generation completed with %d recorded issues.", len(failures))


if __name__ == "__main__":
    main()
