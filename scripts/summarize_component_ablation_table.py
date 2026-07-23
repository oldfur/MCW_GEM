#!/usr/bin/env python3
"""Summarize MP-20 component ablation metrics into CSV and LaTeX."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


CONFIGS = [
    (
        "raw_geometry_raw_logits",
        "Raw geometry + raw logits",
        "No",
        "Raw argmax",
    ),
    (
        "raw_geometry_constrained_decode",
        "Raw geometry + constrained decoding",
        "No",
        "Constrained search",
    ),
    (
        "corrected_geometry_raw_logits",
        "Corrected geometry + raw logits",
        "Yes",
        "Raw argmax",
    ),
    (
        "full_pipeline",
        "Full pipeline",
        "Yes",
        "Constrained search",
    ),
]

SOFTZ_ZBL_CONFIGS = [
    (
        "softz_geometry_raw_logits",
        "Soft-Z + raw logits",
        "Soft-Z ZBL",
        "Raw argmax",
    ),
    (
        "full_pipeline",
        "Full pipeline",
        "Soft-Z ZBL",
        "Constrained search",
    ),
]

CONFIG_PRESETS = {
    "default": CONFIGS,
    "softz_zbl": SOFTZ_ZBL_CONFIGS,
}


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def get_metric(metrics: dict[str, Any], key: str, config_name: str) -> Any:
    if key not in metrics or metrics[key] is None:
        warn(f"{config_name}: missing {key}")
        return None
    return metrics[key]


def percent(metrics: dict[str, Any], key: str, config_name: str) -> str:
    value = get_metric(metrics, key, config_name)
    if value is None:
        return "--"
    return f"{100.0 * float(value):.2f}"


def scalar(metrics: dict[str, Any], key: str, config_name: str) -> str:
    value = get_metric(metrics, key, config_name)
    if value is None:
        return "--"
    return str(value)


def search_fail(metrics: dict[str, Any], config_name: str) -> str:
    if metrics.get("atom_decode_mode") == "raw_argmax" or metrics.get("raw_argmax_mode"):
        return "n/a"
    value = get_metric(metrics, "search_failure_rate", config_name)
    if value is None:
        return "--"
    return f"{100.0 * float(value):.2f}"


def build_rows(root: Path, configs: list[tuple[str, str, str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for config_name, label, geometry, decoding in configs:
        metrics_path = root / config_name / "metrics.json"
        if not metrics_path.exists():
            warn(f"{config_name}: metrics.json not found at {metrics_path}")
            metrics: dict[str, Any] = {}
        else:
            metrics = load_metrics(metrics_path)
        rows.append(
            {
                "Configuration": label,
                "Geometry correction": geometry,
                "Atom decoding": decoding,
                "Sample count": scalar(metrics, "num_generated", config_name),
                "Struct. Valid (%)": percent(metrics, "structural_valid_rate", config_name),
                "Comp. Valid (%)": percent(metrics, "composition_valid_rate", config_name),
                "Total Valid (%)": percent(metrics, "total_valid_rate", config_name),
                "UN rate (%)": percent(metrics, "UN_rate", config_name),
                "All-H (%)": percent(metrics, "all_H_rate", config_name),
                "Close-contact fail (%)": percent(metrics, "close_contact_fail_rate", config_name),
                "Search fail (%)": search_fail(metrics, config_name),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def write_tex(path: Path, rows: list[dict[str, str]]) -> None:
    headers = list(rows[0].keys())
    align = "lllrrrrrrrr"
    lines = [
        "\\begin{tabular}{" + align + "}",
        "\\toprule",
        " & ".join(tex_escape(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(str(row[header])) for header in headers) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/ablation_component_diagnostics"),
        help="Directory containing the four ablation output subdirectories.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV output path. Defaults to <root>/component_ablation_summary.csv.",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=None,
        help="LaTeX output path. Defaults to <root>/component_ablation_table.tex.",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(CONFIG_PRESETS.keys()),
        default="default",
        help="Ablation table preset. Use softz_zbl for the two-row soft-Z ZBL diagnostic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.root, CONFIG_PRESETS[args.preset])
    output_csv = args.output_csv or args.root / "component_ablation_summary.csv"
    output_tex = args.output_tex or args.root / "component_ablation_table.tex"
    write_csv(output_csv, rows)
    write_tex(output_tex, rows)
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_tex}")


if __name__ == "__main__":
    main()
