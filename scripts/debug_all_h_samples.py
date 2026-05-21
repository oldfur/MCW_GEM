#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

try:
    from pymatgen.core import Structure
except ImportError as exc:
    raise SystemExit(
        "pymatgen is required for scripts/debug_all_h_samples.py. "
        "Please run it inside the project environment."
    ) from exc


SAMPLE_INDEX_RE = re.compile(r"sample_(\d+)")


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def infer_sample_index(path):
    match = SAMPLE_INDEX_RE.search(path.stem)
    if not match:
        return None
    return int(match.group(1))


def summarize_cifs(sample_dir):
    cif_paths = sorted(sample_dir.rglob("*.cif"))
    rows = []
    element_frequency = Counter()
    all_h_files = []
    single_element_files = []
    failures = []

    for cif_path in cif_paths:
        sample_index = infer_sample_index(cif_path)
        try:
            struct = Structure.from_file(cif_path)
            atomic_numbers = [int(z) for z in struct.atomic_numbers]
            symbols = [str(site.specie) for site in struct]
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append({"file": str(cif_path), "error": str(exc)})
            continue

        counts = Counter(symbols)
        element_frequency.update(symbols)
        is_all_h = len(symbols) > 0 and all(symbol == "H" for symbol in symbols)
        is_single_element = len(counts) == 1
        if is_all_h:
            all_h_files.append(str(cif_path))
        if is_single_element:
            single_element_files.append(str(cif_path))

        rows.append(
            {
                "file": str(cif_path),
                "sample_index": sample_index,
                "num_atoms": len(symbols),
                "symbols": symbols,
                "atomic_numbers": atomic_numbers,
                "species_counts": dict(sorted(counts.items())),
                "all_h": is_all_h,
                "single_element": is_single_element,
            }
        )

    return {
        "rows": rows,
        "cif_paths": [str(p) for p in cif_paths],
        "failures": failures,
        "element_frequency": dict(sorted(element_frequency.items())),
        "all_h_files": all_h_files,
        "single_element_files": single_element_files,
    }


def summarize_debug_dir(debug_dir):
    data = {
        "exists": debug_dir.exists(),
        "analyze_rows": [],
        "all_h_rows": [],
        "roundtrip_rows": [],
        "empty_graph_rows": [],
        "pt_rows": [],
    }
    if not debug_dir.exists():
        return data

    data["analyze_rows"] = load_jsonl(debug_dir / "analyze_test_samples.jsonl")
    data["all_h_rows"] = load_jsonl(debug_dir / "all_h_samples.jsonl")
    data["roundtrip_rows"] = load_jsonl(debug_dir / "cif_roundtrip.jsonl")
    data["empty_graph_rows"] = load_jsonl(debug_dir / "equiformer_empty_graph.jsonl")

    if torch is not None:
        for pt_path in sorted(debug_dir.glob("all_h_sample_*.pt")):
            try:
                payload = torch.load(pt_path, map_location="cpu")
            except Exception as exc:  # pragma: no cover - diagnostic path
                data["pt_rows"].append({"file": str(pt_path), "error": str(exc)})
                continue

            logits = payload.get("logits_valid")
            probs = payload.get("probs_valid")
            row = {
                "file": str(pt_path),
                "sample_global_index": payload.get("sample_global_index"),
                "sample_local_index": payload.get("sample_local_index"),
                "round_index": payload.get("round_index"),
                "logits_shape": list(logits.shape) if hasattr(logits, "shape") else None,
                "probs_shape": list(probs.shape) if hasattr(probs, "shape") else None,
                "logits_has_nan": bool(logits is not None and (~logits.isfinite()).any().item()),
                "probs_has_nan": bool(probs is not None and (~probs.isfinite()).any().item()),
                "logits_all_zero": bool(logits is not None and torch.allclose(logits, torch.zeros_like(logits))),
            }
            data["pt_rows"].append(row)

    return data


def join_debug_to_cifs(cif_summary, debug_summary):
    by_index = defaultdict(dict)
    for row in cif_summary["rows"]:
        if row["sample_index"] is not None:
            by_index[row["sample_index"]]["cif"] = row

    for row in debug_summary["analyze_rows"]:
        if row.get("sample_global_index") is not None:
            by_index[int(row["sample_global_index"])]["analyze"] = row

    for row in debug_summary["pt_rows"]:
        if row.get("sample_global_index") is not None:
            by_index[int(row["sample_global_index"])]["pt"] = row

    return by_index


def print_report(sample_dir, debug_dir, cif_summary, debug_summary, joined):
    total_samples = len(cif_summary["rows"])
    all_h_count = sum(1 for row in cif_summary["rows"] if row["all_h"])
    single_element_count = sum(1 for row in cif_summary["rows"] if row["single_element"])

    print(f"Sample dir: {sample_dir}")
    print(f"Debug dir: {debug_dir}")
    print(f"Total CIF samples: {total_samples}")
    print(f"All-H CIF samples: {all_h_count}")
    print(f"Single-element CIF samples: {single_element_count}")
    print(f"Element frequency: {cif_summary['element_frequency']}")

    if cif_summary["failures"]:
        print(f"CIF parse failures: {len(cif_summary['failures'])}")
        for row in cif_summary["failures"][:10]:
            print(f"  - {row['file']}: {row['error']}")

    if cif_summary["all_h_files"]:
        print("All-H CIF files:")
        for path in cif_summary["all_h_files"]:
            print(f"  - {path}")

    if debug_summary["exists"]:
        print(f"Debug analyze rows: {len(debug_summary['analyze_rows'])}")
        print(f"Debug all-H rows: {len(debug_summary['all_h_rows'])}")
        print(f"Debug CIF roundtrip rows: {len(debug_summary['roundtrip_rows'])}")
        print(f"Debug empty-graph rows: {len(debug_summary['empty_graph_rows'])}")
        print(f"Saved all-H tensor dumps: {len(debug_summary['pt_rows'])}")

        matched = []
        for sample_index, bundle in sorted(joined.items()):
            cif_row = bundle.get("cif")
            analyze_row = bundle.get("analyze")
            if not cif_row or not analyze_row:
                continue
            if cif_row["all_h"] or analyze_row.get("all_H"):
                matched.append(
                    {
                        "sample_index": sample_index,
                        "cif_file": cif_row["file"],
                        "species_counts": cif_row["species_counts"],
                        "debug_species_counts": analyze_row.get("species_counts"),
                        "pt_file": bundle.get("pt", {}).get("file"),
                    }
                )

        if matched:
            print("Matched all-H sample records:")
            for row in matched:
                print(
                    f"  - sample {row['sample_index']}: {row['cif_file']} | "
                    f"CIF={row['species_counts']} | debug={row['debug_species_counts']} | "
                    f"logits={row['pt_file']}"
                )


def main():
    parser = argparse.ArgumentParser(description="Inspect sampling outputs for all-H crystal bugs.")
    parser.add_argument("sample_dir", type=Path, help="Sampling output directory containing CIF files.")
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional atom-type debug directory. Defaults to <sample_dir>/atom_type_debug.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the combined report as JSON.",
    )
    args = parser.parse_args()

    sample_dir = args.sample_dir.resolve()
    debug_dir = (args.debug_dir or (sample_dir / "atom_type_debug")).resolve()

    cif_summary = summarize_cifs(sample_dir)
    debug_summary = summarize_debug_dir(debug_dir)
    joined = join_debug_to_cifs(cif_summary, debug_summary)

    print_report(sample_dir, debug_dir, cif_summary, debug_summary, joined)

    if args.output_json is not None:
        report = {
            "sample_dir": str(sample_dir),
            "debug_dir": str(debug_dir),
            "cif_summary": cif_summary,
            "debug_summary": debug_summary,
            "joined": {str(k): v for k, v in joined.items()},
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
