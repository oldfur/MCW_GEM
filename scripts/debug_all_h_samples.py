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


def infer_default_debug_dir(sample_dir):
    direct_debug_dir = sample_dir / "atom_type_debug"
    if direct_debug_dir.exists():
        return direct_debug_dir

    if sample_dir.name.startswith("epoch_"):
        sibling_debug_dir = sample_dir.parent / "atom_type_debug"
        if sibling_debug_dir.exists():
            return sibling_debug_dir

    return direct_debug_dir


def find_worker_dirs(sample_dir):
    if not sample_dir.exists():
        return []
    return sorted(
        path
        for path in sample_dir.iterdir()
        if path.is_dir() and path.name.startswith("worker_")
    )


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


def summarize_single_run(sample_dir, debug_dir):
    cif_summary = summarize_cifs(sample_dir)
    debug_summary = summarize_debug_dir(debug_dir)
    joined = join_debug_to_cifs(cif_summary, debug_summary)
    total_samples = len(cif_summary["rows"])
    all_h_count = sum(1 for row in cif_summary["rows"] if row["all_h"])
    single_element_count = sum(1 for row in cif_summary["rows"] if row["single_element"])
    return {
        "sample_dir": str(sample_dir),
        "debug_dir": str(debug_dir),
        "cif_summary": cif_summary,
        "debug_summary": debug_summary,
        "joined": joined,
        "total_samples": total_samples,
        "all_h_count": all_h_count,
        "single_element_count": single_element_count,
    }


def summarize_multi_worker(sample_dir, worker_dirs):
    element_frequency = Counter()
    worker_reports = []
    total_samples = 0
    total_all_h = 0
    total_single_element = 0
    total_failures = 0
    total_debug_all_h_rows = 0
    all_h_files = []

    for worker_dir in worker_dirs:
        worker_sample_dir = worker_dir / "epoch_0"
        worker_debug_dir = infer_default_debug_dir(worker_dir)
        worker_report = summarize_single_run(worker_sample_dir, worker_debug_dir)
        worker_reports.append(
            {
                "worker": worker_dir.name,
                **worker_report,
            }
        )

        total_samples += worker_report["total_samples"]
        total_all_h += worker_report["all_h_count"]
        total_single_element += worker_report["single_element_count"]
        total_failures += len(worker_report["cif_summary"]["failures"])
        total_debug_all_h_rows += len(worker_report["debug_summary"]["all_h_rows"])
        element_frequency.update(worker_report["cif_summary"]["element_frequency"])
        all_h_files.extend(worker_report["cif_summary"]["all_h_files"])

    return {
        "sample_dir": str(sample_dir),
        "worker_reports": worker_reports,
        "aggregate": {
            "total_samples": total_samples,
            "all_h_count": total_all_h,
            "single_element_count": total_single_element,
            "cif_parse_failures": total_failures,
            "debug_all_h_rows": total_debug_all_h_rows,
            "element_frequency": dict(sorted(element_frequency.items())),
            "all_h_files": all_h_files,
        },
    }


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


def print_multi_worker_report(sample_dir, multi_report):
    aggregate = multi_report["aggregate"]

    print(f"Multi-worker sample dir: {sample_dir}")
    print(f"Detected workers: {len(multi_report['worker_reports'])}")
    print(f"Total CIF samples: {aggregate['total_samples']}")
    print(f"All-H CIF samples: {aggregate['all_h_count']}")
    print(f"Single-element CIF samples: {aggregate['single_element_count']}")
    print(f"CIF parse failures: {aggregate['cif_parse_failures']}")
    print(f"Debug all-H rows: {aggregate['debug_all_h_rows']}")
    print(f"Element frequency: {aggregate['element_frequency']}")

    if aggregate["all_h_files"]:
        print("All-H CIF files:")
        for path in aggregate["all_h_files"]:
            print(f"  - {path}")

    print("Per-worker summary:")
    for worker_report in multi_report["worker_reports"]:
        debug_summary = worker_report["debug_summary"]
        print(
            f"  - {worker_report['worker']}: samples={worker_report['total_samples']} "
            f"all_H={worker_report['all_h_count']} single_element={worker_report['single_element_count']} "
            f"debug_all_H={len(debug_summary['all_h_rows'])} sample_dir={worker_report['sample_dir']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Inspect sampling outputs for all-H crystal bugs.")
    parser.add_argument(
        "sample_dir",
        type=Path,
        help="Sampling output directory. Accepts a single-run CIF dir, a worker dir, or a multi-worker base save dir.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional atom-type debug directory for single-run mode. In multi-worker mode, per-worker debug dirs are auto-detected.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the combined report as JSON.",
    )
    args = parser.parse_args()

    sample_dir = args.sample_dir.resolve()
    worker_dirs = find_worker_dirs(sample_dir)

    if worker_dirs:
        multi_report = summarize_multi_worker(sample_dir, worker_dirs)
        print_multi_worker_report(sample_dir, multi_report)

        if args.output_json is not None:
            output_report = {
                "mode": "multi_worker",
                "sample_dir": str(sample_dir),
                "worker_reports": [
                    {
                        **worker_report,
                        "joined": {str(k): v for k, v in worker_report["joined"].items()},
                    }
                    for worker_report in multi_report["worker_reports"]
                ],
                "aggregate": multi_report["aggregate"],
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with args.output_json.open("w", encoding="utf-8") as f:
                json.dump(output_report, f, ensure_ascii=True, indent=2)
        return

    debug_dir = (args.debug_dir or infer_default_debug_dir(sample_dir)).resolve()

    single_report = summarize_single_run(sample_dir, debug_dir)
    print_report(
        sample_dir,
        debug_dir,
        single_report["cif_summary"],
        single_report["debug_summary"],
        single_report["joined"],
    )

    if args.output_json is not None:
        report = {
            "mode": "single_run",
            "sample_dir": str(sample_dir),
            "debug_dir": str(debug_dir),
            "cif_summary": single_report["cif_summary"],
            "debug_summary": single_report["debug_summary"],
            "joined": {str(k): v for k, v in single_report["joined"].items()},
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
