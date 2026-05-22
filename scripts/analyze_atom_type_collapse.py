#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mp20.crystal import chemical_symbols


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


def class_index_to_symbol(class_idx):
    class_idx = int(class_idx)
    if 0 <= class_idx < len(chemical_symbols):
        return chemical_symbols[class_idx]
    return f"Z{class_idx}"


def stats_dict(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    array = [float(v) for v in values]
    array_sorted = sorted(array)
    return {
        "count": int(len(array)),
        "mean": float(sum(array) / len(array)),
        "median": float(statistics.median(array_sorted)),
        "min": float(array_sorted[0]),
        "max": float(array_sorted[-1]),
    }


def mean_pairwise_cosine_similarity(values):
    if values.dim() != 2 or values.size(0) < 2:
        return None
    values = values.float()
    if values.size(-1) > 1:
        values = values[:, 1:]
    if values.size(-1) == 0:
        return None
    normalized = torch.nn.functional.normalize(values, p=2, dim=-1, eps=1e-12)
    cosine_matrix = normalized @ normalized.transpose(0, 1)
    pair_mask = ~torch.eye(cosine_matrix.size(0), dtype=torch.bool, device=cosine_matrix.device)
    if not pair_mask.any():
        return None
    return float(cosine_matrix[pair_mask].mean().item())


def infer_debug_dir(sample_dir):
    direct = sample_dir / "atom_type_debug"
    if direct.exists():
        return direct
    if sample_dir.name.startswith("epoch_"):
        sibling = sample_dir.parent / "atom_type_debug"
        if sibling.exists():
            return sibling
    return direct


def infer_run_specs(path):
    path = path.resolve()
    worker_dirs = sorted(
        child for child in path.iterdir()
        if path.exists() and child.is_dir() and child.name.startswith("worker_")
    ) if path.exists() else []
    if worker_dirs:
        specs = []
        for worker_dir in worker_dirs:
            specs.append(
                {
                    "run_name": worker_dir.name,
                    "sample_dir": worker_dir / "epoch_0",
                    "debug_dir": worker_dir / "atom_type_debug",
                }
            )
        return specs

    if path.name.startswith("worker_") and (path / "epoch_0").exists():
        return [
            {
                "run_name": path.name,
                "sample_dir": path / "epoch_0",
                "debug_dir": path / "atom_type_debug",
            }
        ]

    if (path / "epoch_0").exists() and infer_debug_dir(path).exists():
        return [
            {
                "run_name": path.name,
                "sample_dir": path / "epoch_0",
                "debug_dir": infer_debug_dir(path),
            }
        ]

    return [
        {
            "run_name": path.parent.name if path.name.startswith("epoch_") else path.name,
            "sample_dir": path,
            "debug_dir": infer_debug_dir(path),
        }
    ]


def load_step_histories(debug_dir):
    tensor_dir = debug_dir / "atom_type_step_tensors"
    histories = defaultdict(dict)
    if not tensor_dir.exists():
        return histories

    for pt_path in sorted(tensor_dir.glob("round_*_step_*.pt")):
        payload = torch.load(pt_path, map_location="cpu")
        step_index = int(payload["step_index"])
        logits = payload["logits"]
        raw_probs = payload.get("raw_probs")
        decode_probs = payload.get("decode_probs")
        decoded_idx = payload.get("decoded_idx")
        node_mask = payload["node_mask"]
        for local_index, sample_index in enumerate(payload["sample_global_indices"]):
            histories[int(sample_index)][step_index] = {
                "step_index": step_index,
                "round_index": int(payload["round_index"]),
                "sample_global_index": int(sample_index),
                "sample_local_index": int(payload["sample_local_indices"][local_index]),
                "logits": logits[local_index],
                "raw_probs": raw_probs[local_index] if raw_probs is not None else None,
                "decode_probs": decode_probs[local_index] if decode_probs is not None else None,
                "decoded_idx": decoded_idx[local_index] if decoded_idx is not None else None,
                "node_mask": node_mask[local_index],
                "atom_type_state_input_available": bool(payload.get("atom_type_state_input_available", False)),
                "input_state_updated_step_index": payload.get("input_state_updated_step_index"),
                "empty_graph_fallback": bool(payload.get("empty_graph_fallback", False)),
                "fallback_source": payload.get("fallback_source"),
                "used_previous_atom_logits_fallback": bool(payload.get("used_previous_atom_logits_fallback", False)),
            }
    return histories


def sample_from_probs(probs, seed):
    if probs.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**31))
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


def all_h_from_decoded(decoded):
    return bool(decoded.numel() > 0 and decoded.eq(1).all().item())


def decode_with_temperature(logits, temperature):
    masked_logits = logits.clone()
    masked_logits[:, 0] = -1e9
    probs = torch.softmax(masked_logits / float(temperature), dim=-1)
    return probs


def analyze_sample_history(sample_key, step_records, final_sample_row):
    step_indices = sorted(step_records)
    per_step = []
    h_probs_all = []
    margins_all = []
    logits_diversity = []
    probs_diversity = []
    cosine_values = []
    argmax_all_h_by_step = []
    fallback_used = False
    previous_logits_fallback_used = False

    for step_index in step_indices:
        record = step_records[step_index]
        valid_mask = record["node_mask"].squeeze(-1).bool()
        logits = record["logits"][valid_mask]
        decode_probs = record["decode_probs"][valid_mask]
        decoded = record["decoded_idx"][valid_mask]
        if logits.numel() == 0:
            continue

        top2 = min(2, decode_probs.size(-1))
        top_probs, _ = torch.topk(decode_probs, k=top2, dim=-1)
        if top2 == 1:
            margin = top_probs[:, 0]
        else:
            margin = top_probs[:, 0] - top_probs[:, 1]

        h_prob = decode_probs[:, 1] if decode_probs.size(-1) > 1 else torch.zeros(decode_probs.size(0))
        logits_excluding_unknown = logits[:, 1:] if logits.size(-1) > 1 else logits
        probs_excluding_unknown = decode_probs[:, 1:] if decode_probs.size(-1) > 1 else decode_probs
        cross_node_logit_std_mean = float(
            logits_excluding_unknown.std(dim=0, unbiased=False).mean().item()
        ) if logits.size(0) > 1 and logits_excluding_unknown.numel() > 0 else 0.0
        cross_node_prob_std_mean = float(
            probs_excluding_unknown.std(dim=0, unbiased=False).mean().item()
        ) if decode_probs.size(0) > 1 and probs_excluding_unknown.numel() > 0 else 0.0
        cosine_mean = mean_pairwise_cosine_similarity(logits_excluding_unknown)
        argmax_all_h = all_h_from_decoded(decoded)

        per_step.append(
            {
                "step_index": int(step_index),
                "all_h_argmax": bool(argmax_all_h),
                "mean_h_probability": float(h_prob.mean().item()),
                "mean_top1_top2_margin": float(margin.mean().item()),
                "logits_std_dim0_mean": cross_node_logit_std_mean,
                "probs_std_dim0_mean": cross_node_prob_std_mean,
                "mean_cosine_similarity_between_node_logits": cosine_mean,
                "empty_graph_fallback": bool(record["empty_graph_fallback"]),
                "used_previous_atom_logits_fallback": bool(record["used_previous_atom_logits_fallback"]),
            }
        )
        h_probs_all.extend(float(v) for v in h_prob.tolist())
        margins_all.extend(float(v) for v in margin.tolist())
        logits_diversity.append(cross_node_logit_std_mean)
        probs_diversity.append(cross_node_prob_std_mean)
        if cosine_mean is not None:
            cosine_values.append(cosine_mean)
        argmax_all_h_by_step.append(bool(argmax_all_h))
        fallback_used = fallback_used or bool(record["empty_graph_fallback"])
        previous_logits_fallback_used = previous_logits_fallback_used or bool(record["used_previous_atom_logits_fallback"])

    final_record = step_records[step_indices[-1]]
    final_valid_mask = final_record["node_mask"].squeeze(-1).bool()
    final_logits = final_record["logits"][final_valid_mask]
    final_decode_probs = final_record["decode_probs"][final_valid_mask]
    avg_logits = torch.stack(
        [step_records[idx]["logits"][step_records[idx]["node_mask"].squeeze(-1).bool()] for idx in step_indices],
        dim=0,
    ).mean(dim=0)
    avg_probs = torch.stack(
        [step_records[idx]["decode_probs"][step_records[idx]["node_mask"].squeeze(-1).bool()] for idx in step_indices],
        dim=0,
    ).mean(dim=0)
    avg_probs = avg_probs / avg_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    sample_seed_base = abs(hash(sample_key)) % (2**31)
    alt_decodes = {
        "argmax_final": torch.argmax(final_decode_probs, dim=-1),
        "categorical_final_T1.0": sample_from_probs(
            decode_with_temperature(final_logits, 1.0),
            sample_seed_base + 11,
        ),
        "categorical_final_T0.7": sample_from_probs(
            decode_with_temperature(final_logits, 0.7),
            sample_seed_base + 17,
        ),
        "avg_logits_last10_argmax": torch.argmax(decode_with_temperature(avg_logits, 1.0), dim=-1),
        "avg_probs_last10_sampling": sample_from_probs(avg_probs, sample_seed_base + 23),
    }
    alt_summary = {
        name: {
            "all_h": all_h_from_decoded(decoded),
            "decoded_class_ids": [int(v) for v in decoded.tolist()],
            "decoded_species": [class_index_to_symbol(v) for v in decoded.tolist()],
        }
        for name, decoded in alt_decodes.items()
    }

    return {
        "sample_key": sample_key,
        "sample_global_index": int(final_sample_row["sample_global_index"]) if final_sample_row else None,
        "default_final_all_h": bool(final_sample_row.get("all_H", False)) if final_sample_row else False,
        "default_final_species_counts": final_sample_row.get("species_counts") if final_sample_row else None,
        "step_indices": step_indices,
        "per_step": per_step,
        "step_count": len(per_step),
        "h_probability_stats": stats_dict(h_probs_all),
        "top1_top2_margin_stats": stats_dict(margins_all),
        "logits_std_dim0_mean_stats": stats_dict(logits_diversity),
        "probs_std_dim0_mean_stats": stats_dict(probs_diversity),
        "mean_cosine_similarity_stats": stats_dict(cosine_values),
        "top1_stable_as_all_h_across_991_1000": bool(argmax_all_h_by_step and all(argmax_all_h_by_step)),
        "all_h_first_step_index": step_indices[next((idx for idx, flag in enumerate(argmax_all_h_by_step) if flag), -1)]
        if any(argmax_all_h_by_step) else None,
        "all_h_only_at_last_step": bool(
            argmax_all_h_by_step
            and argmax_all_h_by_step[-1]
            and not any(argmax_all_h_by_step[:-1])
        ),
        "fallback_used_in_final_window": bool(fallback_used),
        "previous_logits_fallback_used_in_final_window": bool(previous_logits_fallback_used),
        "alternative_decodes": alt_summary,
        "_raw_h_probability_values": h_probs_all,
        "_raw_margin_values": margins_all,
        "_raw_logits_std_dim0_values": logits_diversity,
        "_raw_probs_std_dim0_values": probs_diversity,
        "_raw_cosine_values": cosine_values,
    }


def summarize_group(sample_reports):
    h_prob_values = []
    margin_values = []
    logits_std_values = []
    probs_std_values = []
    cosine_values = []
    stable_h_flags = []
    last_step_only_flags = []
    fallback_flags = []
    previous_fallback_flags = []

    for report in sample_reports:
        h_prob_values.extend(report.get("_raw_h_probability_values", []))
        margin_values.extend(report.get("_raw_margin_values", []))
        logits_std_values.extend(report.get("_raw_logits_std_dim0_values", []))
        probs_std_values.extend(report.get("_raw_probs_std_dim0_values", []))
        cosine_values.extend(report.get("_raw_cosine_values", []))
        stable_h_flags.append(bool(report["top1_stable_as_all_h_across_991_1000"]))
        last_step_only_flags.append(bool(report["all_h_only_at_last_step"]))
        fallback_flags.append(bool(report["fallback_used_in_final_window"]))
        previous_fallback_flags.append(bool(report["previous_logits_fallback_used_in_final_window"]))

    return {
        "sample_count": len(sample_reports),
        "P_H_stats": stats_dict(h_prob_values),
        "top1_top2_margin_stats": stats_dict(margin_values),
        "logits_std_dim0_mean_stats": stats_dict(logits_std_values),
        "probs_std_dim0_mean_stats": stats_dict(probs_std_values),
        "cosine_similarity_stats": stats_dict(cosine_values),
        "top1_stable_as_all_h_rate": float(sum(stable_h_flags) / len(stable_h_flags)) if stable_h_flags else None,
        "all_h_only_at_last_step_rate": float(sum(last_step_only_flags) / len(last_step_only_flags)) if last_step_only_flags else None,
        "fallback_used_rate": float(sum(fallback_flags) / len(fallback_flags)) if fallback_flags else None,
        "previous_logits_fallback_used_rate": float(sum(previous_fallback_flags) / len(previous_fallback_flags)) if previous_fallback_flags else None,
    }


def analyze_run(run_name, sample_dir, debug_dir):
    final_rows = {
        int(row["sample_global_index"]): row
        for row in load_jsonl(debug_dir / "analyze_test_samples.jsonl")
        if row.get("event") == "sample_pre_cif_save" and row.get("sample_global_index") is not None
    }
    step_histories = load_step_histories(debug_dir)
    state_flow_rows = load_jsonl(debug_dir / "atom_type_state_flow.jsonl")

    sample_reports = []
    alt_all_h_counts = Counter()
    default_all_h_count = 0
    all_h_reports = []
    non_all_h_reports = []

    for sample_index in sorted(step_histories):
        sample_key = f"{run_name}:{sample_index}"
        final_row = final_rows.get(sample_index, {})
        report = analyze_sample_history(sample_key, step_histories[sample_index], final_row)
        sample_reports.append(report)
        if report["default_final_all_h"]:
            default_all_h_count += 1
            all_h_reports.append(report)
        else:
            non_all_h_reports.append(report)
        for alt_name, alt_info in report["alternative_decodes"].items():
            if alt_info["all_h"]:
                alt_all_h_counts[alt_name] += 1

    prediction_window_steps = [row for row in state_flow_rows if row.get("prediction_window_step")]
    prediction_window_start_step_index = prediction_window_steps[0]["prediction_window_start_step_index"] \
        if prediction_window_steps else None
    step_before_window_carried = any(
        row.get("step_index") == prediction_window_start_step_index
        and row.get("atom_type_state_input_available")
        and row.get("input_state_updated_step_index") is not None
        and int(row["input_state_updated_step_index"]) < int(prediction_window_start_step_index)
        for row in state_flow_rows
    ) if prediction_window_start_step_index is not None else None
    prediction_window_overwrites_state = all(
        row.get("state_overwritten_with_current_logits")
        for row in prediction_window_steps
    ) if prediction_window_steps else None
    fallback_only_when_empty_graph = all(
        (not row.get("used_previous_atom_logits_fallback")) or row.get("empty_graph_fallback")
        for row in state_flow_rows
    ) if state_flow_rows else None

    return {
        "run_name": run_name,
        "sample_dir": str(sample_dir),
        "debug_dir": str(debug_dir),
        "sample_count": len(sample_reports),
        "default_all_h_count": int(default_all_h_count),
        "alternative_decode_all_h_counts": dict(sorted(alt_all_h_counts.items())),
        "all_h_group": summarize_group(all_h_reports),
        "non_all_h_group": summarize_group(non_all_h_reports),
        "state_flow_checks": {
            "prediction_window_start_step_index": prediction_window_start_step_index,
            "step_before_window_carried_into_step_991": step_before_window_carried,
            "prediction_window_steps_overwrite_previous_state": prediction_window_overwrites_state,
            "fallback_only_when_empty_graph": fallback_only_when_empty_graph,
        },
        "sample_reports": sample_reports,
    }


def compute_training_label_stats(dataset_root):
    dataset_root = dataset_root.resolve()
    cached_path = dataset_root / "processed" / "all_ori.pt"
    if not cached_path.exists():
        raise FileNotFoundError(f"Could not find cached dataset at {cached_path}")

    cached_rows = torch.load(cached_path, map_location="cpu", weights_only=False)
    element_counts = Counter()
    total_structures = 0
    all_h_structures = 0
    single_element_structures = 0

    for row in cached_rows:
        atom_types = [int(v) for v in row["graph_arrays"]["atom_types"]]
        total_structures += 1
        element_counts.update(int(v) for v in atom_types)
        unique_types = set(atom_types)
        if atom_types and unique_types == {1}:
            all_h_structures += 1
        if len(unique_types) == 1:
            single_element_structures += 1

    total_atoms = sum(element_counts.values())
    top20 = []
    for class_idx, count in element_counts.most_common(20):
        top20.append(
            {
                "class_idx": int(class_idx),
                "symbol": class_index_to_symbol(class_idx),
                "count": int(count),
                "frequency": float(count / total_atoms) if total_atoms > 0 else None,
            }
        )

    return {
        "dataset_root": str(dataset_root),
        "total_structures": int(total_structures),
        "total_atoms": int(total_atoms),
        "class0_count": int(element_counts.get(0, 0)),
        "class0_frequency": float(element_counts.get(0, 0) / total_atoms) if total_atoms > 0 else None,
        "H_count": int(element_counts.get(1, 0)),
        "H_frequency": float(element_counts.get(1, 0) / total_atoms) if total_atoms > 0 else None,
        "top20_element_frequencies": top20,
        "all_H_structures": int(all_h_structures),
        "all_H_structure_rate": float(all_h_structures / total_structures) if total_structures > 0 else None,
        "single_element_structures": int(single_element_structures),
        "single_element_structure_rate": float(single_element_structures / total_structures) if total_structures > 0 else None,
    }


def print_run_summary(run_report):
    print(f"Run: {run_report['run_name']}")
    print(f"  Sample count: {run_report['sample_count']}")
    print(f"  Default all-H count: {run_report['default_all_h_count']}")
    print(f"  Alternative decode all-H counts: {run_report['alternative_decode_all_h_counts']}")
    print(f"  State flow checks: {run_report['state_flow_checks']}")
    print(f"  All-H group: {run_report['all_h_group']}")
    print(f"  Non-all-H group: {run_report['non_all_h_group']}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze final-window atom-type predictor collapse using debug traces from LF_wrap sampling."
    )
    parser.add_argument(
        "sample_path",
        type=Path,
        help="Sampling save dir, worker dir, or epoch_0 dir.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional dataset root (e.g. ./mp20) for training-label statistics.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full report as JSON.",
    )
    args = parser.parse_args()

    run_specs = infer_run_specs(args.sample_path)
    run_reports = []
    for spec in run_specs:
        run_report = analyze_run(
            run_name=spec["run_name"],
            sample_dir=spec["sample_dir"],
            debug_dir=spec["debug_dir"],
        )
        run_reports.append(run_report)
        print_run_summary(run_report)

    training_stats = None
    if args.dataset_root is not None:
        training_stats = compute_training_label_stats(args.dataset_root)
        print("Training label stats:")
        print(training_stats)

    full_report = {
        "sample_path": str(args.sample_path.resolve()),
        "run_reports": run_reports,
        "training_label_stats": training_stats,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(full_report, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
