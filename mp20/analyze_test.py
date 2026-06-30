# 确保这是文件的第一行
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF")

from mp20.sample_epoch import sample, sample_pure_x, sample_L, sample_withL, sample_F
from mp20.crystal import lattice_matrix, cart_to_frac, frac_to_cart, array_dict_to_crystal, chemical_symbols
from mp20.utils import RankedLogger, joblib_map, prepare_context_test, compute_loss_and_nll,\
    assert_correctly_masked, remove_mean_with_mask, assert_mean_zero_with_mask, check_mask_correct,\
    compute_loss_and_nll_pure_x, compute_loss_and_nll_L
from mp20.ase_tools.viewer import AseView
from mp20.batch_reshape import reshape
from mp20.geometry_diagnostics import diagnose_geometry_records, save_raw_geometry_npz

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from typing import Dict
from functools import partial   # 固定某个函数的一部分参数，返回一个新的函数
from mp20.novelty import (
    DEFAULT_SKIP_STRUCTURE_REDUCTION,
    build_structure_matcher,
    is_structure_novel,
)

import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import json
import torch
import wandb
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from pymatgen.core import Lattice, Structure


log = RankedLogger(__name__)  # 代码输出日志
ase_view = AseView(
    rotations="45x,45y,45z",
    atom_font_size=16,
    axes_length=30,
    canvas_size=(400, 400),
    zoom=1.2,
    show_bonds=False,
    # uc_dash_pattern=(.6, .4),
    atom_show_label=True,
    canvas_background_opacity=0.0,
)


def _atom_type_debug_enabled(args):
    return bool(getattr(args, "debug_atom_types", False)) or os.environ.get("DEBUG_ATOM_TYPES", "0") == "1"


def _all_h_guard_enabled(args):
    if getattr(args, "atom_decode_mode", "constrained_search") == "raw_argmax":
        return False
    if hasattr(args, "disable_all_h_guard"):
        return not bool(getattr(args, "disable_all_h_guard", False))
    env_value = os.environ.get("MCW_ALL_H_GUARD_ENABLED")
    return env_value == "1"


def _all_h_guard_fail_fast(args):
    if getattr(args, "atom_decode_mode", "constrained_search") == "raw_argmax":
        return False
    return bool(_all_h_guard_enabled(args) or _atom_type_debug_enabled(args))


def _set_all_h_guard_env(args):
    enabled = _all_h_guard_enabled(args)
    disable_arg = bool(getattr(args, "disable_all_h_guard", False))
    atom_decode_mode = getattr(args, "atom_decode_mode", "constrained_search")
    os.environ["MCW_ALL_H_GUARD_ENABLED"] = "1" if enabled else "0"
    os.environ["MCW_ALL_H_GUARD_DISABLED_ARG"] = "1" if disable_arg else "0"
    os.environ["MCW_ATOM_DECODE_MODE"] = atom_decode_mode
    print("analyze_test disable_all_h_guard arg:", disable_arg)
    print("analyze_test all_h_guard_enabled:", enabled)
    print("analyze_test atom_decode_mode:", atom_decode_mode)
    print("analyze_test geometry_correction:", getattr(args, "geometry_correction", True))
    _write_atom_type_debug_line(
        args,
        "atom_type_session.jsonl",
        {
            "event": "analyze_test_guard_config",
            "disable_all_h_guard_arg": disable_arg,
            "all_h_guard_enabled": enabled,
            "atom_decode_mode": atom_decode_mode,
            "geometry_correction": bool(getattr(args, "geometry_correction", True)),
            "all_h_guard_fail_fast_enabled": bool(_all_h_guard_fail_fast(args)),
        },
    )
    return enabled


def _atom_type_debug_dir(args):
    debug_dir = getattr(args, "debug_atom_dir", "") or os.environ.get("DEBUG_ATOM_TYPES_DIR", "")
    if not debug_dir:
        debug_dir = os.path.join(getattr(args, "save_dir", "mp20/analyze_test/"), "atom_type_debug")
    return debug_dir


def _write_atom_type_debug_line(args, filename, payload):
    if not _atom_type_debug_enabled(args):
        return
    debug_dir = _atom_type_debug_dir(args)
    os.makedirs(debug_dir, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("pid", os.getpid())
    with open(os.path.join(debug_dir, filename), "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _atom_type_symbols(atom_types):
    symbols = []
    for atom_type in atom_types:
        atom_type = int(atom_type)
        if 0 <= atom_type < len(chemical_symbols):
            symbols.append(chemical_symbols[atom_type])
        else:
            symbols.append(f"Z{atom_type}")
    return symbols


def _guard_atom_types_or_raise(args, atom_types, species_symbols, sample_global_index, stage):
    atom_types = np.array(atom_types, dtype=int)
    is_all_h = bool(atom_types.size > 0 and np.all(atom_types == 1))
    if is_all_h and _all_h_guard_fail_fast(args):
        payload = {
            "event": "all_h_guard_violation",
            "stage": stage,
            "sample_global_index": int(sample_global_index),
            "atom_types": [int(v) for v in atom_types.tolist()],
            "species_symbols": list(species_symbols),
            "all_h_guard_enabled": bool(_all_h_guard_enabled(args)),
            "disable_all_h_guard_arg": bool(getattr(args, "disable_all_h_guard", False)),
        }
        _write_atom_type_debug_line(args, "atom_type_guard_failures.jsonl", payload)
        raise RuntimeError(
            f"[AllHGuard] {stage} observed all-H sample at global sample index "
            f"{int(sample_global_index)} with species {list(species_symbols)}"
        )
    return is_all_h


def _to_float(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return [float(v) for v in value.reshape(-1).tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return float(value)


def _build_sampling_metrics_payload(epoch, metrics_dict):
    payload = {
        "epoch": int(epoch),
    }
    for key in ("valid_rate", "comp_valid_rate", "struct_valid_rate"):
        metric = metrics_dict.get(key)
        if metric is None:
            continue
        metric_cpu = metric.detach().cpu()
        total = int(metric_cpu.numel())
        count = int(metric_cpu.sum().item())
        rate = float(metric_cpu.float().mean().item()) if total > 0 else None
        payload[key] = {
            "count": count,
            "total": total,
            "rate": rate,
        }

    payload["valid_count"] = payload.get("valid_rate", {}).get("count", 0)
    payload["total_samples"] = payload.get("valid_rate", {}).get("total", 0)
    payload["valid_rate_mean"] = payload.get("valid_rate", {}).get("rate")
    payload["comp_valid_count"] = payload.get("comp_valid_rate", {}).get("count", 0)
    payload["comp_valid_rate_mean"] = payload.get("comp_valid_rate", {}).get("rate")
    payload["struct_valid_count"] = payload.get("struct_valid_rate", {}).get("count", 0)
    payload["struct_valid_rate_mean"] = payload.get("struct_valid_rate", {}).get("rate")

    for key in ("unique_rate", "novel_rate"):
        if key in metrics_dict:
            payload[key] = _to_float(metrics_dict[key])
    return payload


def _write_sampling_metrics(args, epoch, metrics_dict):
    metrics_payload = _build_sampling_metrics_payload(epoch, metrics_dict)
    epoch_dir = os.path.join(args.save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    metrics_path = os.path.join(epoch_dir, "sampling_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=True, indent=2)
    _write_atom_type_debug_line(
        args,
        "sampling_metrics_written.jsonl",
        {
            "event": "sampling_metrics_written",
            "epoch": int(epoch),
            "metrics_path": metrics_path,
            "metrics": metrics_payload,
        },
    )
    return metrics_payload


def _jsonify(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _jsonify(value.detach().cpu().item())
        return [_jsonify(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.reshape(-1).tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _rate(count, total):
    return float(count / total) if total else 0.0


def _metric_count(metrics_dict, key):
    metric = metrics_dict.get(key)
    if metric is None:
        return 0, 0, None
    metric_cpu = metric.detach().cpu().reshape(-1)
    total = int(metric_cpu.numel())
    count = int(metric_cpu.sum().item())
    rate = float(metric_cpu.float().mean().item()) if total > 0 else None
    return count, total, rate


def _unwrap_model(model):
    return getattr(model, "module", model)


def _git_commit_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _run_config_payload(args, epoch):
    num_samples = int(getattr(args, "num_rounds", 1) * getattr(args, "sample_batch_size", 0))
    atom_decode_mode = getattr(args, "atom_decode_mode", "constrained_search")
    return {
        "config_name": getattr(args, "component_config_name", "") or Path(args.save_dir).name,
        "checkpoint_path": getattr(args, "pretrained_model", ""),
        "lattice_checkpoint_path": getattr(args, "pretrained_Lattice_model", ""),
        "sampling_config_path": getattr(args, "sampling_config_path", ""),
        "random_seed": int(getattr(args, "sample_seed", getattr(args, "seed", 0))),
        "num_samples": num_samples,
        "num_rounds": int(getattr(args, "num_rounds", 1)),
        "batch_size": int(getattr(args, "sample_batch_size", 0)),
        "geometry_correction": bool(getattr(args, "geometry_correction", True)),
        "atom_decode_mode": atom_decode_mode,
        "final_window_threshold_tau": int(getattr(args, "prediction_threshold_t", 10)),
        "top_k": int(getattr(args, "atom_type_repair_topk", 4)),
        "all_h_guard_top_k": int(getattr(args, "all_h_guard_topk", 4)),
        "max_high_entropy_replacement_sites": int(getattr(args, "atom_type_max_replace_atoms", 2)),
        "all_H_guard_enabled": bool(_all_h_guard_enabled(args)),
        "emergency_repair_enabled": bool(os.environ.get("MCW_ENABLE_EMERGENCY_ALL_H_REPAIR", "0") == "1"
                                          and atom_decode_mode == "constrained_search"),
        "unknown_class_masked": True,
        "epoch": int(epoch),
        "save_dir": getattr(args, "save_dir", ""),
        "date_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit_hash": _git_commit_hash(),
        "command_line": " ".join(sys.argv),
    }


def _write_run_config(args, epoch):
    payload = _run_config_payload(args, epoch)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for path in (save_dir / "run_config.json", save_dir / f"epoch_{epoch}" / "run_config.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonify(payload), handle, ensure_ascii=True, indent=2, allow_nan=False)
    return payload


def _evaluate_structural_validity_from_records(records):
    if not records:
        return {"count": None, "total": 0, "rate": None}
    flags = []
    for record in records:
        try:
            lengths = np.asarray(record.get("lengths"), dtype=float)
            angles = np.asarray(record.get("angles"), dtype=float)
            if lengths.size != 3 or angles.size != 3:
                from mp20.geometry_diagnostics import lattice_lengths_angles
                lengths, angles = lattice_lengths_angles(np.asarray(record["lattice"], dtype=float))
            crystal = array_dict_to_crystal(
                {
                    "frac_coords": np.asarray(record["frac_coords"], dtype=float),
                    "atom_types": np.asarray(record["atom_types"], dtype=int),
                    "lengths": lengths,
                    "angles": angles,
                    "sample_idx": record.get("sample_id", ""),
                },
                save=False,
                save_dir_name="",
            )
            flags.append(bool(crystal.struct_valid))
        except Exception:
            flags.append(False)
    count = int(sum(flags))
    total = int(len(flags))
    return {"count": count, "total": total, "rate": _rate(count, total)}


def _count_generated_cifs(save_dir, epoch):
    epoch_dir = Path(save_dir) / f"epoch_{epoch}"
    if not epoch_dir.exists():
        return 0
    return sum(1 for _ in epoch_dir.rglob("*.cif"))


def _diagnostic_counts_from_predictions(pred_arrays, pred_crys):
    all_h_count = 0
    single_element_count = 0
    close_contact_fail_count = 0
    invalid_lattice_count = 0
    invalid_lattice_reasons = {
        "non_positive_lattice",
        "nan_value",
        "unrealistically_small_lattice",
        "construction_raises_exception",
    }
    for pred, crystal in zip(pred_arrays, pred_crys):
        atom_types = np.asarray(pred.get("atom_types", []), dtype=int).reshape(-1)
        if atom_types.size > 0:
            all_h_count += int(np.all(atom_types == 1))
            single_element_count += int(len(set(atom_types.tolist())) == 1)
        invalid_reason = getattr(crystal, "invalid_reason", "")
        if invalid_reason == "constructed but structure invalid":
            close_contact_fail_count += 1
        if invalid_reason in invalid_lattice_reasons:
            invalid_lattice_count += 1
    return {
        "all_H_count": int(all_h_count),
        "single_element_count": int(single_element_count),
        "close_contact_fail_count": int(close_contact_fail_count),
        "invalid_lattice_count": int(invalid_lattice_count),
    }


def _guard_summary_from_model(model_sample):
    model = _unwrap_model(model_sample)
    summary = getattr(model, "_atom_type_all_h_guard_summary", {}) or {}
    return {
        "search_failure_count": int(summary.get("search_failed_count", 0)),
        "search_guard_call_count": int(summary.get("search_guard_call_count", 0)),
        "raw_all_H_count_from_logits": int(summary.get("raw_all_H_count", 0)),
        "final_all_H_count_from_logits": int(summary.get("final_all_H_count", 0)),
        "final_composition_valid_count_from_decode": int(summary.get("final_composition_valid_count", 0)),
    }


def _build_component_metrics_payload(
    args,
    epoch,
    metrics_dict,
    evaluator,
    model_sample,
    pre_correction_structural,
    geometry_summary,
    run_config,
):
    num_requested = int(getattr(args, "num_rounds", 1) * getattr(args, "sample_batch_size", 0))
    num_generated = int(len(evaluator.pred_arrays_list))
    valid_count, _, valid_rate = _metric_count(metrics_dict, "valid_rate")
    comp_valid_count, _, comp_valid_rate = _metric_count(metrics_dict, "comp_valid_rate")
    struct_valid_count, _, struct_valid_rate = _metric_count(metrics_dict, "struct_valid_rate")
    diagnostics = _diagnostic_counts_from_predictions(evaluator.pred_arrays_list, evaluator.pred_crys_list)
    guard_summary = _guard_summary_from_model(model_sample)

    valid_structs = getattr(evaluator, "_last_valid_structs", None)
    if valid_structs is None:
        valid_structs = [c.structure for c in evaluator.pred_crys_list if c.valid]
    unique_groups = getattr(evaluator, "_last_unique_struct_groups", None)
    if unique_groups is None:
        unique_groups = evaluator.matcher.group_structures(valid_structs) if valid_structs else []
    unique_count = int(len(unique_groups))
    novel_rate_value = _to_float(metrics_dict.get("novel_rate", torch.tensor(-1.0)))
    unique_group_is_novel = getattr(evaluator, "_last_unique_group_is_novel", None)
    if unique_group_is_novel is not None:
        novel_count = int(sum(bool(v) for v in unique_group_is_novel))
        unique_and_novel_count = novel_count
        un_rate = _rate(unique_and_novel_count, len(valid_structs))
    elif novel_rate_value is not None and novel_rate_value >= 0:
        novel_count = int(round(unique_count * novel_rate_value))
        unique_and_novel_count = novel_count
        un_rate = _rate(unique_and_novel_count, len(valid_structs))
    else:
        novel_count = None
        unique_and_novel_count = None
        un_rate = None

    atom_decode_mode = getattr(args, "atom_decode_mode", "constrained_search")
    constrained_mode = atom_decode_mode == "constrained_search"
    search_failure_count = guard_summary["search_failure_count"] if constrained_mode else None
    search_failure_rate = (
        _rate(search_failure_count, num_generated)
        if constrained_mode and search_failure_count is not None else None
    )

    payload = {
        "config_name": run_config["config_name"],
        "num_requested": num_requested,
        "num_generated": num_generated,
        "num_finalized_cifs": _count_generated_cifs(args.save_dir, epoch),
        "structural_valid_count": int(struct_valid_count),
        "structural_valid_rate": struct_valid_rate,
        "composition_valid_count": int(comp_valid_count),
        "composition_valid_rate": comp_valid_rate,
        "total_valid_count": int(valid_count),
        "total_valid_rate": valid_rate,
        "unique_count": unique_count,
        "novel_count": novel_count,
        "unique_and_novel_count": unique_and_novel_count,
        "UN_rate": un_rate,
        "all_H_count": diagnostics["all_H_count"],
        "all_H_rate": _rate(diagnostics["all_H_count"], num_generated),
        "single_element_count": diagnostics["single_element_count"],
        "single_element_rate": _rate(diagnostics["single_element_count"], num_generated),
        "close_contact_fail_count": diagnostics["close_contact_fail_count"],
        "close_contact_fail_rate": _rate(diagnostics["close_contact_fail_count"], num_generated),
        "invalid_lattice_count": diagnostics["invalid_lattice_count"],
        "invalid_lattice_rate": _rate(diagnostics["invalid_lattice_count"], num_generated),
        "search_failure_count": search_failure_count,
        "search_failure_rate": search_failure_rate,
        "raw_argmax_mode": bool(atom_decode_mode == "raw_argmax"),
        "constrained_search_mode": bool(constrained_mode),
        "geometry_correction": bool(getattr(args, "geometry_correction", True)),
        "atom_decode_mode": atom_decode_mode,
        "pre_correction_structural_valid_count": pre_correction_structural["count"],
        "pre_correction_structural_valid_rate": pre_correction_structural["rate"],
        "post_correction_structural_valid_count": int(struct_valid_count),
        "post_correction_structural_valid_rate": struct_valid_rate,
        "unknown_class_masked": True,
        "guard_summary": guard_summary,
        "geometry_pre_correction_summary": geometry_summary or None,
        "run_config": run_config,
    }
    return payload


def _write_component_metrics(args, epoch, payload):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    json_paths = [save_dir / "metrics.json", save_dir / f"epoch_{epoch}" / "metrics.json"]
    for path in json_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonify(payload), handle, ensure_ascii=True, indent=2, allow_nan=False)

    flat_fields = [
        "config_name", "num_requested", "num_generated", "num_finalized_cifs",
        "structural_valid_count", "structural_valid_rate",
        "composition_valid_count", "composition_valid_rate",
        "total_valid_count", "total_valid_rate",
        "unique_count", "novel_count", "unique_and_novel_count", "UN_rate",
        "all_H_count", "all_H_rate", "single_element_count", "single_element_rate",
        "close_contact_fail_count", "close_contact_fail_rate",
        "invalid_lattice_count", "invalid_lattice_rate",
        "search_failure_count", "search_failure_rate",
        "raw_argmax_mode", "constrained_search_mode",
        "geometry_correction", "atom_decode_mode",
        "pre_correction_structural_valid_count", "pre_correction_structural_valid_rate",
        "post_correction_structural_valid_count", "post_correction_structural_valid_rate",
    ]
    row = {field: _jsonify(payload.get(field)) for field in flat_fields}
    for filename in ("summary.csv", "metrics.csv"):
        path = save_dir / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=flat_fields)
            writer.writeheader()
            writer.writerow(row)
    print(f"[ComponentDiagnostics] wrote metrics.json and summary.csv under {save_dir}")


def analyze_and_save(args, epoch, model_sample, nodes_dist, dataset_info, 
                     prop_dist, evaluate_condition_generation):
    print(f'Analyzing crystal validity at epoch {epoch}...')
    _set_all_h_guard_env(args)
    batch_size = args.sample_batch_size
    device = args.device
    mp20_evaluator = CrystalGenerationEvaluator(
            dataset_cif_list=pd.read_csv(
                os.path.join(args.dataset_folder_path, f"all.csv")
            )["cif"].tolist(),
            compute_novelty=args.compute_novelty \
                if epoch >= args.compute_novelty_epoch else False
        )

    # sample the crystal structures
    nodesxsample = nodes_dist.sample(batch_size)
    if args.property_pred:
        if args.frac_coords_mode:
            one_hot, charges, frac_coords, node_mask, pred, length, angle = sample(args, device, model_sample, prop_dist=prop_dist,
                                            nodesxsample=nodesxsample, dataset_info=dataset_info)
        else:
            one_hot, charges, x, node_mask, pred, length, angle = sample(args, device, model_sample, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample, dataset_info=dataset_info)
    else:
        if args.frac_coords_mode:
            one_hot, charges, frac_coords, node_mask, length, angle = sample(args, device, model_sample, prop_dist=prop_dist,
                                            nodesxsample=nodesxsample, dataset_info=dataset_info)
        else:
            one_hot, charges, x, node_mask, length, angle = sample(args, device, model_sample, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample, dataset_info=dataset_info)
    length = length.detach().cpu().numpy()
    angle = angle.detach().cpu().numpy() 

    for i in range(int(batch_size)):
        
        lattice = lattice_matrix(length[i, 0], length[i, 1], length[i, 2],
                                    angle[i, 0], angle[i, 1], angle[i, 2])
        mask = node_mask[i].squeeze(-1).bool()

        if args.frac_coords_mode:
            frac_coords_valid = frac_coords[i][mask].detach().cpu().numpy()
            x_valid = frac_to_cart(frac_coords_valid, lattice)
        else: 
            x_valid = x[i][mask].detach().cpu().numpy()
            frac_coords_valid = cart_to_frac(x_valid, lattice)

        one_hot_valid = one_hot[i][mask].detach().cpu().numpy()
        atom_types = np.argmax(one_hot_valid, axis=-1)  # convert one-hot to atom types
        species_symbols = _atom_type_symbols(atom_types.tolist())
        _guard_atom_types_or_raise(
            args,
            atom_types,
            species_symbols,
            sample_global_index=i,
            stage="analyze_and_save_pre_append",
        )
        # charges = charges[i][mask].detach().cpu().numpy()

        if i <= 3:
            # print("sampled frac_coords:", frac_coords_valid)
            print("sampled x", x_valid)
            print("sampled lengths:", length[i])
            print("sampled angles:", angle[i])
            # print("sampled atom types:", atom_types)

        mp20_evaluator.append_pred_array(
                {
                    "atom_types": atom_types,
                    "pos": x_valid,
                    "frac_coords": frac_coords_valid,
                    "lengths": length[i],
                    "angles": angle[i],
                    "sample_idx": f"epoch_{epoch}_sample_{i}"
                }
            )

    # Compute generation metrics
    metrics_dict = mp20_evaluator.get_metrics(
        save=args.visualize,
        save_dir=args.save_dir + f"/epoch_{epoch}",
    )   # warning!


    for k, v in metrics_dict.items():
        print(f"{k}: {v.tolist() if isinstance(v, torch.Tensor) else v}")

    wandb.log(metrics_dict)
    wandb.log({'Validity': metrics_dict["valid_rate"].sum()/batch_size, 
               'Uniqueness': metrics_dict["unique_rate"], 
               'Novelty': metrics_dict["novel_rate"]})
    
    print({'Validity': metrics_dict["valid_rate"].sum()/batch_size, 
               'Uniqueness': metrics_dict["unique_rate"], 
               'Novelty': metrics_dict["novel_rate"]})
    _write_sampling_metrics(args, epoch, metrics_dict)

    return metrics_dict


def analyze_and_save_withL(args, epoch, model_sample, LatticeGenModel, nodes_dist, dataset_info, 
                     prop_dist, evaluate_condition_generation):
    print(f'Analyzing crystal validity at epoch {epoch}...')
    _set_all_h_guard_env(args)
    batch_size = args.sample_batch_size
    device = args.device
    mp20_evaluator = CrystalGenerationEvaluator(
            dataset_cif_list=pd.read_csv(
                os.path.join(args.dataset_folder_path, f"all.csv")
            )["cif"].tolist(),
            compute_novelty=args.compute_novelty \
                if epoch >= args.compute_novelty_epoch else False
        )

    # sample the crystal structures
    nodesxsample = nodes_dist.sample(batch_size)
    one_hot, charges, x, node_mask, length, angle = sample_withL(args, device, model_sample, LatticeGenModel, 
                                                                 prop_dist=prop_dist, nodesxsample=nodesxsample, 
                                                                 dataset_info=dataset_info)
    length = length.detach().cpu().numpy()
    angle = angle.detach().cpu().numpy() 

    for i in range(int(batch_size)):
        lattice = lattice_matrix(length[i, 0], length[i, 1], length[i, 2],
                                    angle[i, 0], angle[i, 1], angle[i, 2])
        # 实际上是一般文献的Lattice的转置
        mask = node_mask[i].squeeze(-1).bool()
        x_valid = x[i][mask].detach().cpu().numpy()
        frac_coords_valid = cart_to_frac(x_valid, lattice) % 1.0  # within [0, 1)
        one_hot_valid = one_hot[i][mask].detach().cpu().numpy()
        atom_types = np.argmax(one_hot_valid, axis=-1)
        species_symbols = _atom_type_symbols(atom_types.tolist())
        _guard_atom_types_or_raise(
            args,
            atom_types,
            species_symbols,
            sample_global_index=i,
            stage="analyze_and_save_withL_pre_append",
        )
        # charges = charges[i][mask].detach().cpu().numpy()
        
        if i <= 5:
            print("sampled x", x_valid)
            print("sampled frac_coords:", frac_coords_valid)
            print("sampled lengths:", length[i])
            print("sampled angles:", angle[i])
            # print("sampled atom types:", atom_types)
        mp20_evaluator.append_pred_array(
                {
                    "atom_types": atom_types,
                    "pos": x_valid,
                    "frac_coords": frac_coords_valid,
                    "lengths": length[i],
                    "angles": angle[i],
                    "sample_idx": f"epoch_{epoch}_sample_{i}"
                }
            )

    # Compute generation metrics
    metrics_dict = mp20_evaluator.get_metrics(
        save=args.visualize,
        save_dir=args.save_dir + f"/epoch_{epoch}",
    )   # warning!

    for k, v in metrics_dict.items():
        print(f"{k}: {v.tolist() if isinstance(v, torch.Tensor) else v}")
    wandb.log(metrics_dict)
    wandb.log({'Validity': metrics_dict["valid_rate"].sum()/batch_size, 
               'Uniqueness': metrics_dict["unique_rate"], 
               'Novelty': metrics_dict["novel_rate"]})
    print({ 'Validity': metrics_dict["valid_rate"].sum()/batch_size, 
            'Uniqueness': metrics_dict["unique_rate"], 
            'Novelty': metrics_dict["novel_rate"]})
    _write_sampling_metrics(args, epoch, metrics_dict)

    return metrics_dict


def analyze_and_save_F(args, epoch, model_sample, LatticeGenModel, nodes_dist, dataset_info, 
                     prop_dist, evaluate_condition_generation, dataloader=None):
    print(f'Analyzing crystal validity at epoch {epoch}...')
    _set_all_h_guard_env(args)
    run_config = _write_run_config(args, epoch)
    batch_size = args.sample_batch_size
    device = args.device
    dtype = args.dtype
    if _atom_type_debug_enabled(args):
        _write_atom_type_debug_line(
            args,
            "analyze_test_samples.jsonl",
            {
                "event": "analyze_and_save_F_start",
                "epoch": int(epoch),
                "batch_size": int(batch_size),
                "num_rounds": int(args.num_rounds),
                "disable_all_h_guard_arg": bool(getattr(args, "disable_all_h_guard", False)),
                "all_h_guard_enabled": bool(_all_h_guard_enabled(args)),
                "atom_decoder": dataset_info.get("atom_decoder"),
                "atom_decoder_index0_is_H": bool(dataset_info.get("atom_decoder", [None])[0] == "H"),
            },
        )
    mp20_evaluator = CrystalGenerationEvaluator(
            dataset_cif_list=pd.read_csv(
                os.path.join(args.dataset_folder_path, f"all.csv")
            )["cif"].tolist(),
            compute_novelty=args.compute_novelty \
                if epoch >= args.compute_novelty_epoch else False
        )

    # sample the crystal structures
    nodesxsample = nodes_dist.sample(batch_size)
    collect_pre_correction = bool(
        getattr(args, "diagnose_geometry_before_correction", False)
        or getattr(args, "save_pre_correction_geometry_npz", False)
        or getattr(args, "geometry_diagnostics_every_batch", False)
        or not getattr(args, "geometry_correction", True)
    )
    if args.sample_realistic_LA:
        first_batch = next(iter(dataloader))
        first_batch = reshape(first_batch, device, dtype, include_charges=True)
        rl, ra = first_batch['lengths'], first_batch['angles']
        # test sampling with given L
        sample_result = sample_F(args, device, model_sample, LatticeGenModel,
                                 prop_dist=prop_dist, nodesxsample=nodesxsample,
                                 dataset_info=dataset_info, rl=rl, ra=ra)
    else:
        sample_result = sample_F(args, device, model_sample, LatticeGenModel,
                                 prop_dist=prop_dist, nodesxsample=nodesxsample,
                                 dataset_info=dataset_info)
    if collect_pre_correction:
        one_hot, charges, frac_pos, node_mask, length, angle, pre_correction_frac_pos = sample_result
    else:
        one_hot, charges, frac_pos, node_mask, length, angle = sample_result
        pre_correction_frac_pos = None
    
    length = length.detach().cpu().numpy()
    angle = angle.detach().cpu().numpy() 

    num = int(args.num_rounds * batch_size)

    pre_correction_records = []
    for i in range(num):
        lattice = lattice_matrix(length[i, 0], length[i, 1], length[i, 2],
                                    angle[i, 0], angle[i, 1], angle[i, 2])
        # 实际上是一般文献的Lattice的转置
        mask = node_mask[i].squeeze(-1).bool()
        real_atom_count = int(mask.sum().item())
        frac_pos_valid = frac_pos[i][mask].detach().cpu().numpy()
        cart_pos_valid = frac_to_cart(frac_pos_valid, lattice)
        one_hot_valid = one_hot[i][mask].detach().cpu().numpy()
        assert one_hot_valid.ndim == 2, f"decoded one_hot must be [N, C], got {one_hot_valid.shape}"
        atom_types = np.argmax(one_hot_valid, axis=-1)
        species_symbols = _atom_type_symbols(atom_types.tolist())
        unique_atom_types, counts = np.unique(atom_types, return_counts=True) if atom_types.size > 0 else (np.array([]), np.array([]))
        species_counts = {
            chemical_symbols[int(atom_type)] if 0 <= int(atom_type) < len(chemical_symbols) else f"Z{int(atom_type)}": int(count)
            for atom_type, count in zip(unique_atom_types.tolist(), counts.tolist())
        }
        all_h = bool(atom_types.size > 0 and np.all(atom_types == 1))
        _guard_atom_types_or_raise(
            args,
            atom_types,
            species_symbols,
            sample_global_index=i,
            stage="analyze_and_save_F_pre_append",
        )
        if all_h:
            print(f"[AtomTypeDebug] analyze_and_save_F observed all-H sample at global sample index {i}")
        _write_atom_type_debug_line(
            args,
            "analyze_test_samples.jsonl",
            {
                "event": "sample_pre_cif_save",
                "epoch": int(epoch),
                "sample_global_index": int(i),
                "real_atom_count": real_atom_count,
                "atom_types": [int(v) for v in atom_types.tolist()],
                "species_symbols": species_symbols,
                "species_counts": species_counts,
                "unique_species_count": int(len(unique_atom_types)),
                "all_H": all_h,
                "lengths": [float(v) for v in length[i].tolist()],
                "angles": [float(v) for v in angle[i].tolist()],
                "padding_exists": bool(mask.numel() > real_atom_count),
            },
        )
        if collect_pre_correction:
            pre_correction_records.append(
                {
                    "sample_id": f"epoch_{epoch}_sample_{i}",
                    "source": "main_LF_sample:before_first_geometry_correction",
                    # mp20.crystal.lattice_matrix is row-wise. Diagnostics use
                    # the LF model's column-wise convention: cart=frac@L.T.
                    "lattice": lattice.T,
                    "lengths": length[i],
                    "angles": angle[i],
                    "frac_coords": pre_correction_frac_pos[i][mask].detach().cpu().numpy(),
                    "num_atoms": real_atom_count,
                    "atom_types": atom_types,
                }
            )
        # charges = charges[i][mask].detach().cpu().numpy()
        
        if i <= 5:
            print("[PostCorrectionGeometry] sampled frac_pos:", frac_pos_valid)
            print("[PostCorrectionGeometry] sampled cart_pos:", cart_pos_valid)
            print("[PostCorrectionGeometry] sampled lengths:", length[i])
            print("[PostCorrectionGeometry] sampled angles:", angle[i])
            # print("sampled atom types:", atom_types)
        mp20_evaluator.append_pred_array(
                {
                    "atom_types": atom_types,
                    "pos": cart_pos_valid,
                    "frac_coords": frac_pos_valid,
                    "lengths": length[i],
                    "angles": angle[i],
                    "sample_idx": f"epoch_{epoch}_sample_{i}"
                }
            )

    diagnostics_output_dir = getattr(args, "geometry_diagnostics_output_dir", "")
    if not diagnostics_output_dir:
        diagnostics_output_dir = os.path.join(
            args.save_dir, f"epoch_{epoch}", "geometry_pre_correction"
        )
    if getattr(args, "save_pre_correction_geometry_npz", False):
        npz_path = save_raw_geometry_npz(
            pre_correction_records,
            os.path.join(diagnostics_output_dir, "geometry_pre_correction_raw.npz"),
        )
        print(f"[PreCorrectionGeometry] saved raw NPZ: {npz_path}")
    geometry_summary = None
    if getattr(args, "diagnose_geometry_before_correction", False):
        _, geometry_summary, _ = diagnose_geometry_records(
            records=pre_correction_records,
            output_dir=diagnostics_output_dir,
            total_samples=len(pre_correction_records),
            train_csv=getattr(args, "geometry_diagnostics_train_csv", ""),
            make_plots=True,
        )
        print("[PreCorrectionGeometry] sampling-complete summary")
        for key in (
            "ratio_dmin_lt_0.7",
            "ratio_pairs_lt_0.7_ge_2",
            "median_volume_per_atom",
            "median_atom_number_density",
            "p05_d_min",
        ):
            print(f"  {key}: {geometry_summary.get(key)}")
        print(f"[PreCorrectionGeometry] outputs: {diagnostics_output_dir}")
    if getattr(args, "geometry_diagnostics_every_batch", False):
        for round_index in range(int(args.num_rounds)):
            start = round_index * batch_size
            stop = start + batch_size
            round_records = pre_correction_records[start:stop]
            round_output = os.path.join(
                diagnostics_output_dir, f"batch_{round_index:04d}"
            )
            _, round_summary, _ = diagnose_geometry_records(
                records=round_records,
                output_dir=round_output,
                total_samples=len(round_records),
                train_csv="",
                make_plots=False,
            )
            print(
                f"[PreCorrectionGeometry][batch={round_index}] "
                f"ratio_dmin_lt_0.7={round_summary.get('ratio_dmin_lt_0.7')}, "
                f"median_volume_per_atom={round_summary.get('median_volume_per_atom')}"
            )

    # Compute generation metrics
    metrics_dict = mp20_evaluator.get_metrics(
        save=args.visualize,
        save_dir=args.save_dir + f"/epoch_{epoch}",
    )   # warning!

    for k, v in metrics_dict.items():
        print(f"{k}: {v.tolist() if isinstance(v, torch.Tensor) else v}")
    wandb.log(metrics_dict)
    wandb.log({ 'struct_validity': metrics_dict["struct_valid_rate"].sum()/num,
                'comp_validity': metrics_dict["comp_valid_rate"].sum()/num,
                'total_validity': metrics_dict["valid_rate"].sum()/num, 
                'Uniqueness': metrics_dict["unique_rate"], 
                'Novelty': metrics_dict["novel_rate"]})
    print({ 'struct_validity': metrics_dict["struct_valid_rate"].sum()/num,
            'comp_validity': metrics_dict["comp_valid_rate"].sum()/num,
            'total_validity': metrics_dict["valid_rate"].sum()/num, 
            'Uniqueness': metrics_dict["unique_rate"], 
            'Novelty': metrics_dict["novel_rate"]})
    _write_sampling_metrics(args, epoch, metrics_dict)
    pre_correction_structural = _evaluate_structural_validity_from_records(pre_correction_records)
    component_metrics = _build_component_metrics_payload(
        args=args,
        epoch=epoch,
        metrics_dict=metrics_dict,
        evaluator=mp20_evaluator,
        model_sample=model_sample,
        pre_correction_structural=pre_correction_structural,
        geometry_summary=geometry_summary,
        run_config=run_config,
    )
    _write_component_metrics(args, epoch, component_metrics)

    return metrics_dict


def analyze_and_save_pure_x(args, epoch, model_sample, nodes_dist, dataset_info, 
                     prop_dist, evaluate_condition_generation, lattice_pred_model):
    print(f'Analyzing crystal validity at epoch {epoch}...')
    _set_all_h_guard_env(args)
    batch_size = args.sample_batch_size
    device = args.device
    mp20_evaluator = CrystalGenerationEvaluator(
            dataset_cif_list=pd.read_csv(
                os.path.join(args.dataset_folder_path, f"all.csv")
            )["cif"].tolist(),
            compute_novelty=args.compute_novelty \
                if epoch >= args.compute_novelty_epoch else False
        )

    # sample the crystal structures
    nodesxsample = nodes_dist.sample(batch_size)
    if args.property_pred:
        one_hot, charges, x, node_mask, pred= sample_pure_x(args, device, model_sample, prop_dist=prop_dist,
                                            nodesxsample=nodesxsample, dataset_info=dataset_info)
    else:
        one_hot, charges, x, node_mask= sample_pure_x(args, device, model_sample, prop_dist=prop_dist,
                                            nodesxsample=nodesxsample, dataset_info=dataset_info)

    
    x = remove_mean_with_mask(x, node_mask)
    check_mask_correct([x, one_hot, charges], node_mask)
    assert_mean_zero_with_mask(x, node_mask)
    h = {'categorical': one_hot, 'integer': charges}
    xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
    atom_mask = (h['integer'].squeeze(-1)) > 0
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask.to(edge_mask.device)
    
    # predict the length, angle with the lattice_pred_model
    lattice_pred_model.eval()
    with torch.no_grad():
        length, angle = lattice_pred_model.lattice_pred(xh, node_mask, edge_mask)
    length = length.detach().cpu().numpy()
    angle = angle.detach().cpu().numpy() 

    for i in range(int(batch_size)):
        
        lattice = lattice_matrix(length[i, 0], length[i, 1], length[i, 2],
                                    angle[i, 0], angle[i, 1], angle[i, 2])
        mask = node_mask[i].squeeze(-1).bool()

        if args.frac_coords_mode:
            frac_coords = frac_coords[i][mask].detach().cpu().numpy()
            x_valid = frac_to_cart(frac_coords, lattice)
        else: 
            x_valid = x[i][mask].detach().cpu().numpy()
            frac_coords = cart_to_frac(x_valid, lattice)

        one_hot_valid = one_hot[i][mask].detach().cpu().numpy()
        atom_types = np.argmax(one_hot_valid, axis=-1)  # convert one-hot to atom types
        species_symbols = _atom_type_symbols(atom_types.tolist())
        _guard_atom_types_or_raise(
            args,
            atom_types,
            species_symbols,
            sample_global_index=i,
            stage="analyze_and_save_pure_x_pre_append",
        )
        # charges = charges[i][mask].detach().cpu().numpy()

        if i <=2:
            print("sampled lengths:", length[i])
            print("sampled angles:", angle[i])
            # print("sampled atom types:", atom_types)

        mp20_evaluator.append_pred_array(
                {
                    "atom_types": atom_types,
                    "pos": x_valid,
                    "frac_coords": frac_coords,
                    "lengths": length[i],
                    "angles": angle[i],
                    "sample_idx": f"epoch_{epoch}_sample_{i}"
                }
            )

    # Compute generation metrics
    metrics_dict = mp20_evaluator.get_metrics(
        save=args.visualize,
        save_dir=args.save_dir + f"/epoch_{epoch}",
    )   # warning!


    for k, v in metrics_dict.items():
        print(f"{k}: {v.tolist() if isinstance(v, torch.Tensor) else v}")

    wandb.log(metrics_dict)
    wandb.log({'Validity': metrics_dict["valid_rate"].sum()/batch_size, 
               'Uniqueness': metrics_dict["unique_rate"], 
               'Novelty': metrics_dict["novel_rate"]})
    
    print({'Validity': metrics_dict["valid_rate"].sum()/batch_size, 
               'Uniqueness': metrics_dict["unique_rate"], 
               'Novelty': metrics_dict["novel_rate"]})
    _write_sampling_metrics(args, epoch, metrics_dict)

    return metrics_dict


def analyze_and_save_L(args, epoch, model_sample, nodes_dist, dataset_info):
    print(f'Analyzing crystal validity at epoch {epoch}...')
    batch_size = args.sample_batch_size
    device = args.device

    # sample the crystal structures
    nodesxsample = nodes_dist.sample(batch_size)

    length, angle = sample_L(args, device, model_sample, dataset_info, nodesxsample=nodesxsample)
    length = length.detach().cpu().numpy()
    angle = angle.detach().cpu().numpy() 

    for i in range(int(batch_size)):
        lattice = lattice_matrix(length[i, 0], length[i, 1], length[i, 2],
                                    angle[i, 0], angle[i, 1], angle[i, 2])
        if i <= 5:
            print("sampled lengths:", length[i])
            print("sampled angles:", angle[i])
            print("sampled lattice:", lattice)


#######################################################################################################

class CrystalGenerationEvaluator:
    """Evaluator for crystal generation tasks.

    Can be used within a Lightning module by appending sampled structures and computing metrics at
    the end of an epoch.
    """

    def __init__(
        self,
        dataset_cif_list,
        stol=0.5,
        angle_tol=10,
        ltol=0.3,
        device="cpu",
        compute_novelty=False,
    ):
        self.dataset_cif_list = dataset_cif_list
        self.dataset_struct_list = None  # loader first time it is required
        self.matcher = build_structure_matcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.pred_arrays_list = []
        self.pred_crys_list = []
        self.device = device
        self.compute_novelty = compute_novelty

    def append_pred_array(self, pred: Dict):
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)

    def clear(self):
        """Clear the stored predictions, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.pred_crys_list = []

    def _arrays_to_crystals(self, save: bool = False, save_dir: str = ""):
        """Convert stored predictions and ground truths to Crystal objects for evaluation."""
        converter = partial(
            array_dict_to_crystal,
            save=save,
            save_dir_name=save_dir,
        )
        if len(self.pred_arrays_list) <= 16:
            self.pred_crys_list = [
                converter(pred)
                for pred in tqdm(
                    self.pred_arrays_list,
                    desc="    Pred to Crystal",
                    total=len(self.pred_arrays_list),
                )
            ]
            return
        self.pred_crys_list = joblib_map(
            converter,
            self.pred_arrays_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc=f"    Pred to Crystal",
            total=len(self.pred_arrays_list),
        )

    def _dataset_cif_to_struct(self):
        """Convert dataset CIFs to Structure objects for novelty evaluation."""
        if self.dataset_struct_list is None:
            self.dataset_struct_list = joblib_map(
                partial(Structure.from_str, fmt="cif"),
                self.dataset_cif_list,
                n_jobs=-4,
                inner_max_num_threads=1,
                desc="    Load dataset CIFs (one time)",
                total=len(self.dataset_cif_list),
            )

    def _get_novelty(self, struct):
        # matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
        return is_structure_novel(
            struct,
            self.dataset_struct_list,
            self.matcher,
            skip_structure_reduction=DEFAULT_SKIP_STRUCTURE_REDUCTION,
        )

    # !!!!!!
    def get_metrics(self, save: bool = False, save_dir: str = ""):
        assert len(self.pred_arrays_list) > 0, "No predictions to evaluate."

        # Convert predictions and ground truths to Crystal objects
        self._arrays_to_crystals(save, save_dir)

        # Compute validity metrics
        metrics_dict = {
            "valid_rate": torch.tensor([c.valid for c in self.pred_crys_list], device=self.device),
            "comp_valid_rate": torch.tensor(
                [c.comp_valid for c in self.pred_crys_list], device=self.device
            ),
            "struct_valid_rate": torch.tensor(
                [c.struct_valid for c in self.pred_crys_list], device=self.device
            ),
        }

        # Compute uniqueness
        valid_structs = [c.structure for c in self.pred_crys_list if c.valid]
        unique_struct_groups = self.matcher.group_structures(valid_structs)
        self._last_valid_structs = valid_structs
        self._last_unique_struct_groups = unique_struct_groups
        if len(valid_structs) > 0:
            metrics_dict["unique_rate"] = torch.tensor(
                len(unique_struct_groups) / len(valid_structs), device=self.device
            )
        else:
            metrics_dict["unique_rate"] = torch.tensor(0.0, device=self.device)

        # Compute novelty (slow to compute)
        if self.compute_novelty:
            self._dataset_cif_to_struct()
            struct_is_novel = []
            for struct in tqdm(
                [group[0] for group in unique_struct_groups],
                desc="    Novelty",
                total=len(unique_struct_groups),
            ):
                struct_is_novel.append(self._get_novelty(struct))
            self._last_unique_group_is_novel = struct_is_novel

            metrics_dict["novel_rate"] = torch.tensor(
                sum(struct_is_novel) / len(struct_is_novel), device=self.device
            )
        else:
            self._last_unique_group_is_novel = None
            metrics_dict["novel_rate"] = torch.tensor(-1.0, device=self.device)

        return metrics_dict

    def get_wandb_table(self, current_epoch: int = 0, save_dir: str = ""):
        # Log crystal structures and metrics to wandb
        pred_table = wandb.Table(
            columns=[
                "Global step",
                "Sample idx",
                "Num atoms",
                "Valid?",
                "Comp valid?",
                "Struct valid?",
                "Pred atom types",
                "Pred lengths",
                "Pred angles",
                "Pred 2D",
            ]
        )

        for idx in range(len(self.pred_crys_list)):
            sample_idx = self.pred_crys_list[idx].sample_idx

            num_atoms = len(self.pred_crys_list[idx].atom_types)

            pred_atom_types = " ".join([str(int(t)) for t in self.pred_crys_list[idx].atom_types])

            pred_lengths = " ".join([f"{l:.2f}" for l in self.pred_crys_list[idx].lengths])

            pred_angles = " ".join([f"{a:.2f}" for a in self.pred_crys_list[idx].angles])

            try:
                pred_2d = ase_view.make_wandb_image(
                    self.pred_crys_list[idx].structure,
                    center_in_uc=False,
                )
            except Exception as e:
                log.error(f"Failed to load 2D structure for pred sample {sample_idx}.")
                pred_2d = None

            # Update table
            pred_table.add_data(
                current_epoch,
                sample_idx,
                num_atoms,
                self.pred_crys_list[idx].valid,
                self.pred_crys_list[idx].comp_valid,
                self.pred_crys_list[idx].struct_valid,
                pred_atom_types,
                pred_lengths,
                pred_angles,
                pred_2d,
            )

        return pred_table
    
    

def test(args, loader, info, epoch, eval_model, property_norms, nodes_dist, partition='Test'):
    print(f"Testing {partition} at epoch {epoch}...")
    one_hot_shape = max(info['atom_encoder'].values())
    device = args.device
    dtype = args.dtype
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            props = data.propertys # 理化性质, a list of dict
            data = reshape(data, device, dtype, include_charges=True)
            x = data['positions'].to(device, dtype) 
            frac_coords = data['frac_coords'].to(device, dtype)
            lengths = data['lengths'].to(device, dtype)
            angles = data['angles'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'][:,:,:one_hot_shape].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
            
            if args.bond_pred:
                edge_index = data['edge_index'].to(device, dtype)
                edge_attr = data['edge_attr'].to(device, dtype)
                bond_info = {'edge_index': edge_index, 'edge_attr': edge_attr}
            else:
                bond_info = None

            x = remove_mean_with_mask(x, node_mask) # 后续暂时不给x加噪声
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = prepare_context_test(args.conditioning, data, props, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            # print(x.shape, h['categorical'].shape, h['integer'].shape, lengths.shape, angles.shape)
            if args.frac_coords_mode:
                # print("using frac_coords to compute loss")
                nll, _, _, loss_dict = compute_loss_and_nll(args, eval_model, nodes_dist, 
                                                frac_coords, h, lengths, angles, node_mask, edge_mask, context,
                                                property_label=props[args.target_property].to(device, dtype) \
                                                    if args.target_property in props else None,)
            else:
                nll, _, _, loss_dict = compute_loss_and_nll(args, eval_model, nodes_dist, 
                                                x, h, lengths, angles, node_mask, edge_mask, context, 
                                                bond_info=bond_info, property_label=props[args.target_property].to(device, dtype) \
                                                    if args.target_property in props else None)
        
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                if args.probabilistic_model == 'diffusion_transformer' or args.probabilistic_model == 'diffusion_Lfirst':
                    if 'total_error' in loss_dict:
                        print(f"\r {partition} \t epoch: {epoch}, iter: {i}/{n_iterations}, " 
                              f"NLL: {nll_epoch/n_samples:.2f}", end=', ')
                        print(f"denoise x: {loss_dict['x_error'].mean().item():.3f}, " 
                              f"denoise l: {loss_dict['l_error'].mean().item():.3f}, "
                              f"denoise a: {loss_dict['a_error'].mean().item():.3f} ",
                              f"total xla denoise: {loss_dict['total_error'].mean().item():.3f}", 
                              end = '')
                        wandb.log({f"{partition}_denoise_x": loss_dict['x_error'].mean().item()}, commit=True)
                        wandb.log({f"{partition}_denoise_l": loss_dict['l_error'].mean().item()}, commit=True)
                        wandb.log({f"{partition}_denoise_a": loss_dict['a_error'].mean().item()}, commit=True)
                        wandb.log({f"{partition}_denoise_xla": loss_dict['total_error'].mean().item()}, commit=True)
                    if 'atom_type_loss' in loss_dict:
                        print(f', atom_type_loss: {loss_dict["atom_type_loss"].mean():.3f}', end='\n')
                        wandb.log({f"{partition}_denoise_atom_type_loss": loss_dict['atom_type_loss'].mean().item()}, commit=True)
                    if args.property_pred:
                        if not isinstance(loss_dict['pred_loss'], int):
                            print(f", pred_loss: {loss_dict['pred_loss'].mean().item():.3f}", end='')
                        print(f", pred_rate: {loss_dict['pred_rate'].mean().item():.3f}")
                elif args.probabilistic_model == 'diffusion_Lhard':
                    print(f"\r {partition} \t epoch: {epoch}, iter: {i}/{n_iterations}, " 
                              f"NLL: {nll_epoch/n_samples:.2f}", end=', ')
                    print(f"denoise x: {loss_dict['x_error'].mean().item():.3f}", end = '')
                    if 'atom_type_loss' in loss_dict:
                        print(f', atom_type_loss: {loss_dict["atom_type_loss"].mean():.3f}', end='\n')
                    if args.property_pred:
                        if not isinstance(loss_dict['pred_loss'], int):
                            print(f", pred_loss: {loss_dict['pred_loss'].mean().item():.3f}", end='')
                        print(f", pred_rate: {loss_dict['pred_rate'].mean().item():.3f}")
                    wandb.log({f"{partition}_denoise_x": loss_dict['x_error'].mean().item()}, commit=True)
                else: # other models
                    print(f"\r {partition} \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                        f"NLL: {nll_epoch/n_samples:.2f}")
                    print(f"error: {loss_dict['error'].mean().item():.3f}, ", end='')
                    if 'lattice_loss' in loss_dict:
                        print(f"lattice_loss: {loss_dict['lattice_loss'].mean().item():.3f}, ", end='')
                    print(f"kl_prior: {loss_dict['kl_prior'].mean().item():.3f}, "
                        f"loss_term_0: {loss_dict['loss_term_0'].mean().item():.2f}, "
                        f"neg_log_constants: {loss_dict['neg_log_constants'].mean().item():.3f}, "
                        f"estimator_loss_terms: {loss_dict['estimator_loss_terms'].mean().item():.3f}, ",
                        f"loss: {loss_dict['loss'].mean().item():.3f}, ",
                        f"loss_t: {loss_dict['loss_t'].mean().item():.3f}, "
                        f"loss_t_larger_than_zero: {loss_dict['loss_t_larger_than_zero'].mean().item():.3f}, ",
                        f"atom_type_loss: {loss_dict['atom_type_loss'].mean().item():.3f}"
                        )

    return nll_epoch/n_samples
 

def test_F(args, loader, info, epoch, eval_model, property_norms, nodes_dist, partition='Test'):
    print(f"Testing {partition} at epoch {epoch}...")
    one_hot_shape = max(info['atom_encoder'].values())
    device = args.device
    dtype = args.dtype
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            props = data.propertys # 理化性质, a list of dict
            data = reshape(data, device, dtype, include_charges=True)
            x = data['positions'].to(device, dtype) 
            frac_coords = data['frac_coords'].to(device, dtype)
            lengths = data['lengths'].to(device, dtype)
            angles = data['angles'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'][:,:,:one_hot_shape].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
            
            if args.bond_pred:
                edge_index = data['edge_index'].to(device, dtype)
                edge_attr = data['edge_attr'].to(device, dtype)
                bond_info = {'edge_index': edge_index, 'edge_attr': edge_attr}
            else:
                bond_info = None

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = prepare_context_test(args.conditioning, data, props, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            nll, _, _, loss_dict = compute_loss_and_nll(args, eval_model, nodes_dist, 
                                            frac_coords, h, lengths, angles, node_mask, edge_mask, context,
                                            property_label=props[args.target_property].to(device, dtype) \
                                                if args.target_property in props else None,)
        
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size

            if i % args.n_report_steps == 0:
                if args.probabilistic_model == 'diffusion_LF' or args.probabilistic_model == 'diffusion_LF_wrap':
                    print(f"\r {partition} \t epoch: {epoch}, iter: {i}/{n_iterations}, " 
                            f"NLL: {nll_epoch/n_samples:.4f}", 
                            f"loss: {loss_dict['loss'].mean().item():.4f}, ",
                            end='')
                    print(f"denoise x: {loss_dict['x_error'].mean().item():.4f}", end = '')
                    wandb.log({f"{partition}_denoise_x": loss_dict['x_error'].mean().item()}, commit=True)
                    if 'atom_type_loss' in loss_dict:
                        print(f', atom_type_loss: {loss_dict["atom_type_loss"].mean():.4f}', end='')
                        wandb.log({f"{partition}_atom_type_loss": loss_dict['atom_type_loss'].mean().item()}, commit=True)
                    if 'atom_type_loss_adjust' in loss_dict:
                        print(f', atom_type_loss_adjust: {loss_dict["atom_type_loss_adjust"].mean():.4f}', end='')
                        wandb.log({f"{partition}_atom_type_loss_adjust": loss_dict['atom_type_loss_adjust'].mean().item()}, commit=True)
                    if 'repulsion_loss' in loss_dict:
                        print(f', repulsion_loss: {loss_dict["repulsion_loss"].mean():.4f}', end='')
                        wandb.log({f"{partition}_repulsion_loss": loss_dict['repulsion_loss'].mean().item()}, commit=True)
                    if 'loss_term_0' in loss_dict and 'neg_log_constants' in loss_dict \
                        and 'estimator_loss_terms' in loss_dict and 'loss_t' in loss_dict:
                        print(f"loss_term_0: {loss_dict['loss_term_0'].mean().item():.2f}, "
                            f"neg_log_constants: {loss_dict['neg_log_constants'].mean().item():.3f}, "
                            f"estimator_loss_terms: {loss_dict['estimator_loss_terms'].mean().item():.3f}, ",
                            f"loss_t: {loss_dict['loss_t'].mean().item():.3f}, ",
                            f"loss_t_larger_than_zero: {loss_dict['loss_t_larger_than_zero'].mean().item():.3f}",
                            end='\n')
                    if args.property_pred:
                        if not isinstance(loss_dict['pred_loss'], int):
                            print(f"pred_loss: {loss_dict['pred_loss'].mean().item():.3f}", end='')
                        print(f"pred_rate: {loss_dict['pred_rate'].mean().item():.3f}")
                    print("", end='\n')
                else: 
                    raise NotImplementedError

    return nll_epoch/n_samples


def test_pure_x(args, loader, info, epoch, eval_model, property_norms, nodes_dist, partition='Test'):
    print(f"Testing {partition} at epoch {epoch}...")
    one_hot_shape = max(info['atom_encoder'].values())
    device = args.device
    dtype = args.dtype
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            props = data.propertys # 理化性质, a list of dict
            data = reshape(data, device, dtype, include_charges=True)
            x = data['positions'].to(device, dtype) 
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'][:,:,:one_hot_shape].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
            
            if args.bond_pred:
                edge_index = data['edge_index'].to(device, dtype)
                edge_attr = data['edge_attr'].to(device, dtype)
                bond_info = {'edge_index': edge_index, 'edge_attr': edge_attr}
            else:
                bond_info = None

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = prepare_context_test(args.conditioning, data, props, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            nll, _, _, loss_dict = compute_loss_and_nll_pure_x(args, eval_model, nodes_dist, x, h,
                                            node_mask, edge_mask, context, bond_info=bond_info,
                                            property_label=props[args.target_property].to(device, dtype) \
                                                if args.target_property in props else None)
        
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch/n_samples:.2f}", end='\n')
                print(f"error: {loss_dict['error'].mean().item():.3f}, "
                      f"kl_prior: {loss_dict['kl_prior'].mean().item():.3f}, "
                      f"loss_term_0: {loss_dict['loss_term_0'].mean().item():.2f}, "
                      f"neg_log_constants: {loss_dict['neg_log_constants'].mean().item():.3f}, "
                      f"estimator_loss_terms: {loss_dict['estimator_loss_terms'].mean().item():.3f}, ",
                      f"loss: {loss_dict['loss'].mean().item():.3f}, ",
                      f"loss_t: {loss_dict['loss_t'].mean().item():.3f}, "
                      f"loss_t_larger_than_zero: {loss_dict['loss_t_larger_than_zero'].mean().item():.3f}, ",
                      f"atom_type_loss: {loss_dict['atom_type_loss'].mean().item():.3f}"
                      )

    return nll_epoch/n_samples


def test_L(args, loader, info, epoch, eval_model, partition='Test'):
    print(f"Testing {partition} at epoch {epoch}...")
    one_hot_shape = max(info['atom_encoder'].values())
    device = args.device
    dtype = args.dtype
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0
        n_iterations = len(loader)

        for i, data in enumerate(loader):
            data = reshape(data, device, dtype, include_charges=True)
            lengths = data['lengths'].to(device, dtype)
            angles = data['angles'].to(device, dtype)
            num_atoms = data['num_atoms'].to(device).long().reshape(-1)
            batch_size = lengths.size(0)
            
            nll, _, _, loss_dict = compute_loss_and_nll_L(
                args, eval_model, lengths, angles, num_atoms=num_atoms
            )

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                if args.probabilistic_model == 'diffusion_L' or args.probabilistic_model == 'diffusion_L_another':
                    if 'total_error' in loss_dict:
                        print(f"\r {partition} \t epoch: {epoch}, iter: {i}/{n_iterations}, " 
                              f"NLL: {nll_epoch/n_samples:.2f}", end=', ')
                        print(f"denoise l: {loss_dict['l_error'].mean().item():.3f}, "
                              f"denoise a: {loss_dict['a_error'].mean().item():.3f} ",
                              f"total la denoise: {loss_dict['total_error'].mean().item():.3f}", 
                              end = '\n')
                        wandb.log({f"{partition}_denoise_l": loss_dict['l_error'].mean().item()}, commit=True)
                        wandb.log({f"{partition}_denoise_a": loss_dict['a_error'].mean().item()}, commit=True)
                        wandb.log({f"{partition}_denoise_la": loss_dict['total_error'].mean().item()}, commit=True)

                else: 
                    raise ValueError(args.probabilistic_model)
    return nll_epoch/n_samples
