import torch
import os

def analyze_tensor(t, name, indent="  "):
    """分析单个 Tensor 的统计信息"""
    if not isinstance(t, torch.Tensor):
        print(f"{indent}- {name}: (not a tensor, type={type(t)})")
        return

    total = t.numel()
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    n_finite = torch.isfinite(t).sum().item()

    finite_ratio = n_finite / total if total > 0 else 0
    summary = f"{indent}- {name}: shape={tuple(t.shape)}, dtype={t.dtype}"
    print(summary)
    print(f"{indent}  finite={finite_ratio*100:.2f}%, NaN={n_nan}, Inf={n_inf}")

    if n_finite > 0:
        finite_vals = t[torch.isfinite(t)]
        print(f"{indent}  min={finite_vals.min().item():.4e}, "
              f"max={finite_vals.max().item():.4e}, "
              f"mean={finite_vals.mean().item():.4e}, "
              f"std={finite_vals.std().item():.4e}")
    print()


def analyze_debug_file(path):
    print(f"📂 Loading debug info from: {path}")
    data = torch.load(path, map_location="cpu")

    # ===== Layer Info =====
    layer_name = data.get("layer_name", "UnknownLayer")
    print(f"\n===== 🧭 Layer: {layer_name} =====\n")

    # ===== Parameter Summary =====
    nan_params = data.get("nan_params", [])
    if nan_params:
        print("===== ❌ NaN/Inf Parameters Detected =====")
        for name, nan_count, total in nan_params:
            ratio = nan_count / total * 100
            print(f"  - {name}: {nan_count}/{total} ({ratio:.2f}%) NaN/Inf")
        print()
    else:
        print("✅ All parameters were finite at save time.\n")

    # ===== Parameters =====
    params = data.get("parameters", {})
    if params:
        print("===== ⚙️ Parameter Statistics =====")
        for name, tensor in params.items():
            analyze_tensor(tensor, name)
    else:
        print("⚠️ No parameter tensors saved.\n")

    # ===== Inputs =====
    inputs = data.get("input", [])
    print("===== 🎯 Inputs =====")
    if not inputs:
        print("⚠️ No inputs saved.\n")
    else:
        for i, t in enumerate(inputs):
            analyze_tensor(t, f"input[{i}]")

    # ===== Output =====
    output = data.get("output", None)
    print("===== 📤 Output =====")
    if output is None:
        print("⚠️ No output tensor saved.\n")
    else:
        analyze_tensor(output, "output")

    print("✅ Analysis complete.\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_nan_debug_info.py <path_to_nan_debug.pt>")
        exit(0)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        exit(1)

    analyze_debug_file(path)
