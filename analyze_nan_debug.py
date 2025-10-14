#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze NaN debug info saved by save_nan_debug_info()
Usage:
    python analyze_nan_debug.py nan_debug_fc_time.pt
"""

import sys
import torch
import matplotlib.pyplot as plt

def analyze_tensor(name, tensor, plot=False):
    """打印张量统计信息"""
    if not isinstance(tensor, torch.Tensor):
        print(f"❌ {name}: Not a tensor, type={type(tensor)}")
        return

    finite_mask = torch.isfinite(tensor)
    numel = tensor.numel()
    num_nan = (~finite_mask).sum().item()

    print(f"🔹 {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print(f"   finite={finite_mask.all().item()}, NaN/Inf count={num_nan}/{numel}")
    

    if finite_mask.any():
        t_valid = tensor[finite_mask]
        print(f"   min={t_valid.min().item():.4e}, max={t_valid.max().item():.4e}, "
              f"mean={t_valid.mean().item():.4e}, std={t_valid.std().item():.4e}")
        absmax = t_valid.abs().max().item()
        if absmax > 1e4:
            print(f"   ⚠️  Large magnitude values detected! absmax={absmax:.2e}")
    else:
        print("   ⚠️ Entire tensor is NaN/Inf!")

    if plot and finite_mask.any():
        plt.figure()
        plt.hist(t_valid.cpu().numpy().flatten(), bins=100)
        plt.title(name)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_nan_debug.py <debug_file.pt>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"📂 Loading: {path}")
    data = torch.load(path, map_location="cpu")

    print("\n===== 🧭 Layer Info =====")
    print(f"Layer name: {data.get('layer_name', 'Unknown')}")

    print("\n===== ⚙️ Parameters =====")
    params = data.get("parameters", {})
    if not params:
        print("No parameters found.")
    for name, p in params.items():
        analyze_tensor(f"Param[{name}]", p)

    print("\n===== 🎯 Inputs =====")
    inputs = data.get("input", [])
    if not inputs:
        print("No inputs found.")
    for i, x in enumerate(inputs):
        if isinstance(x, torch.Tensor):
            analyze_tensor(f"Input[{i}]", x)
        else:
            print(f"Input[{i}] is not a tensor (type={type(x)})")

    print("\n===== 📤 Output =====")
    output = data.get("output", None)
    if isinstance(output, torch.Tensor):
        analyze_tensor("Output", output, plot=True)
    else:
        print(f"Output type: {type(output)}")

    print("\n✅ Analysis complete.")


if __name__ == "__main__":
    main()
