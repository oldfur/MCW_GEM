import os
import torch
import datetime

class GradientWatcher:
    """
    自动监控 PyTorch 模型的梯度状况：
    - 检测 NaN / Inf / 爆炸梯度
    - 打印警告并记录到日志文件
    """

    def __init__(self, model, threshold=50.0, log_path="grad_log.txt", verbose=True):
        """
        Args:
            model: nn.Module 对象
            threshold: 超过此值打印警告（默认 50）
            log_path: 日志文件路径（默认 grad_log.txt）
            verbose: 是否打印详细信息
        """
        self.model = model
        self.threshold = threshold
        self.verbose = verbose
        self.log_path = log_path
        self.handles = []
        self._create_log_file()
        self.register_hooks()

    def _create_log_file(self):
        """初始化日志文件"""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True) if "/" in self.log_path else None
        with open(self.log_path, "w") as f:
            f.write(f"Gradient Watcher Log — {datetime.datetime.now()}\n")
            f.write("=" * 80 + "\n")

    def _log(self, message):
        """写入日志文件"""
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

    def register_hooks(self):
        """为每个可训练参数注册梯度 hook"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name):
        def hook(grad):
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            # 检查 NaN / Inf
            if torch.isnan(grad).any():
                msg = f"[{timestamp}] 🚨 NaN detected in gradient of [{name}]"
                print(msg)
                self._log(msg)
            elif torch.isinf(grad).any():
                msg = f"[{timestamp}] 🚨 Inf detected in gradient of [{name}]"
                print(msg)
                self._log(msg)

            # 检查梯度爆炸
            grad_abs_max = grad.abs().max().item()
            grad_norm = grad.norm(2).item()
            if grad_abs_max > self.threshold:
                msg = f"[{timestamp}] ⚠️ Large grad in [{name}] | max={grad_abs_max:.2f}, norm={grad_norm:.2f}"
                # print(msg)
                self._log(msg)

            # 打印调试信息
            if self.verbose and grad_abs_max <= self.threshold:
                msg = f"[{timestamp}] Grad OK in [{name}] | max={grad_abs_max:.2f}, norm={grad_norm:.2f}"
                self._log(msg)

            return grad
        return hook

    def remove(self):
        """移除所有 hook"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        print("✅ GradientWatcher hooks removed.")

