import torch
from torch import nn
from chgnet.trainer.trainer import Trainer
from chgnet.data.dataset import TORCH_DTYPE


class MaskedMSELoss(nn.Module):
    """MSE with NaN mask support for incomplete property labels."""

    def forward(self, pred, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype), 0

        diff = pred[mask] - target[mask]
        loss = (diff ** 2).mean()
        mae = diff.abs().mean()
        size = mask.sum().item()
        return loss, mae, size


class MultiTaskTrainer(Trainer):
    """
    Extends CHGNet Trainer to support an additional dense
    multi-property regression head ("prop").
    """

    def __init__(
        self,
        *args,
        prop_weight: float = 1.0,
        prop_dim: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.prop_weight = prop_weight
        self.prop_dim = prop_dim
        self.prop_loss_fn = MaskedMSELoss()

        print(f"[MultiTaskTrainer] Enabled property head "
              f"(dim={prop_dim}, weight={prop_weight})")

    # -------------------------------
    # override _compute_loss wrapper
    # -------------------------------
    def _compute_multitask_loss(self, targets, prediction):
        """
        Wrap parent CombinedLoss + add property regression loss.
        """

        # base efsm loss
        combined_loss = self.criterion(targets, prediction)
        total_loss = combined_loss["loss"]

        # -----------------------------------
        # property multitask head (optional)
        # -----------------------------------
        if "prop" in targets and "property_pred" in prediction:
            prop_t = torch.stack(targets["prop"]).to(self.device, dtype=TORCH_DTYPE)
            prop_p = prediction["property_pred"]

            prop_loss, prop_mae, prop_n = self.prop_loss_fn(prop_p, prop_t)

            combined_loss["prop_loss"] = prop_loss
            combined_loss["prop_MAE"] = prop_mae
            combined_loss["prop_MAE_size"] = prop_n

            total_loss = total_loss + self.prop_weight * prop_loss

        combined_loss["loss"] = total_loss
        return combined_loss

    # --------------------------------
    # training step override
    # --------------------------------
    def _train(self, train_loader, current_epoch, wandb_log_freq="batch"):
        """
        Same as base Trainer, except loss = base + property_loss.
        """
        from chgnet.trainer.trainer import AverageMeter
        import time
        import numpy as np

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        mae_errors = {k: AverageMeter() for k in self.targets}
        mae_errors["prop"] = AverageMeter()

        self.model.train()

        start = time.perf_counter()

        for idx, (graphs, targets) in enumerate(train_loader):
            data_time.update(time.perf_counter() - start)

            for g in graphs:
                requires_force = "f" in self.targets
                g.atom_frac_coord.requires_grad = requires_force

            graphs = [g.to(self.device) for g in graphs]
            targets = {k: self.move_to(v, self.device) for k, v in targets.items()}

            prediction = self.model(graphs, task=self.targets)

            combined_loss = self._compute_multitask_loss(targets, prediction)

            losses.update(combined_loss["loss"].item(), len(graphs))

            # efsm MAE
            for key in self.targets:
                mae_errors[key].update(
                    combined_loss[f"{key}_MAE"].item(),
                    combined_loss[f"{key}_MAE_size"],
                )

            # prop MAE
            if "prop_MAE" in combined_loss:
                mae_errors["prop"].update(
                    combined_loss["prop_MAE"].item(),
                    combined_loss["prop_MAE_size"],
                )

            self.optimizer.zero_grad()
            combined_loss["loss"].backward()
            self.optimizer.step()

            if idx + 1 in np.arange(1, 11) * len(train_loader) // 10:
                self.scheduler.step()

            del graphs, targets, prediction, combined_loss

            batch_time.update(time.perf_counter() - start)
            start = time.perf_counter()

        return {k: round(v.avg, 6) for k, v in mae_errors.items()}

    # --------------------------------
    # validation override
    # --------------------------------
    def _validate(self, *args, **kwargs):
        """
        Reuse parent's validation loop but replace loss combiner.
        """
        from chgnet.trainer.trainer import AverageMeter
        import time

        val_loader = args[0]
        is_test = kwargs.get("is_test", False)

        batch_time = AverageMeter()
        losses = AverageMeter()

        mae_errors = {k: AverageMeter() for k in self.targets}
        mae_errors["prop"] = AverageMeter()

        self.model.eval()

        end = time.perf_counter()

        for graphs, targets in val_loader:
            with torch.no_grad():
                graphs = [g.to(self.device) for g in graphs]
                targets = {k: self.move_to(v, self.device) for k, v in targets.items()}

                prediction = self.model(graphs, task=self.targets)

                combined_loss = self._compute_multitask_loss(targets, prediction)

                losses.update(combined_loss["loss"].item(), len(graphs))

                for key in self.targets:
                    mae_errors[key].update(
                        combined_loss[f"{key}_MAE"].item(),
                        combined_loss[f"{key}_MAE_size"],
                    )

                if "prop_MAE" in combined_loss:
                    mae_errors["prop"].update(
                        combined_loss["prop_MAE"].item(),
                        combined_loss["prop_MAE_size"],
                    )

            del graphs, targets, prediction, combined_loss

            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

        print("VAL MAE:", {k: v.avg for k, v in mae_errors.items()})
        return {k: round(v.avg, 6) for k, v in mae_errors.items()}
