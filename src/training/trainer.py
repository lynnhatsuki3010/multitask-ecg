"""
trainer.py
Training loop for multi-task ECG Transformer.
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts,
    StepLR, LinearLR, SequentialLR, OneCycleLR,
)
from typing import Dict, Optional, Union
from tqdm import tqdm

from src.utils.metrics import (
    MetricsAccumulator, ARRHYTHMIA_LABELS, MI_LABELS,
    find_optimal_thresholds, apply_thresholds,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def print_metrics(metrics: Dict[str, Union[float, str]], prefix: str = ""):
    """Pretty-print a flat metrics dict, grouped by prefix."""
    groups = {}
    reports = {}
    for k, v in metrics.items():
        if isinstance(v, str):
            reports[k] = v
            continue
        parts = k.split("/")
        g = parts[0] if len(parts) > 1 else "other"
        groups.setdefault(g, {})[k] = v

    for g, gm in groups.items():
        print(f"  [{g}]")
        for k, v in gm.items():
            print(f"    {k}: {v:.4f}")

    for k, report in reports.items():
        print(f"\n  === {k} ===")
        print(report)


def save_checkpoint(model, optimizer, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics":   metrics,
    }, path)


# ─── Main Trainer ─────────────────────────────────────────────────────────────

class Trainer:
    """
    Handles training, validation, and checkpointing for the multi-task ECG model.
    Set update_weights=False to run in random-weight eval mode (no gradient updates).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
        arrhythmia_label_indices: list,   # indices in full label vector that belong to arrhythmia
        mi_label_indices: list,           # indices in full label vector that belong to MI
        arrhythmia_label_names: list = ARRHYTHMIA_LABELS,
        mi_label_names: list = MI_LABELS,
        hrv_feature_names: list = ["rmssd", "sdnn", "mean_hr"],
        update_weights: bool = True,
        optuna_trial = None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.optuna_trial = optuna_trial
        self.arrhythmia_idx = arrhythmia_label_indices
        self.mi_idx = mi_label_indices
        self.arrhythmia_label_names = arrhythmia_label_names
        self.mi_label_names = mi_label_names
        self.hrv_feature_names = hrv_feature_names
        self.hrv_enabled = cfg.get("hrv", {}).get("enabled", True)

        train_cfg = cfg.get("training", {})
        self.update_weights    = update_weights
        self.epochs    = train_cfg.get("epochs", 50)
        self.threshold = cfg.get("eval", {}).get("threshold", 0.5)
        self.save_every = train_cfg.get("save_every", 5)
        self.patience   = train_cfg.get("early_stopping_patience", 10)
        self.checkpoint_dir = cfg.get("paths", {}).get("checkpoints", "checkpoints")

        # ── Optimizer — AdamW with differential weight decay ─────────────────────
        # Embedding / LayerNorm / bias params should NOT be weight-decayed:
        # regularizing them causes representation collapse on small datasets.
        # Only projection / attention / FFN weight matrices get weight_decay.
        lr           = float(train_cfg.get("lr", 1e-4))
        weight_decay = float(train_cfg.get("weight_decay", 5e-3))

        no_decay_names = ("bias", "norm", "embed", "cls_token", "pos_embed")
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name.lower() for nd in no_decay_names):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": decay_params,    "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
        )

        # ── Scheduler ──────────────────────────────────────────────────────
        sched_type    = train_cfg.get("scheduler", "cosine")
        warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
        steps_per_epoch = len(train_loader)  # needed for OneCycleLR

        if sched_type == "one_cycle":
            # OneCycleLR: warmup → peak → smooth cosine decay.
            # No discontinuous restarts — ideal for multi-task loss balancing.
            pct_start = train_cfg.get("one_cycle_pct_start", 0.2)  # 20% warmup
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=lr,
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=pct_start,
                anneal_strategy="cos",
                div_factor=25.0,      # initial_lr = max_lr / 25
                final_div_factor=1e4, # final_lr   = initial_lr / 1e4
            )
            self._step_scheduler_per_batch = True  # OneCycleLR steps per batch
        elif sched_type == "cosine_restart":
            # CosineAnnealingWarmRestarts: T_0 chu kỳ đầu, T_mult=2 tăng dần
            T_0 = train_cfg.get("cosine_T0", max(10, self.epochs // 3))
            main_sched = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=T_0, T_mult=2, eta_min=1e-6
            )
            self._step_scheduler_per_batch = False
        elif sched_type == "cosine":
            main_sched = CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=1e-6
            )
            self._step_scheduler_per_batch = False
        elif sched_type == "step":
            main_sched = StepLR(self.optimizer, step_size=10, gamma=0.5)
            self._step_scheduler_per_batch = False
        else:
            main_sched = None
            self._step_scheduler_per_batch = False

        if sched_type != "one_cycle":
            if warmup_epochs > 0 and main_sched is not None:
                warmup_sched = LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_sched, main_sched],
                    milestones=[warmup_epochs],
                )
            else:
                self.scheduler = main_sched

        # History
        self.history: Dict[str, list] = {"train": [], "val": []}
        self.best_val_auroc = -1.0
        self.no_improve_count = 0

    # ── Split labels ──────────────────────────────────────────────────────────

    def _split_labels(self, batch_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split the full label vector into per-task tensors."""
        return {
            "arrhythmia": batch_labels[:, self.arrhythmia_idx],
            "mi":         batch_labels[:, self.mi_idx],
        }

    # ── One epoch ─────────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, Union[float, str]]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        accum = MetricsAccumulator(
            arrhythmia_labels=self.arrhythmia_label_names,
            mi_labels=self.mi_label_names,
            hrv_enabled=self.hrv_enabled,
        )
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            pbar = tqdm(loader, desc="Train" if train else "Val  ", leave=False)
            for batch in pbar:
                signal    = batch["signal"].to(self.device).float()      # (B, 12, T)
                labels    = batch["labels"].to(self.device).float()      # (B, K)
                hrv       = batch["hrv"].to(self.device).float()         # (B, 3)
                hrv_valid = batch["hrv_valid"].to(self.device)   # (B,) bool

                # --- MixUp Augmentation ---
                aug_mixup = self.cfg.get("augmentation", {}).get("aug_mixup", False)
                mixup_alpha = float(self.cfg.get("augmentation", {}).get("mixup_alpha", 0.2))
                
                if train and aug_mixup and mixup_alpha > 0:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    lam = max(lam, 1 - lam)  # Keep majority original
                    index = torch.randperm(signal.size(0)).to(self.device)
                    
                    signal = lam * signal + (1 - lam) * signal[index]
                    labels = lam * labels + (1 - lam) * labels[index]
                    hrv    = lam * hrv + (1 - lam) * hrv[index]

                preds = self.model(signal)
                targets = self._split_labels(labels)
                targets["hrv"]       = hrv
                targets["hrv_valid"] = hrv_valid

                loss_dict = self.loss_fn(preds, targets)

                if train and self.update_weights:
                    self.optimizer.zero_grad()
                    loss_dict["total"].backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    # OneCycleLR requires step() every batch (not every epoch)
                    if self.scheduler is not None and getattr(self, "_step_scheduler_per_batch", False):
                        self.scheduler.step()

                # Binarize labels for metric calculation (sklearn AUROC requires binary targets)
                metric_targets = targets
                if train and aug_mixup and mixup_alpha > 0:
                    metric_targets = {
                        "arrhythmia": (targets["arrhythmia"] >= 0.5).float(),
                        "mi": (targets["mi"] >= 0.5).float(),
                        "hrv": targets["hrv"],
                        "hrv_valid": targets["hrv_valid"],
                    }

                accum.update(preds, metric_targets, loss_dict, self.threshold)
                
                # Update progress bar
                loss_val = loss_dict["total"].item()
                pbar.set_postfix(loss=f"{loss_val:.4f}")

        return accum.compute(self.threshold, self.hrv_feature_names)

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(self):
        mode_label = "[RANDOM WEIGHTS — no update]" if not self.update_weights else ""
        print(f"\n{'='*60}")
        print(f"  Starting training: {self.epochs} epochs  {mode_label}")
        print(f"  Device: {self.device}")
        print(f"  Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics   = self._run_epoch(self.val_loader,   train=False)

            # Epoch-level scheduler step (skip for OneCycleLR — already stepped per batch)
            if self.scheduler is not None and not getattr(self, "_step_scheduler_per_batch", False):
                self.scheduler.step()

            elapsed = time.time() - t0
            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)

            # Summary
            train_loss = train_metrics.get("loss/total", float("nan"))
            val_loss   = val_metrics.get("loss/total",   float("nan"))
            val_auroc_arrhy = val_metrics.get("auroc/arrhy/macro", float("nan"))
            val_auroc_mi    = val_metrics.get("auroc/mi/macro",    float("nan"))

            print(
                f"Epoch [{epoch:3d}/{self.epochs}] "
                f"| train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} "
                f"| arrhy_auroc={val_auroc_arrhy:.4f} "
                f"| mi_auroc={val_auroc_mi:.4f} "
                f"| {elapsed:.1f}s"
            )

            # Best model checkpoint
            val_auroc = float(np.nanmean([val_auroc_arrhy, val_auroc_mi]))

            # ── Optuna Pruning Hook ──────────────────────────────────────────
            if self.optuna_trial is not None:
                import optuna
                self.optuna_trial.report(val_auroc, epoch)
                if self.optuna_trial.should_prune():
                    print(f"  [Optuna Pruned] Trial cut short at epoch {epoch}")
                    raise optuna.TrialPruned()

            if val_auroc > self.best_val_auroc:
                self.best_val_auroc = val_auroc
                self.no_improve_count = 0
                if self.checkpoint_dir:
                    save_checkpoint(
                        self.model, self.optimizer, epoch, val_metrics,
                        os.path.join(self.checkpoint_dir, "best_model.pth"),
                    )
                    print(f"  ✓ New best AUROC: {val_auroc:.4f} → saved best_model.pth")
                else:
                    print(f"  ✓ New best AUROC: {val_auroc:.4f}")
            else:
                self.no_improve_count += 1

            # Periodic checkpoint
            if self.checkpoint_dir and epoch % self.save_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics,
                    os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}.pth"),
                )

            # Early stopping
            if self.no_improve_count >= self.patience:
                print(f"\n⚠ Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break

        # Save training history
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            with open(os.path.join(self.checkpoint_dir, "history.json"), "w") as f:
                json.dump(self.history, f, indent=2)
            print(f"\n{'='*60}")
            print(f"  Training complete. Best val AUROC: {self.best_val_auroc:.4f}")
            print(f"  History saved to {self.checkpoint_dir}/history.json")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"  Training complete. Best val AUROC: {self.best_val_auroc:.4f}")
            print(f"{'='*60}\n")

    def find_thresholds(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Run inference on loader (defaults to val_loader), collect scores,
        and find per-class threshold that maximises F1 for each label.

        Returns:
            dict  label_name → optimal threshold, e.g. {'NORM': 0.45, 'AFIB': 0.25, ...}
        """
        loader = loader or self.val_loader
        self.model.eval()

        a_scores, a_trues = [], []
        mi_scores, mi_trues = [], []

        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(self.device)
                labels = batch["labels"].to(self.device)
                preds  = self.model(signal)
                targets = self._split_labels(labels)

                import torch.nn.functional as F
                a_scores.append(F.sigmoid(preds["arrhythmia"]).cpu().numpy())
                a_trues.append(targets["arrhythmia"].cpu().numpy())
                mi_scores.append(F.sigmoid(preds["mi"]).cpu().numpy())
                mi_trues.append(targets["mi"].cpu().numpy())

        a_score = np.concatenate(a_scores)
        a_true  = np.concatenate(a_trues)
        mi_score = np.concatenate(mi_scores)
        mi_true  = np.concatenate(mi_trues)

        arrhy_t = find_optimal_thresholds(a_true,  a_score,  self.arrhythmia_label_names)
        # MI: standard F1 optimization (beta=1.0) gives balanced precision/recall
        # F_beta=0.5 was overcorrecting, driving recall too low
        mi_t    = find_optimal_thresholds(mi_true, mi_score, self.mi_label_names)

        thresholds = {**arrhy_t, **mi_t}

        print("\n  Optimal thresholds (val F1-max):")
        for name, t in thresholds.items():
            print(f"    {name}: {t:.2f}")

        return thresholds

    def evaluate(self, loader: Optional[DataLoader] = None, thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Union[float, str]]:
        """Run evaluation on a given loader (defaults to val_loader).

        Args:
            loader:     DataLoader to evaluate on.
            thresholds: Per-class thresholds from find_thresholds().
                        If None, uses self.threshold (0.5) for all classes.
        """
        loader = loader or self.val_loader
        if thresholds is not None:
            metrics = self._run_epoch_with_thresholds(loader, thresholds)
        else:
            metrics = self._run_epoch(loader, train=False)
        print_metrics(metrics)
        return metrics

    def _run_epoch_with_thresholds(
        self,
        loader: DataLoader,
        thresholds: Dict[str, float],
    ) -> Dict[str, Union[float, str]]:
        """Like _run_epoch but uses per-class thresholds instead of a single scalar."""
        from src.utils.metrics import (
            compute_classification_metrics, compute_hrv_metrics, apply_thresholds
        )
        import torch.nn.functional as F

        self.model.eval()
        a_scores, a_trues = [], []
        mi_scores, mi_trues = [], []
        hrv_preds, hrv_trues = [], []
        losses = {"total": [], "arrhythmia": [], "mi": [], "hrv": []}

        with torch.no_grad():
            for batch in loader:
                signal    = batch["signal"].to(self.device)
                labels    = batch["labels"].to(self.device)
                hrv       = batch["hrv"].to(self.device)
                hrv_valid = batch["hrv_valid"].to(self.device)
                preds   = self.model(signal)
                targets = self._split_labels(labels)
                targets["hrv"]       = hrv
                targets["hrv_valid"] = hrv_valid

                loss_dict = self.loss_fn(preds, targets)
                for k in losses:
                    if k in loss_dict:
                        losses[k].append(float(loss_dict[k].item()))

                a_scores.append(F.sigmoid(preds["arrhythmia"]).cpu().numpy())
                a_trues.append(targets["arrhythmia"].cpu().numpy())
                mi_scores.append(F.sigmoid(preds["mi"]).cpu().numpy())
                mi_trues.append(targets["mi"].cpu().numpy())
                if self.hrv_enabled and "hrv" in preds:
                    hrv_preds.append(preds["hrv"].cpu().numpy())
                    hrv_trues.append(hrv.cpu().numpy())

        metrics: Dict[str, Union[float, str]] = {}
        for k, v in losses.items():
            if v:
                metrics[f"loss/{k}"] = float(np.mean(v))

        a_score = np.concatenate(a_scores)
        a_true  = np.concatenate(a_trues)
        a_pred  = apply_thresholds(a_score, self.arrhythmia_label_names, thresholds)
        metrics.update(compute_classification_metrics(a_true, a_score, a_pred, self.arrhythmia_label_names, prefix="arrhy/"))

        mi_score = np.concatenate(mi_scores)
        mi_true  = np.concatenate(mi_trues)
        mi_pred  = apply_thresholds(mi_score, self.mi_label_names, thresholds)
        metrics.update(compute_classification_metrics(mi_true, mi_score, mi_pred, self.mi_label_names, prefix="mi/"))

        if self.hrv_enabled and hrv_preds:
            metrics.update(compute_hrv_metrics(
                np.concatenate(hrv_trues), np.concatenate(hrv_preds), self.hrv_feature_names
            ))

        return metrics
