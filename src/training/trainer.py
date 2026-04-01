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
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from typing import Dict, Optional, Union
from tqdm import tqdm

from src.utils.metrics import MetricsAccumulator, ARRHYTHMIA_LABELS, MI_LABELS


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
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
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

        # Optimizer — plain Adam baseline (no weight decay)
        # NOTE: switch to AdamW + weight_decay when enabling that trick
        self.optimizer = Adam(
            model.parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
        )

        # Scheduler
        sched_type = train_cfg.get("scheduler", "cosine")
        if sched_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        elif sched_type == "step":
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        else:
            self.scheduler = None

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
                signal = batch["signal"].to(self.device)   # (B, 12, T)
                labels = batch["labels"].to(self.device)   # (B, K)
                hrv    = batch["hrv"].to(self.device)      # (B, 3)

                preds = self.model(signal)
                targets = self._split_labels(labels)
                targets["hrv"] = hrv

                loss_dict = self.loss_fn(preds, targets)

                if train and self.update_weights:
                    self.optimizer.zero_grad()
                    loss_dict["total"].backward()
                    # NOTE: grad clipping intentionally disabled in baseline
                    # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                accum.update(preds, targets, loss_dict, self.threshold)
                
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

            if self.scheduler is not None:
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
            if val_auroc > self.best_val_auroc:
                self.best_val_auroc = val_auroc
                self.no_improve_count = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics,
                    os.path.join(self.checkpoint_dir, "best_model.pth"),
                )
                print(f"  ✓ New best AUROC: {val_auroc:.4f} → saved best_model.pth")
            else:
                self.no_improve_count += 1

            # Periodic checkpoint
            if epoch % self.save_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics,
                    os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}.pth"),
                )

            # Early stopping
            if self.no_improve_count >= self.patience:
                print(f"\n⚠ Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break

        # Save training history
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        with open(os.path.join(self.checkpoint_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Training complete. Best val AUROC: {self.best_val_auroc:.4f}")
        print(f"  History saved to {self.checkpoint_dir}/history.json")
        print(f"{'='*60}\n")

    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, Union[float, str]]:
        """Run evaluation on a given loader (defaults to val_loader)."""
        loader = loader or self.val_loader
        metrics = self._run_epoch(loader, train=False)
        print_metrics(metrics)
        return metrics
