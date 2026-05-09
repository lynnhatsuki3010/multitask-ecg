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
        eval_cfg = cfg.get("eval", {})
        self.val_tune_thresholds_every = int(eval_cfg.get("val_tune_thresholds_every", 0))

        train_cfg = cfg.get("training", {})
        self.update_weights    = update_weights
        self.epochs    = train_cfg.get("epochs", 50)
        self.threshold = cfg.get("eval", {}).get("threshold", 0.5)
        self.save_every = train_cfg.get("save_every", 5)
        self.patience   = train_cfg.get("early_stopping_patience", 10)
        self.monitor_metric = train_cfg.get("monitor_metric", "macro_f1")
        self.checkpoint_dir = cfg.get("paths", {}).get("checkpoints", "checkpoints")
        self.freeze_epoch = train_cfg.get("freeze_backbone_at_epoch", -1)

        # ── Optimizer — AdamW with differential weight decay + MI head LR ─────────
        # Embedding / LayerNorm / bias params should NOT be weight-decayed:
        # regularizing them causes representation collapse on small datasets.
        # MI head gets higher LR to compensate for gradient competition.
        lr           = float(train_cfg.get("lr", 1e-4))
        weight_decay = float(train_cfg.get("weight_decay", 5e-3))
        mi_lr_mult   = float(train_cfg.get("mi_lr_multiplier", 1.0))

        no_decay_names = ("bias", "norm", "embed", "cls_token", "pos_embed")
        mi_param_names = ("mi_head", "inferior", "reciprocal", "anterior",
                          "imi_token_pool", "asmi_token_pool")

        mi_decay, mi_nodecay = [], []
        other_decay, other_nodecay = [], []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_mi = any(mp in name for mp in mi_param_names)
            is_nodecay = any(nd in name.lower() for nd in no_decay_names)

            if is_mi:
                if is_nodecay:
                    mi_nodecay.append(param)
                else:
                    mi_decay.append(param)
            else:
                if is_nodecay:
                    other_nodecay.append(param)
                else:
                    other_decay.append(param)

        mi_lr = lr * mi_lr_mult
        param_groups = [
            {"params": other_decay,   "weight_decay": weight_decay, "lr": lr},
            {"params": other_nodecay, "weight_decay": 0.0,          "lr": lr},
            {"params": mi_decay,      "weight_decay": weight_decay, "lr": mi_lr},
            {"params": mi_nodecay,    "weight_decay": 0.0,          "lr": mi_lr},
        ]
        # Remove empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]
        self.optimizer = AdamW(param_groups)

        n_mi = len(mi_decay) + len(mi_nodecay)
        n_other = len(other_decay) + len(other_nodecay)
        if mi_lr_mult != 1.0:
            print(f"  Differential LR: backbone+arrhy={lr:.1e} ({n_other} params), "
                  f"MI head={mi_lr:.1e} ({n_mi} params, {mi_lr_mult}x)")

        # Store param refs for gradient norm logging
        self._mi_params = mi_decay + mi_nodecay
        self._other_params = other_decay + other_nodecay

        # ── Scheduler ──────────────────────────────────────────────────────
        sched_type    = train_cfg.get("scheduler", "cosine")
        warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
        steps_per_epoch = len(train_loader)  # needed for OneCycleLR

        if sched_type == "one_cycle":
            # OneCycleLR: warmup → peak → smooth cosine decay.
            # Supports differential LR via per-group max_lr list.
            pct_start = train_cfg.get("one_cycle_pct_start", 0.2)  # 20% warmup
            max_lrs = [g["lr"] for g in self.optimizer.param_groups]
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,
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
        self.best_monitor_value = -1.0
        self.no_improve_count = 0

        _tuned_metrics = ("imi_f1_tuned", "macro_f1_tuned", "mi_macro_f1_tuned")
        if self.monitor_metric in _tuned_metrics and self.val_tune_thresholds_every <= 0:
            raise ValueError(
                f"training.monitor_metric='{self.monitor_metric}' requires "
                "eval.val_tune_thresholds_every > 0 (per-class threshold search on val each N epochs)."
            )

    MONITOR_ALIASES = {
        "mean_auroc": ["auroc/arrhy/macro", "auroc/mi/macro"],
        "macro_f1": ["f1/arrhy/macro", "f1/mi/macro"],
        "arrhythmia_f1": ["f1/arrhy/macro"],
        "mi_f1": ["f1/mi/macro"],
        "imi_f1": ["f1/mi/IMI"],
        "imi_auroc": ["auroc/mi/IMI"],
        "imi_auprc": ["auprc/mi/IMI"],
        "asmi_f1": ["f1/mi/ASMI"],
        "asmi_auroc": ["auroc/mi/ASMI"],
        "asmi_auprc": ["auprc/mi/ASMI"],
        # Val metrics after per-class threshold search (requires eval.val_tune_thresholds_every > 0)
        "imi_f1_tuned": ["tuned/f1/mi/IMI"],
        "macro_f1_tuned": ["tuned/f1/arrhy/macro", "tuned/f1/mi/macro"],
        "mi_macro_f1_tuned": ["tuned/f1/mi/macro"],
    }

    def _compute_monitor_value(self, metrics: Dict[str, Union[float, str]]) -> float:
        metric_keys = self.MONITOR_ALIASES.get(self.monitor_metric, None)
        if metric_keys is None:
            metric_keys = [self.monitor_metric]
        values = [metrics.get(key, float("nan")) for key in metric_keys]
        return float(np.nanmean(values))

    def _monitor_label(self) -> str:
        friendly = {
            "mean_auroc": "mean AUROC",
            "macro_f1": "mean macro F1",
            "arrhythmia_f1": "arrhythmia macro F1",
            "mi_f1": "MI macro F1",
            "imi_f1": "IMI F1",
            "imi_auroc": "IMI AUROC",
            "imi_auprc": "IMI AUPRC",
            "asmi_f1": "ASMI F1",
            "asmi_auroc": "ASMI AUROC",
            "asmi_auprc": "ASMI AUPRC",
            "imi_f1_tuned": "IMI F1 (val tuned thresholds)",
            "macro_f1_tuned": "mean macro F1 (val tuned thresholds)",
            "mi_macro_f1_tuned": "MI macro F1 (val tuned thresholds)",
        }
        return friendly.get(self.monitor_metric, self.monitor_metric)

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

                # --- MixUp / CutMix Augmentation ---
                aug_mixup = self.cfg.get("augmentation", {}).get("aug_mixup", False)
                mixup_alpha = float(self.cfg.get("augmentation", {}).get("mixup_alpha", 0.2))
                aug_cutmix = self.cfg.get("augmentation", {}).get("aug_cutmix", False)
                cutmix_alpha = float(self.cfg.get("augmentation", {}).get("cutmix_alpha", 0.2))
                
                if train and aug_mixup and mixup_alpha > 0:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    lam = max(lam, 1 - lam)  # Keep majority original
                    index = torch.randperm(signal.size(0)).to(self.device)
                    
                    signal = lam * signal + (1 - lam) * signal[index]
                    labels = lam * labels + (1 - lam) * labels[index]
                    hrv    = lam * hrv + (1 - lam) * hrv[index]
                elif train and aug_cutmix and cutmix_alpha > 0:
                    lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                    B, C, T = signal.size()
                    index = torch.randperm(B).to(self.device)
                    
                    # Compute cut window
                    cut_ratio = np.sqrt(1. - lam)
                    cut_len = int(T * cut_ratio)
                    
                    cx = np.random.randint(T)
                    bbx1 = np.clip(cx - cut_len // 2, 0, T)
                    bbx2 = np.clip(cx + cut_len // 2, 0, T)
                    
                    # Update lambda exactly based on actual cut length
                    lam = 1 - ((bbx2 - bbx1) / T)
                    
                    signal_clone = signal.clone()
                    signal[:, :, bbx1:bbx2] = signal_clone[index, :, bbx1:bbx2]
                    
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

                    # Accumulate gradient norms for monitoring task balance
                    if hasattr(self, '_grad_norms_mi'):
                        mi_gn = sum(
                            p.grad.norm().item() ** 2
                            for p in self._mi_params
                            if p.grad is not None
                        ) ** 0.5
                        other_gn = sum(
                            p.grad.norm().item() ** 2
                            for p in self._other_params
                            if p.grad is not None
                        ) ** 0.5
                        self._grad_norms_mi.append(mi_gn)
                        self._grad_norms_other.append(other_gn)

                    self.optimizer.step()
                    # OneCycleLR requires step() every batch (not every epoch)
                    if self.scheduler is not None and getattr(self, "_step_scheduler_per_batch", False):
                        self.scheduler.step()

                # Binarize labels for metric calculation (sklearn AUROC requires binary targets)
                metric_targets = targets
                if train and ((aug_mixup and mixup_alpha > 0) or (aug_cutmix and cutmix_alpha > 0)):
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
        print(f"  Monitoring: {self._monitor_label()}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            if epoch == self.freeze_epoch:
                print(f"\n{'-'*60}")
                print(f"  [Phase 2 Initiated] Freezing Backbone & Arrhythmia Head")
                print(f"  Gradient isolation disabled. Fine-tuning MI Head only.")
                print(f"  Resetting Early Stopping patience.")
                print(f"{'-'*60}\n")
                if hasattr(self.model, "freeze_for_phase2"):
                    self.model.freeze_for_phase2()
                # Reset early stopping so Phase 2 has time to learn
                self.no_improve_count = 0

            t0 = time.time()

            # Init gradient norm accumulators for this epoch
            self._grad_norms_mi = []
            self._grad_norms_other = []

            train_metrics = self._run_epoch(self.train_loader, train=True)

            # Compute and store average gradient norms
            avg_mi_gn = float(np.mean(self._grad_norms_mi)) if self._grad_norms_mi else 0.0
            avg_other_gn = float(np.mean(self._grad_norms_other)) if self._grad_norms_other else 0.0
            train_metrics["grad_norm/mi"] = avg_mi_gn
            train_metrics["grad_norm/backbone"] = avg_other_gn

            val_metrics   = self._run_epoch(self.val_loader,   train=False)

            # Optional: val metrics with per-class thresholds (aligns with post-train test protocol)
            tune_every = self.val_tune_thresholds_every
            if tune_every > 0 and epoch % tune_every == 0:
                thr = self.find_thresholds(self.val_loader, verbose=False)
                tuned = self._run_epoch_with_thresholds(self.val_loader, thr)
                for k, v in tuned.items():
                    if isinstance(v, float):
                        val_metrics[f"tuned/{k}"] = v

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
            val_f1_arrhy    = val_metrics.get("f1/arrhy/macro",    float("nan"))
            val_f1_mi       = val_metrics.get("f1/mi/macro",       float("nan"))
            val_f1_imi      = val_metrics.get("f1/mi/IMI",         float("nan"))
            val_auprc_imi   = val_metrics.get("auprc/mi/IMI",      float("nan"))
            val_f1_imi_t    = val_metrics.get("tuned/f1/mi/IMI",    float("nan"))

            extra = ""
            if tune_every > 0 and epoch % tune_every == 0 and not np.isnan(val_f1_imi_t):
                extra = f" | imi_f1_tuned={val_f1_imi_t:.4f}"

            # Gradient norm info
            gn_mi = train_metrics.get("grad_norm/mi", 0.0)
            gn_bb = train_metrics.get("grad_norm/backbone", 0.0)
            gn_ratio = gn_mi / max(gn_bb, 1e-8)
            grad_info = f" | gn_mi/bb={gn_ratio:.2f}"

            print(
                f"Epoch [{epoch:3d}/{self.epochs}] "
                f"| train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} "
                f"| arrhy_f1={val_f1_arrhy:.4f} "
                f"| mi_f1={val_f1_mi:.4f} "
                f"| imi_f1={val_f1_imi:.4f} "
                f"| imi_auprc={val_auprc_imi:.4f} "
                f"| arrhy_auroc={val_auroc_arrhy:.4f} "
                f"| mi_auroc={val_auroc_mi:.4f} "
                f"{extra}{grad_info}"
                f"| {elapsed:.1f}s"
            )

            # Best model checkpoint
            monitor_value = self._compute_monitor_value(val_metrics)
            monitor_label = self._monitor_label()

            # ── Optuna Pruning Hook ──────────────────────────────────────────
            if self.optuna_trial is not None:
                import optuna
                self.optuna_trial.report(monitor_value, epoch)
                if self.optuna_trial.should_prune():
                    print(f"  [Optuna Pruned] Trial cut short at epoch {epoch}")
                    raise optuna.TrialPruned()

            if monitor_value > self.best_monitor_value:
                self.best_monitor_value = monitor_value
                self.no_improve_count = 0
                if self.checkpoint_dir:
                    save_checkpoint(
                        self.model, self.optimizer, epoch, val_metrics,
                        os.path.join(self.checkpoint_dir, "best_model.pth"),
                    )
                    print(f"  ✓ New best {monitor_label}: {monitor_value:.4f} → saved best_model.pth")
                else:
                    print(f"  ✓ New best {monitor_label}: {monitor_value:.4f}")
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
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement in {self._monitor_label()} for {self.patience} epochs)"
                )
                break

        # Save training history
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            with open(os.path.join(self.checkpoint_dir, "history.json"), "w") as f:
                json.dump(self.history, f, indent=2)
            print(f"\n{'='*60}")
            print(f"  Training complete. Best {self._monitor_label()}: {self.best_monitor_value:.4f}")
            print(f"  History saved to {self.checkpoint_dir}/history.json")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"  Training complete. Best {self._monitor_label()}: {self.best_monitor_value:.4f}")
            print(f"{'='*60}\n")

    def find_thresholds(
        self,
        loader: Optional[DataLoader] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Run inference on loader (defaults to val_loader), collect scores,
        and find per-class threshold that maximises F_beta for each label
        (see eval.threshold_search.class_beta).

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

        threshold_cfg = self.cfg.get("eval", {}).get("threshold_search", {})
        global_floor  = float(threshold_cfg.get("min_threshold", 0.1))
        rare_cap      = float(threshold_cfg.get("max_rare_threshold", 0.4))
        min_pos_count = int(threshold_cfg.get("min_pos_count", 20))
        class_beta    = threshold_cfg.get("class_beta", {})   # e.g. {SBRAD: 2.0, IMI: 1.5}
        arrhy_t = find_optimal_thresholds(
            a_true,
            a_score,
            self.arrhythmia_label_names,
            min_pos_count=min_pos_count,
            max_rare_threshold=rare_cap,
            min_threshold=global_floor,
            class_beta=class_beta,
            class_min_threshold=threshold_cfg.get("arrhythmia_min_threshold", {}),
            class_min_precision=threshold_cfg.get("arrhythmia_min_precision", {}),
            class_max_threshold=threshold_cfg.get("arrhythmia_max_threshold", {}),
        )
        mi_t = find_optimal_thresholds(
            mi_true,
            mi_score,
            self.mi_label_names,
            min_pos_count=min_pos_count,
            max_rare_threshold=rare_cap,
            min_threshold=global_floor,
            class_beta=class_beta,
            class_min_threshold=threshold_cfg.get("mi_min_threshold", {}),
            class_min_precision=threshold_cfg.get("mi_min_precision", {}),
            class_max_threshold=threshold_cfg.get("mi_max_threshold", {}),
        )

        thresholds = {**arrhy_t, **mi_t}

        if verbose:
            print("\n  Optimal thresholds (val set, F_beta sweep):")
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
        losses = {"total": [], "arrhythmia": [], "mi": [], "imi": [], "asmi": [], "hrv": []}

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
