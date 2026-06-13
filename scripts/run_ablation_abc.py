"""
Run ablation A/B/C on processed_dir4 (13 MI labels) and calibrate each checkpoint.

  A: arch_e06_multi_branch.yaml       (soft, num_shared_layers=2)
  B: arch_e08_separated_experts.yaml  (soft, num_shared_layers=0)
  C: arch_e07_ablation_hard_graph_13L.yaml (hard_graph, num_shared_layers=0)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ("A_e06_soft_shared2", ROOT / "configs/experiments/arch_e06_multi_branch.yaml"),
    ("B_e08_soft_shared0", ROOT / "configs/experiments/arch_e08_separated_experts.yaml"),
    ("C_e07_hard_graph", ROOT / "configs/experiments/arch_e07_ablation_hard_graph_13L.yaml"),
]


def latest_run_dir() -> Path:
    ckpt = ROOT / "checkpoints"
    runs = sorted(ckpt.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise RuntimeError("No checkpoint run directories found.")
    return runs[0]


def run_train(config: Path) -> Path:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, str(ROOT / "scripts/02_train.py"), "--config", str(config)]
    print(f"\n{'='*60}\nTRAIN: {config.name}\n{'='*60}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    run_dir = latest_run_dir()
    print(f"  -> {run_dir.name}", flush=True)
    return run_dir


def run_calibrate(run_dir: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, str(ROOT / "scripts/03_calibrate.py"), "--dir", str(run_dir)]
    print(f"CALIBRATE: {run_dir.name}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def extract_mi_metrics(run_dir: Path) -> dict:
    path = run_dir / "test_metrics.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    return {
        "run": run_dir.name,
        "mi_macro_f1": m.get("f1/mi/macro"),
        "imi_f1": m.get("f1/mi/IMI"),
        "asmi_f1": m.get("f1/mi/ASMI"),
        "ilmi_f1": m.get("f1/mi/ILMI"),
        "ami_f1": m.get("f1/mi/AMI"),
        "imi_auprc": m.get("auprc/mi/IMI"),
        "cond_macro_f1": m.get("f1/cond/macro"),
    }


def main() -> None:
    results = []
    for tag, cfg in CONFIGS:
        run_dir = run_train(cfg)
        run_calibrate(run_dir)
        row = extract_mi_metrics(run_dir)
        row["ablation"] = tag
        results.append(row)

    out = ROOT / "artifacts" / "ablation_abc_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary -> {out}", flush=True)
    for r in results:
        print(r, flush=True)


if __name__ == "__main__":
    main()
