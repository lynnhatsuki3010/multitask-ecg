import os
import json
import glob
import pandas as pd
import yaml

# Target date range: 14th to 19th
runs = sorted(glob.glob("checkpoints/run_2026041[4-9]*"))

results = []
for run_dir in runs:
    metrics_path = os.path.join(run_dir, "test_metrics.json")
    config_path = os.path.join(run_dir, "config_snapshot.yaml")
    if not os.path.exists(metrics_path) or not os.path.exists(config_path):
        continue
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    name = os.path.basename(run_dir)
    
    # Extract config details
    lr = cfg.get("training", {}).get("lr", 0)
    wd = cfg.get("training", {}).get("weight_decay", 0)
    bs = cfg.get("training", {}).get("batch_size", 0)
    
    m_cfg = cfg.get("model", {})
    d_model = m_cfg.get("d_model", 0)
    patch = m_cfg.get("patch_size", 0)
    dropout = m_cfg.get("dropout", 0)
    layers = m_cfg.get("num_encoder_layers", 0)
    nhead = m_cfg.get("nhead", 0)
    
    # Extract metrics
    res = {
        "Run": name.replace("run_202604", ""),
        "BS": bs,
        "LR": f"{lr:.0e}",
        "WD": f"{wd:.0e}",
        "Patch": patch,
        "DModel": d_model,
        "Lyrs": layers,
        "Drop": dropout,
        "Arr(Mac)": metrics.get("f1/arrhy/macro", 0),
        "MI(Mac)": metrics.get("f1/mi/macro", 0),
    }
    results.append(res)

df = pd.DataFrame(results)

# Pick only a few milestones to avoid clutter:
# 1. The first one (Baseline on 16)
# 2. First focal
# 3. First focal-aug
# 4. Focal-wrs-aug (18th - with MILeadExtractor)
# 5. The latest one on 19th (500Hz)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x if isinstance(x, float) else x)
print(df)
