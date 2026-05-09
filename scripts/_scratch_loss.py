import json, os, yaml
import numpy as np

runs = [
    'run_20260508_214025_hybrid-tf-focal-aug',
    'run_20260509_004809_hybrid-tf-focal-aug',
    'run_20260509_033053_hybrid-tf-focal-aug',
]
keys = ['auroc/mi/IMI','auprc/mi/IMI','f1/mi/IMI','auroc/arrhy/macro','f1/arrhy/macro']

results = {k: [] for k in keys}

for run in runs:
    print(f'\n=== {run} ===')
    cfg_path = os.path.join('checkpoints', run, 'config_snapshot.yaml')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            t = cfg.get('training', {})
            print(f"  seed: {t.get('seed', 'N/A')}")
            
    met_path = os.path.join('checkpoints', run, 'test_metrics.json')
    if os.path.exists(met_path):
        with open(met_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
            for k in keys:
                val = d.get(k, None)
                if val is not None:
                    print(f'  {k}: {val}')
                    results[k].append(val)
                else:
                    print(f'  {k}: N/A')
    else:
        print('  Metrics NOT FOUND')

print('\n=== AGGREGATE RESULTS ===')
for k in keys:
    vals = results[k]
    if len(vals) == 3:
        mean = np.mean(vals)
        std = np.std(vals)
        print(f'{k}: {mean:.4f} ± {std:.4f}')
    else:
        print(f'{k}: Not enough data')
