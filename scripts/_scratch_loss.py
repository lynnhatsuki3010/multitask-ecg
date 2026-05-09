import json, os, yaml

runs = [
    'run_20260509_110546_hybrid-tf',
]
keys = ['auroc/mi/IMI','auprc/mi/IMI','f1/mi/IMI','auroc/arrhy/macro','f1/arrhy/macro']

for run in runs:
    print(f'\n=== {run} ===')
    cfg_path = os.path.join('checkpoints', run, 'config_snapshot.yaml')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            t = cfg.get('training', {})
            print(f"  use_asl: {t.get('use_asl', False)}")
            
    met_path = os.path.join('checkpoints', run, 'test_metrics.json')
    if os.path.exists(met_path):
        with open(met_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
            for k in keys:
                val = d.get(k, "N/A")
                print(f'  {k}: {val}')
    else:
        print('  Metrics NOT FOUND')
