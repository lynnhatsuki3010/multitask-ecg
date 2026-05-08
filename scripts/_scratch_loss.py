import json, os, yaml

runs = [
    'run_20260508_005129_hybrid-tf',
    'run_20260508_015300_hybrid-tf-focal',
    'run_20260508_025238_hybrid-tf-focal',
]
keys = ['auroc/mi/IMI','auprc/mi/IMI','f1/mi/IMI','auroc/arrhy/macro','f1/arrhy/macro']

for run in runs:
    print(f'\n=== {run} ===')
    cfg_path = os.path.join('checkpoints', run, 'config_snapshot.yaml')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            t = cfg.get('training', {})
            pw = t.get('use_pos_weight', False)
            foc = t.get('use_focal', False)
            grad = cfg.get('model', {}).get('mi_gradient_scale', 1.0)
            print(f'  pos_weight: {pw}, focal: {foc}, grad_scale: {grad}')
            
    met_path = os.path.join('checkpoints', run, 'test_metrics.json')
    if os.path.exists(met_path):
        with open(met_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
            for k in keys:
                val = d.get(k, "N/A")
                print(f'  {k}: {val}')
    else:
        print('  Metrics NOT FOUND')
