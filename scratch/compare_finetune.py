import json

def get_best(path, monitor='auprc/mi/IMI'):
    h = json.load(open(path))
    val = h['val']
    best = max(val, key=lambda x: x.get(monitor, 0))
    return best

g = get_best('checkpoints/finetune_georgia/history.json')
p = get_best('checkpoints/finetune_ptb/history.json')

def row(label, key_std, key_tuned, g_data, p_data):
    gv = g_data.get(key_tuned, g_data.get(key_std, float('nan')))
    pv = p_data.get(key_tuned, p_data.get(key_std, float('nan')))
    gs = f'{gv:.4f}' if isinstance(gv, float) and not str(gv) == 'nan' else 'N/A'
    ps = f'{pv:.4f}' if isinstance(pv, float) and not str(pv) == 'nan' else 'N/A'
    print(f'{label:<32} | {gs:>10} | {ps:>10}')

print('=== ARRHYTHMIA HEAD (Macro, Tuned Thresholds) ===')
print(f'{"Metric":<32} | {"Georgia":>10} | {"PTB":>10}')
print('-' * 58)
row('AUROC macro',   'auroc/arrhy/macro',  'tuned/auroc/arrhy/macro',  g, p)
row('AUPRC macro',  'auprc/arrhy/macro',  'tuned/auprc/arrhy/macro',  g, p)
row('F1 macro',     'f1/arrhy/macro',     'tuned/f1/arrhy/macro',     g, p)
row('Precision',    'prec/arrhy/macro',   'tuned/prec/arrhy/macro',   g, p)
row('Recall',       'rec/arrhy/macro',    'tuned/rec/arrhy/macro',    g, p)

print()
print('=== MI HEAD (Macro, Tuned Thresholds) ===')
print(f'{"Metric":<32} | {"Georgia":>10} | {"PTB":>10}')
print('-' * 58)
row('AUROC macro',   'auroc/mi/macro',  'tuned/auroc/mi/macro',  g, p)
row('AUPRC macro',  'auprc/mi/macro',  'tuned/auprc/mi/macro',  g, p)
row('F1 macro',     'f1/mi/macro',     'tuned/f1/mi/macro',     g, p)
row('Precision',    'prec/mi/macro',   'tuned/prec/mi/macro',   g, p)
row('Recall',       'rec/mi/macro',    'tuned/rec/mi/macro',    g, p)

print()
print('=== PER-CLASS MI ===')
print(f'{"Metric":<32} | {"Georgia":>10} | {"PTB":>10}')
print('-' * 58)
for cls in ['IMI', 'ASMI']:
    for metric in ['auroc', 'auprc', 'f1']:
        key_t = f'tuned/{metric}/mi/{cls}'
        key_s = f'{metric}/mi/{cls}'
        row(f'{cls} {metric.upper()}', key_s, key_t, g, p)
