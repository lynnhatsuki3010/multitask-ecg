import json
with open('checkpoints/run_20260504_212532_hybrid-tf-focal/history.json') as f:
    h = json.load(f)

print('Epoch | IMI F1 | IMI Prec | Arrhy F1')
for i, v in enumerate(h['val']):
    f1 = v.get('tuned/f1/mi/IMI', v.get('f1/mi/IMI', 0))
    prec = v.get('prec/mi/IMI', 0)
    arrhy_f1 = v.get('f1/arrhy/macro', 0)
    print(f"{i+1:5d} | {f1:.4f} | {prec:.4f} | {arrhy_f1:.4f}")
