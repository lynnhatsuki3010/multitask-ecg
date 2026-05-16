# ECG Multi-Task Transformer

Dự án phân tích ECG 12 chuyển đạo trên PTB-XL dataset với Transformer + multi-task learning.

## Cấu trúc thư mục
```
ECG_HRV/
├── data/
│   ├── raw/PTB-XL/          ← Dataset gốc (đã có)
│   ├── processed/           ← metadata.csv, label_matrix.npy, hrv_matrix.npy
│   └── splits/              ← train/val/test split indices
├── src/
│   ├── data/
│   │   ├── label_builder.py ← Xây dựng label matrix từ scp_codes
│   │   ├── hrv_features.py  ← Tính RMSSD, SDNN, mean HR
│   │   └── preprocessing.py ← Lọc tín hiệu, Dataset class
│   ├── models/
│   │   ├── ecg_transformer.py ← Patch-based Transformer backbone
│   │   └── multitask_head.py  ← Arrhythmia + MI + HRV heads
│   ├── training/
│   │   ├── losses.py        ← Multi-task loss (BCE + MSE)
│   │   └── trainer.py       ← Training loop, early stopping, checkpoint
│   └── utils/
│       └── metrics.py       ← AUROC, AUPRC, F1, HRV MAE/RMSE
├── configs/
│   └── config.yaml          ← Toàn bộ hyperparameter
├── scripts/
│   ├── 01_build_metadata.py ← Preprocessing + tạo splits
│   └── 02_train.py          ← Training script chính
└── checkpoints/             ← Saved models (tạo tự động)
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy

### Bước 1: Preprocessing (bắt buộc chạy trước)
```bash
# Full (có HRV, ~30 phút)
python scripts/01_build_metadata.py

# Không HRV (nhanh hơn, ~5 phút)
python scripts/01_build_metadata.py --no-hrv

# Test nhanh chỉ 100 samples
python scripts/01_build_metadata.py --max-samples 100 --no-hrv
```

### Bước 2: Training
```bash
# Full training
python scripts/02_train.py

# Debug mode (200 samples, 3 epochs — để kiểm tra pipeline)
python scripts/02_train.py --debug

# Không HRV task
python scripts/02_train.py --no-hrv

# Override hyperparameters
python scripts/02_train.py --epochs 30 --batch-size 32 --lr 5e-5

# Resume từ checkpoint
python scripts/02_train.py --resume checkpoints/epoch_010.pth
```

## Nhãn (7 labels)

| Label | Task | Diễn giải |
|-------|------|-----------|
| NORM  | normal | ECG bình thường |
| AFIB  | arrhythmia | Rung nhĩ |
| STACH | arrhythmia | Nhịp nhanh xoang |
| SBRAD | arrhythmia | Nhịp chậm xoang |
| AFLT  | arrhythmia | Cuồng nhĩ |
| IMI   | mi | Nhồi máu cơ tim vùng dưới |
| ASMI  | mi | Nhồi máu cơ tim anteroseptal |

## Model Architecture

```
Input (B, 12, 1000)
  → PatchEmbedding (patch_size=25 → 40 patches × 12 leads)
  → Linear projection (300 → d_model=128)
  → [CLS] token + Positional Encoding
  → TransformerEncoder (4 layers, 4 heads, Pre-LN)
  → CLS token (B, 128)
      ├── Arrhythmia Head → (B, 5) logits
      ├── MI Head         → (B, 2) logits
      └── HRV Head        → (B, 3) [rmssd, sdnn, mean_hr]
```

## Outputs (checkpoints/)
- `best_model.pth`   — best val AUROC checkpoint
- `epoch_XXX.pth`    — periodic checkpoints
- `history.json`     — training/val metrics per epoch
- `test_metrics.json` — final test evaluation

## Giải thích Mô hình (XAI - Explainable AI)

Module XAI sử dụng attention weights từ Transformer Encoder để làm nổi bật (highlight) các vùng tín hiệu ECG mà mô hình tập trung vào khi đưa ra dự đoán các bệnh lý loạn nhịp và nhồi máu cơ tim.

```bash
# Tạo XAI heatmaps cho test set, phân loại rõ True Positives và False Positives
python scripts/16_xai_explainer.py \
    --checkpoint checkpoints/run_YYYYMMDD_HHMMSS \
    --config configs/experiments/dynamic_threshold/dynth_e03_dynamic.yaml \
    --split test \
    --max-per-class 5 \
    --max-total 50
```

Ảnh đầu ra sẽ được tự động lưu vào `artifacts/xai/` với cấu trúc thư mục rõ ràng:
- `artifacts/xai/TP/{label}/`: Các ca bệnh dự đoán **ĐÚNG** (True Positives).
- `artifacts/xai/FP/{label}/`: Các ca bệnh dự đoán **SAI** (False Positives).
