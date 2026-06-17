# BRANCHE10 — Branch Optimization Matrix (E10 Decoupled)

Sau khi tách nhánh hoàn toàn (E10), ma trận này tối ưu **từng nhánh độc lập** mà không đổi kiến trúc cốt lõi.

## Metric chuẩn để so sánh

| Nhánh | Metric chính (go/no-go) | Metric phụ (trade-off) |
|-------|-------------------------|-------------------------|
| **MI** | `f1/mi/macro`, `f1/mi/IMI`, `f1/mi/AMI` | `auprc/mi/IMI`, `auroc/mi/macro` |
| **Conduction** | `f1/cond/macro` | `auroc/cond/macro`, per-class LBBB/RBBB/1AVB |
| **Arrhythmia** | `f1/arrhy/macro` | `auroc/arrhy/macro`, tail classes PVC/AFLT |

Tất cả run dùng cùng data: `processed_dir4` / `splits_dir4`, split `strat_fold` (fold 10 test, fold 9 val).

## Ma trận thí nghiệm

| ID | Config | Track | Thay đổi so với baseline |
|----|--------|-------|---------------------------|
| 00 | `branche10_00_baseline.yaml` | baseline | E10 gốc, `monitor_metric: imi_auprc` |
| 01 | `branche10_01_mi_asl.yaml` | MI | + ASL + pos_weight |
| 02 | `branche10_02_mi_imi_weight.yaml` | MI | 01 + `loss_weights.imi: 1.75` |
| 03 | `branche10_03_mi_macro_select.yaml` | MI | 02 + `mi_macro_f1_tuned` + val threshold tune mỗi epoch |
| 04 | `branche10_04_cond_lead_dim.yaml` | Conduction | `cd_lead_out_dim: 192` |
| 05 | `branche10_05_cond_depth.yaml` | Conduction | 04 + `cond_layers: 3` |
| 06 | `branche10_06_arr_calib.yaml` | Arrhythmia | val threshold tune + ràng buộc arrhythmia |

## Cách chạy

```bash
# Baseline (nếu data đã build sẵn cho dir4 thì bỏ qua build)
python scripts/02_train.py --config configs/experiments/branche10_00_baseline.yaml

# MI track
python scripts/02_train.py --config configs/experiments/branche10_01_mi_asl.yaml
python scripts/02_train.py --config configs/experiments/branche10_02_mi_imi_weight.yaml
python scripts/02_train.py --config configs/experiments/branche10_03_mi_macro_select.yaml

# Conduction track (độc lập, từ baseline)
python scripts/02_train.py --config configs/experiments/branche10_04_cond_lead_dim.yaml
python scripts/02_train.py --config configs/experiments/branche10_05_cond_depth.yaml

# Arrhythmia track (độc lập, từ baseline)
python scripts/02_train.py --config configs/experiments/branche10_06_arr_calib.yaml
```

Run folder sẽ có prefix `branche10_XX_*` nhờ field `experiment.id` trong config.

## So sánh kết quả sau khi train

```bash
python scripts/compare_branche10_runs.py \
  --runs checkpoints/run_*_branche10_00_baseline \
           checkpoints/run_*_branche10_01_mi_asl \
           checkpoints/run_*_branche10_02_mi_imi_weight
```

Hoặc truyền thư mục cụ thể:

```bash
python scripts/compare_branche10_runs.py --runs checkpoints/run_20260616_120000_branche10_00_baseline checkpoints/run_20260616_140000_branche10_01_mi_asl
```

## Tiêu chí chọn config tốt nhất

1. **MI track thắng** nếu cải thiện `f1/mi/macro` và `f1/mi/IMI` mà không làm `f1/cond/macro` giảm > 2 điểm phần trăm tuyệt đối.
2. **Conduction track thắng** nếu `f1/cond/macro` tăng rõ, MI/Arrhy không sụt > 1–2 điểm.
3. **Arrhythmia track** chỉ áp dụng nếu arrhythmia tail (PVC/AFLT) cải thiện mà MI không đổi xấu.

## Ghi chú về task interference

E10 đã loại shared CNN → interference chủ yếu còn ở:
- loss balancing giữa task (`loss_weights`, ASL/pos_weight)
- checkpoint criterion (`monitor_metric`)
- threshold calibration trên lớp hiếm

Nếu IMI vẫn thấp sau track MI, so sánh thêm với **random split** của bạn bạn để kiểm tra protocol bias (không phải do kiến trúc).
