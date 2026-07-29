# Demo Inference B10-05

Thư mục này chứa notebook phục vụ quay video demo ngắn cho checkpoint B10-05 `cond_depth`.

## Nội Dung Demo

Notebook `demo_b10_05_inference.ipynb` đi theo 5 phần:

1. Giới thiệu nhanh dataset PTB-XL, ECG 12 đạo trình, sampling rate 500 Hz và mục tiêu phân loại đa nhãn.
2. Load dữ liệu test bằng cùng pipeline preprocessing với quá trình đánh giá và hiển thị waveform 12 đạo trình.
3. Load model B10-05 đã huấn luyện sẵn, load calibration nếu có, chạy inference 1-3 mẫu.
4. Hiển thị bảng kết quả gồm task, label, probability, threshold, predicted và ground truth.
5. Chạy `scripts/16_xai_explainer.py` để tạo heatmap XAI chính thức và hiển thị lại ảnh trong notebook.

Notebook không train lại model.

## Checkpoint Mặc Định

Notebook mặc định dùng:

```text
checkpoints/run_20260619_193427_branche10_05_cond_depth-decoupled_multitask-aug
```

Thư mục checkpoint cần có:

```text
config_snapshot.yaml
best_model.pth
calibration_results.json
```

Nếu chưa có `calibration_results.json`, chạy từ repo root:

```bash
python scripts/03_calibrate.py --dir checkpoints/run_20260619_193427_branche10_05_cond_depth-decoupled_multitask-aug
```

Lệnh này chỉ hiệu chỉnh hậu kỳ trên checkpoint đã train, không huấn luyện lại model.

## Cách Chạy Notebook

Từ repo root:

```bash
jupyter notebook notebooks/demo_inference_b10_05/demo_b10_05_inference.ipynb
```

Hoặc mở trực tiếp file notebook trong Cursor/VSCode rồi chạy lần lượt từng cell.

Nếu chỉ muốn demo một mẫu, sửa trong cell đầu:

```python
N_DEMO = 1
```

Mặc định notebook dùng:

```python
N_DEMO = 3
```

## Gợi Ý Lời Thoại Video

Phần inference có thể nói ngắn:

> Em load checkpoint B10-05 đã huấn luyện sẵn, không train lại. Mỗi mẫu ECG sau preprocessing có kích thước 12 đạo trình x 5000 điểm mẫu. Mô hình trả về logits cho ba nhóm tác vụ: arrhythmia, myocardial infarction và conduction. Logits được hiệu chỉnh bằng temperature nếu có, chuyển thành xác suất bằng sigmoid, sau đó áp dụng ngưỡng riêng từng lớp để tạo nhãn dự đoán.

## Outputs

Waveform được lưu vào:

```text
notebooks/demo_inference_b10_05/outputs/
```

Ảnh XAI được lưu vào:

```text
notebooks/demo_inference_b10_05/outputs/xai/
```

Cell XAI trong notebook gọi script chính thức với cấu hình nhanh:

```bash
python scripts/16_xai_explainer.py \
  --checkpoint checkpoints/run_20260619_193427_branche10_05_cond_depth-decoupled_multitask-aug \
  --split test \
  --max-per-class 1 \
  --max-total 3 \
  --min-classes 1 \
  --method gxi \
  --grid-mode none \
  --out notebooks/demo_inference_b10_05/outputs/xai \
  --clear
```
