# Báo Cáo Kỹ Thuật: Tối ưu hóa Hybrid-Transformer cho Dữ liệu Điện tâm đồ Đa nhiệm

**Date**: 2026-05-15
**Status**: Đang tái cấu trúc theo Pipeline Ablation
**Branch**: `THESIS-REVISION`

---

## Cấu trúc Bài báo cáo
Báo cáo này được chia làm 3 phần chính:
1. **Phần 0: Phân tích Kiến trúc (Architectural Ablation)** - Chứng minh thiết kế cốt lõi của mạng Hybrid-Transformer và bài toán nhiễu loạn Gradient (Gradient Interference).
2. **Phần 1: Khám phá Kỹ thuật Huấn luyện (Pipeline Ablation)** - Hành trình 5 Stage đi tìm công thức tối ưu hóa (Split, Norm, Augmentation, Sampling, Loss).
3. **Phần 2: Tinh chỉnh và Thẩm định lâm sàng (Fine-tuning & Cross-Validation)** - Mở khóa giới hạn cuối cùng và kiểm chứng chéo trên Dataset mới.

---

## PHẦN 0: PHÂN TÍCH KIẾN TRÚC MÔ HÌNH (ARCHITECTURAL ABLATION)

Trước khi đi vào các kỹ thuật tối ưu hóa dữ liệu và hàm loss, ta cần chứng minh tại sao cấu trúc **Hybrid-Transformer** (tích hợp CNN, Transformer, Nhóm đạo trình - Lead Group, và Token Nhiệm vụ - Task Token) lại được lựa chọn.

Để đánh giá chính xác sức mạnh thô (raw power) của từng thành phần kiến trúc, tất cả 4 mô hình dưới đây đều được huấn luyện trên **Pipeline cơ bản nhất**:
- `Split`: Stratified Fold (Tiêu chuẩn)
- `Loss`: BCE Loss (Không Focal Loss, Không Weighted Sampler)
- `Augmentation`: Không sử dụng (Không Noise, Warp, MixUp)
- `Gradient Isolation`: Tắt (`mi_gradient_scale = 1.0`)
- `Epochs`: 15

### Bảng Kết Quả: Tác động của từng module kiến trúc

| Run ID | Config | Kiến Trúc (Architecture Components) | Arrhythmia (Macro F1) | MI (Macro F1) | IMI (AUPRC) | Checkpoint Path | Nhận Xét Cốt Lõi |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `ARCH-1` | `arch_e01` | Base CNN (ResNet1d + Average Pool) | 0.769 | 0.594 | **0.484** | `run_20260514_221003_cnn` | CNN thuần túy bắt được cục bộ khá tốt, IMI AUPRC rất cao. |
| `ARCH-2` | `arch_e02` | Base CNN + Transformer (No Task Token) | **0.781** | 0.589 | 0.397 | `run_20260514_231910_hybrid-tf` | TF giúp học ngữ cảnh toàn cục -> Arrhythmia tăng mạnh, nhưng làm mất tập trung vào dấu hiệu cục bộ -> MI/IMI giảm sâu. |
| `ARCH-3` | `arch_e03` | CNN + TF + **Lead Group Encoder** | 0.745 | **0.607** | 0.446 | `run_20260515_002105_hybrid-tf` | Nhóm đạo trình ép mạng nhìn đúng góc độ lâm sàng -> MI phục hồi mạnh. Nhưng bắt đầu xuất hiện xung đột Gradient làm giảm Arrhythmia. |
| `ARCH-4` | `arch_e04` | CNN + TF + Lead Group + **Task Token** | 0.714 | 0.570 | 0.461 | `run_20260515_012342_hybrid-tf` | Task Token giúp phân luồng chú ý tốt hơn cho IMI (AUPRC phục hồi). Tuy nhiên, Arrhythmia bị kéo sập xuống 0.714 do hai hàm Loss đánh nhau dữ dội. |

### 💡 Bài học rút ra (Takeaway) từ Phần 0:
1. **Sức mạnh của Transformer & Lead Group**: Transformer sinh ra để lo mảng nhịp tim toàn cục (Arrhythmia), trong khi Lead Group Encoder là "cứu tinh" để bắt được dấu hiệu nhồi máu cơ tim cục bộ (MI).
2. **Hiện tượng Chuyển giao Tiêu cực (Negative Transfer)**: Khi ta ráp toàn bộ các mảnh ghép lại thành cỗ máy phức tạp nhất (`ARCH-4`), mô hình lại bị "đảo lộn". Hàm Loss của MI quá gắt gao cạnh tranh trực tiếp với Arrhythmia, làm hỏng các bộ lọc (filters) của mạng CNN dùng chung, kéo Arrhythmia F1 từ đỉnh 0.781 xuống tận 0.714.
3. **Mệnh đề cho Phần 1**: Rõ ràng, một kiến trúc hoàn hảo là chưa đủ. Mạng `ARCH-4` chứa tiềm năng khổng lồ, nhưng nó **đòi hỏi một hệ thống Pipeline tối ưu hóa (Gradient Isolation, Focal Loss, Data Augmentation) cực kỳ tinh tế** để kìm hãm sự xung đột và phát huy tối đa công năng. Đó chính là lý do ra đời của chuỗi thử nghiệm ở Phần 1 dưới đây!

---

## PHẦN 1: KHÁM PHÁ KỸ THUẬT HUẤN LUYỆN (PIPELINE ABLATION)

Giai đoạn 1 (Split Strategy) và Giai đoạn 2 (Normalization) đã được xác nhận với cấu hình tối ưu lần lượt là **Random Grouped (seed 42)** và **Z-Score Normalization**. Từ Giai đoạn 3 trở đi, các thử nghiệm sẽ kế thừa hai cấu hình này.

### Stage 3: Data Augmentation
**Mục tiêu**: Đánh giá tác động của các kỹ thuật nội suy dữ liệu lên sự cân bằng giữa nhịp tim (Arrhythmia) và hình thái cục bộ (MI).
**Base Config**: Random Grouped Split + Z-Score Norm + BCE Loss (Không Focal, Không Weighted).

| Run ID | Cấu hình Augmentation | Arrhy (Macro F1) | MI (Macro F1) | IMI (AUPRC) | Nhận xét | Winner |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| `STAGE3-NONE` | Không dùng Augmentation | 0.674 | **0.645** | **0.534** | Overfit Arrhythmia (AFLT F1 = 0.0) nhưng giữ nguyên hình thái gốc nên MI cao. | |
| `STAGE3-NOISEWARP` | Gaussian Noise + Time Warp | **0.823** | 0.638 | 0.524 | Phục hồi hoàn toàn Arrhythmia (AFLT tăng từ 0 lên 0.689) với sự đánh đổi rất nhỏ ở IMI. | ✅ |
| `STAGE3-MIXUP` | MixUp (Alpha=0.2) | 0.813 | 0.635 | 0.494 | Kéo giảm IMI AUPRC mạnh do pha trộn phá vỡ cấu trúc ST-segment cục bộ. | |

### Stage 4: Data Sampling
**Mục tiêu**: Đánh giá hiệu quả của phương pháp lấy mẫu có trọng số (Weighted Random Sampler) so với lấy mẫu đồng đều (Uniform Sampler).
**Base Config**: Random Grouped Split + Z-Score Norm + **Noise & Warp** + BCE Loss.

| Run ID | Cấu hình Sampling | Arrhy (Macro F1) | MI (Macro F1) | IMI (AUPRC) | Nhận xét | Winner |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| `STAGE4-UNIFORM` | Uniform Sampler | **0.821** | **0.642** | 0.529 | Hiệu năng tổng thể cân bằng và tốt nhất. | ✅ |
| `STAGE4-WEIGHTED` | Weighted Random Sampler | 0.799 | 0.608 | **0.537** | IMI AUPRC tăng nhẹ nhưng kéo tụt hoàn toàn F1 của cả 2 mảng Arrhythmia và MI do bóp méo phân phối gốc. | |

**Kết luận Giai đoạn 4**: Lấy mẫu đồng đều (`Uniform Sampler`) giành chiến thắng. Việc ép mô hình học các mẫu hiếm quá nhiều bằng Weighted Sampler gây ra hiện tượng Over-representation, phá hỏng ranh giới quyết định (Decision Boundary) của các class đa số.

---

### Stage 5: Loss Function & Gradient Isolation
**Mục tiêu**: Giải quyết bài toán mất cân bằng nội tại (Class Imbalance) và xung đột Gradient (Gradient Interference) giữa các Task.
**Base Config**: Random Grouped Split + Z-Score Norm + Noise & Warp + Uniform Sampling.

| Run ID | Cấu hình Loss | Arrhy (Macro F1) | MI (Macro F1) | IMI (AUPRC) | Nhận xét | Winner |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| `STAGE5-BCE` | Standard BCE Loss | 0.821 | **0.642** | **0.529** | Đạt mức hiệu năng cực kỳ tốt và ổn định. | ✅ |
| `STAGE5-FOCAL` | Focal Loss (Gamma=2.0) | 0.808 | 0.642 | 0.474 | Focal Loss "trừng phạt" quá mạnh các mẫu khó, làm giảm sút IMI AUPRC nghiêm trọng. | |
| `STAGE5-FOCAL-GRADISO` | Focal Loss + **Gradient Isolation** | **0.823** | 0.642 | 0.523 | GradIso cứu sống Focal Loss, đưa IMI AUPRC và Arrhythmia phục hồi trở lại mức tương đương BCE. | |

**Kết luận Giai đoạn 5**: Khác với kỳ vọng ban đầu, khi ta đã có Data Augmentation (Noise + Warp) đủ mạnh, hệ thống đã tự học được cách chống chịu mất cân bằng. Việc thêm **Focal Loss** vào lúc này là dư thừa và thậm chí làm hại mô hình (AUPRC giảm). Mặc dù **Gradient Isolation** có thể sửa chữa sai lầm của Focal Loss (đưa Arrhy lên 0.823), cấu hình **Standard BCE** vẫn là cấu hình chiến thắng vì sự đơn giản (Occam's razor) và AUPRC nhỉnh hơn một chút.

---

## TỔNG KẾT PHASE 1 (PIPELINE ABLATION)
Sau 5 Giai đoạn thử nghiệm khắt khe, **Cấu hình Huấn luyện Tối ưu nhất (Golden Pipeline)** cho mạng Hybrid-Transformer được chốt như sau:
1. **Split**: Random Grouped (seed 42)
2. **Normalization**: Per-lead Z-Score
3. **Data Augmentation**: Gaussian Noise + Time Warp
4. **Sampling**: Uniform Sampler
5. **Loss Function**: Standard BCE Loss (Mi_gradient_scale = 1.0)

Cấu hình này đã giải quyết trọn vẹn bài toán **Negative Transfer** xuất hiện ở Phần 0, đẩy F1 của Arrhythmia từ `0.714 -> 0.821` và IMI AUPRC từ `0.461 -> 0.529` trên cùng một kiến trúc `Arch-4`.

---

## PHẦN 2: TINH CHỈNH VÀ THẨM ĐỊNH LÂM SÀNG

### Giai đoạn 6: So sánh chiến lược Threshold (Nhánh: `TEST-DYNAMIC-THRESHOLD`)

**Mục tiêu**: Trả lời câu hỏi nghiên cứu: Liệu việc để mô hình **tự dự đoán ngưỡng cắt** (Dynamic Threshold) có cải thiện macro-F1 so với dùng ngưỡng cố định 0.5 hay ngưỡng tìm kiếm trên Val Set (Per-class Tuning) không?

**3 cấu hình so sánh** (tất cả đều dùng Golden Pipeline từ Phase 1):

| ID | Config | Mô tả | File Config |
| :--- | :--- | :--- | :--- |
| `DYNTH-E01` | Fixed 0.5 | **Control**: ngưỡng cố định 0.5 | `configs/experiments/dynth_e01_fixed.yaml` |
| `DYNTH-E02` | Per-class Tuning | Tìm ngưỡng F-beta tối ưu trên Val Set mỗi 5 epoch | `configs/experiments/dynth_e02_perclass.yaml` |
| `DYNTH-E03` | Dynamic Head | Model tự dự đoán ngưỡng theo feature bệnh nhân | `configs/experiments/dynth_e03_dynamic.yaml` |

**Lệnh chạy:**
```bash
python scripts/02_train.py --config configs/experiments/dynth_e01_fixed.yaml
python scripts/02_train.py --config configs/experiments/dynth_e02_perclass.yaml
python scripts/02_train.py --config configs/experiments/dynth_e03_dynamic.yaml
```

**Cơ chế hoạt động của Dynamic Threshold Head (`DynamicThresholdHead`):**
- Module nhỏ: `Linear(256→64) → LayerNorm → GELU → Linear(64→7) → Sigmoid`
- Nhận vào `shared_features` (256-dim, bị `.detach()` để không ảnh hưởng backbone).
- Đầu ra: vector `τ ∈ (0,1)^7` — một ngưỡng riêng cho từng class của từng bệnh nhân.
- **Consistency Loss**: khi class là positive (y=1) → target τ thấp; khi negative (y=0) → target τ cao. Loss weight = 0.05.

### Kết quả So sánh Threshold (cập nhật sau khi train xong)

| Run ID | Chiến lược Threshold | Arrhy (Macro F1) | MI (Macro F1) | IMI (AUPRC) | Checkpoint | Winner |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| `DYNTH-E01` | Fixed 0.5 | TBD | TBD | TBD | TBD | |
| `DYNTH-E02` | Per-class Tuning (F-beta) | TBD | TBD | TBD | TBD | |
| `DYNTH-E03` | Dynamic Threshold Head | TBD | TBD | TBD | TBD | |

---

*(Đang chờ kết quả training)*
