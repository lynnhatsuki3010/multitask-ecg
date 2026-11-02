# Nhầm lẫn giữa các nhãn anh em trên tập test: kiến trúc so với baseline (10 seed)

Nguồn:
- Script: `scripts/23_sibling_confusion.py`
- Số liệu thô: `results/sibling_confusion_test.csv`
- Checkpoint: các run gốc chạy trên 5060 Ti (`checkpoints/run_*`), 4 arm × 10 seed

Cách tính:
- Lấy 320 bản ghi test chỉ mang **đúng một** nhãn MI: IMI 134, ASMI 144, ILMI 24, AMI 18.
- Quy tắc dự đoán giống Bảng 7: chia logit cho nhiệt độ T_mi, lấy sigmoid, rồi so với ngưỡng
  riêng của từng nhãn trong `calibration_results.json` của run đó.
- Kiểm tra: ARCH-10 seed 42 cho lại đúng Bảng 7 (ILMI→IMI 11/24, IMI→ILMI 15/134, ILMI đúng 15).

## 1. Ma trận theo ngưỡng — trung bình 10 seed

| Chỉ số | ARCH-05 | ARCH-09 | ARCH-10 | ARCH-12 |
|---|---|---|---|---|
| ILMI bị đoán thành IMI (/24) | 0,638 | 0,612 | **0,417** | **0,383** |
| IMI bị đoán thành ILMI (/134) | 0,081 | 0,063 | 0,052 | 0,090 |
| IMI đoán đúng | 0,628 | 0,595 | 0,596 | 0,557 |
| ILMI đoán đúng | 0,533 | 0,583 | 0,542 | 0,538 |
| AMI bị đoán thành ASMI (/18) | 0,189 | 0,211 | 0,256 | 0,228 |

Kiểm định ghép cặp Wilcoxon, ILMI→IMI:
- ARCH-10 − ARCH-09: −0,196, 10/10 seed giảm, p = 0,002
- ARCH-10 − ARCH-05: −0,221, 9/10, p = 0,004
- ARCH-12 − ARCH-05: −0,254, 10/10, p = 0,002
- ARCH-09 − ARCH-05: −0,025, p = 0,48 (tách nhánh đơn thuần không giúp gì)

Các tác dụng phụ:
- ARCH-12 bỏ sót IMI nhiều hơn: tỷ lệ đoán đúng IMI −0,070, p = 0,018.
- ARCH-10 không bị như vậy so với ARCH-09 (+0,001).
- AMI→ASMI tăng nhẹ ở ARCH-10 so với ARCH-05: +0,067, p = 0,023. Mẫu chỉ có 18 ca.

## 2. Không phụ thuộc ngưỡng — AUROC từng đầu nhãn, chỉ trên bản ghi của cặp

Câu hỏi: đầu IMI có cho bản ghi chỉ-IMI điểm cao hơn bản ghi chỉ-ILMI không?

| AUROC | ARCH-05 | ARCH-09 | ARCH-10 | ARCH-12 |
|---|---|---|---|---|
| Đầu IMI: IMI-only so với ILMI-only | 0,545 | 0,535 | **0,645** | **0,631** |
| Đầu ILMI: ILMI-only so với IMI-only | 0,853 | 0,824 | 0,824 | 0,850 |
| Hiệu logit z_IMI − z_ILMI | 0,880 | 0,872 | 0,862 | 0,875 |
| Đầu ASMI: ASMI-only so với AMI-only | 0,830 | 0,840 | 0,815 | 0,826 |
| Đầu AMI: AMI-only so với ASMI-only | 0,734 | 0,722 | 0,747 | 0,713 |

Kiểm định cho đầu IMI:
- ARCH-10 − ARCH-09: +0,111, 10/10, p = 0,002
- ARCH-10 − ARCH-05: +0,100, 10/10, p = 0,002
- ARCH-12 − ARCH-05: +0,086, 9/10, p = 0,004

Đầu ILMI, đầu ASMI, đầu AMI và hiệu logit: không cải thiện, có chỗ còn giảm nhẹ
(ASMI ARCH-10 − 09: −0,026, p = 0,027).

## 3. Đọc kết quả

- Ở baseline, **đầu IMI gần như không phân biệt được IMI với ILMI** (AUROC 0,54, gần mức
  ngẫu nhiên).
  - Lý do giải phẫu: ILMI mang toàn bộ dấu hiệu thành dưới của IMI, nên đầu IMI bật lên với
    cả ILMI.
  - Đầu ILMI thì đã phân biệt tốt (0,82–0,85), vì nó dựa vào phần V5–V6 mà IMI không có.
- **Cả hai con đường kiến trúc đều sửa đúng chỗ này**:
  - đầu IMI tăng lên 0,63–0,65
  - ILMI→IMI giảm từ khoảng 62% xuống khoảng 40%
  - đây là cơ chế cụ thể đứng sau mức tăng IMI AUPRC
- **Phần chưa giải quyết**:
  - Đầu IMI vẫn còn xa mức đầu ILMI đạt được.
  - Cặp ASMI/AMI không có hiện tượng này. Ở baseline đầu ASMI đã phân biệt tốt (0,83). Vì vậy
    mức tăng ASMI AUPRC trên validation không đến từ việc loại AMI.
- **Giới hạn**:
  - ILMI chỉ có 24 bản ghi một nhãn, nên mỗi bản ghi bằng 4,2 điểm phần trăm.
  - Ngưỡng là của từng run.
  - Checkpoint được chọn theo `imi_auprc` trên validation, nên có thể thiên về IMI.

## 4. Che đạo trình — kiểm chứng cơ chế (test, 10 seed)

Nguồn:
- Script: `scripts/24_sibling_occlusion.py`
- Số liệu thô: `results/sibling_occlusion_test.csv`

Cách làm: đặt các đạo trình được chọn về 0 trên toàn bộ tập test. Đây là cùng phép biến đổi
với augmentation lead-dropout lúc train. Thứ tự đạo trình: I II III aVR aVL aVF V1–V6.

### 4.1 Đầu IMI: AUROC IMI-only so với ILMI-only

| Che | ARCH-05 | ARCH-09 | ARCH-10 | ARCH-12 |
|---|---|---|---|---|
| không che | 0,545 | 0,535 | 0,645 | 0,631 |
| **V5, V6** | 0,520 | 0,535 | **0,474** | **0,461** |
| V3, V4 (đối chứng) | 0,550 | 0,535 | 0,645 | 0,640 |
| V1, V2 (đối chứng) | 0,527 | 0,535 | 0,645 | 0,634 |
| V1–V4 (đối chứng) | 0,555 | 0,535 | 0,645 | 0,664 |
| I, aVL, V5, V6 | 0,446 | 0,420 | 0,385 | 0,391 |

Tỷ lệ ILMI bị đoán thành IMI khi che V5–V6:
- ARCH-10: 42% → 71% (+0,292, 10/10 seed, p = 0,002)
- ARCH-12: 38% → 70% (+0,312, 10/10, p = 0,002)
- ARCH-09: không đổi
- ARCH-05: +0,021 (p = 0,31)

Khả năng nhận đúng IMI khi che V5–V6 gần như không đổi:
- ARCH-10: −0,022 (p = 0,24)
- ARCH-12: +0,013

Đọc kết quả:
- **Toàn bộ lợi thế của hai kiến trúc nằm ở V5–V6.** Che hai đạo trình này thì AUROC của đầu IMI
  giảm 0,17 (10/10 seed). Mức còn lại thấp hơn cả baseline: ARCH-10 − ARCH-09 dưới V5–V6
  bị che là −0,061 (p = 0,006).
- **Phép che đối chứng không ảnh hưởng.** Che cùng số đạo trình ở vùng trước (V1–V4) không
  thay đổi gì.
- **Che V5–V6 không làm đầu IMI kém nhận ra IMI.** V5–V6 được dùng để **phủ quyết ILMI**, không
  phải để phát hiện IMI.
- **ARCH-09 hoàn toàn không phụ thuộc vào V1–V6.** Nhánh IMI của nó chỉ đọc II, III, aVF, I, aVL.
  Vì vậy ở baseline tách hoàn toàn, đầu IMI không có cách nào thấy được dấu hiệu thành bên.
- **ARCH-10 cũng không phụ thuộc vào V1–V4**, đúng với bố cục cụm: nhánh IMI đọc
  inferior, reciprocal, lateral và inferolateral.
- **Ở ARCH-12, sự phụ thuộc vào V5–V6 chỉ xuất hiện khi có khối đối lập.** Baseline dùng chung
  backbone (ARCH-05) hầu như không dùng V5–V6 (−0,025). Điều này khớp với giải thích rằng δ =
  x − x_ILMI mang thông tin thành bên từ nhánh ILMI sang. Muốn khẳng định chắc cần chạy che
  đạo trình trên ARCH-14, nhưng trọng số của ARCH-14 không được tải về từ box B.
- **Che thêm I và aVL làm mọi mô hình giảm**, vì tất cả đều đọc I và aVL như đạo trình soi gương
  (reciprocal). Phép che này vì vậy kém đặc hiệu hơn.

### 4.2 Cặp ASMI/AMI

- **Đầu ASMI phân biệt AMI bằng V1–V4 ở mọi mô hình như nhau.** Che V3–V4 làm AUROC giảm
  0,12–0,20, che V1–V4 làm giảm 0,21–0,34, ở cả bốn arm. Đây là bằng chứng có mặt, và mọi kiến
  trúc đều đã khai thác được nó.
- **Trên test không có mức tăng ASMI:**
  - ASMI AUPRC: ARCH-10 − 09 = −0,0004 (p = 0,85); ARCH-12 − 05 = −0,004 (p = 0,77)
  - AUROC của đầu ASMI so với NORM, LBBB, các bệnh không-MI khác và các MI khác: chênh nhau
    ≤ 0,005, không có ý nghĩa
  - Mức tăng ASMI trên validation chỉ thấy ở ARCH-13 − 09 và ARCH-12 − 14 (box B). Nó **không lặp
    lại** trên test ở các arm gốc, nên không đưa thành kết luận.

### 4.3 IMI AUPRC trên test (để đối chiếu)

- ARCH-10 − ARCH-09: +0,053 (10/10, p = 0,002)
- ARCH-12 − ARCH-05: +0,043 (9/10, p = 0,004)

ILMI và AMI: không có khác biệt có ý nghĩa.
