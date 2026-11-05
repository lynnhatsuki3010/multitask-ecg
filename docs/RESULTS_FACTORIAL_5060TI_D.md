# Kết quả hai bảng 2×2: bố cục token × khối đối lập (box D, 4×RTX 5060 Ti, 05/11/2026)

## Nguồn

- 80 run: 8 arm × 10 seed, cùng một máy, driver 580.173.02. Kết quả ở `checkpoints/box_5060ti_d/`.
- Kiểm định validation: `docs/stat_tests_factorial_5060ti_d.md`.
- Che đạo trình trên tập test: `results/occ_d_*.csv`, sinh bởi `scripts/24_sibling_occlusion.py`.
- Đây là so sánh thêm sau khi chốt giao thức: báo cáo mô tả, không hiệu chỉnh Holm.

## 0. Tái lập

ARCH-09, ARCH-10 và ARCH-12 chạy lại trên box D cho IMI AUPRC **trùng từng bit** với đợt chạy gốc
trong bài (5060 Ti, 15/08): 30/30 run giống hệt.

Vì vậy số của box D ghép thẳng được với các bảng hiện có, không cần hiệu chỉnh gì.

Ba kết quả của box B (5070 Ti) cũng lặp lại trên 5060 Ti:

| Cặp | Box B (5070 Ti) | Box D (5060 Ti) |
|---|---|---|
| ARCH-13 − 09 (bố cục token) | +0,041 | +0,039 |
| ARCH-12 − 14 (khối đối lập) | +0,038 | +0,041 |
| ARCH-10 − 13 (đối lập trên nền bố cục) | −0,001 | +0,001 |

## 1. Hai bảng 2×2 — IMI AUPRC trên validation, trung bình 10 seed

| | Không có gì | Chỉ bố cục | Chỉ đối lập | Cả hai |
|---|---|---|---|---|
| **Tách hoàn toàn** | ARCH-09: 0,621 | ARCH-13: 0,660 | ARCH-16: 0,651 | ARCH-10: 0,661 |
| **Dùng chung backbone** | ARCH-14: 0,616 | ARCH-17: 0,668 | ARCH-12: 0,658 | ARCH-18: 0,663 |

Các cặp so sánh, ghép cặp theo seed:

| Cặp | Δ | 95% CI | Thắng | p |
|---|---|---|---|---|
| ARCH-13 − 09: bố cục, tách hoàn toàn | **+0,039** | [+0,026; +0,051] | 9/10 | 0,004 |
| ARCH-16 − 09: đối lập, tách hoàn toàn | **+0,029** | [+0,022; +0,035] | 9/10 | 0,004 |
| ARCH-17 − 14: bố cục, dùng chung backbone | **+0,051** | [+0,042; +0,060] | 10/10 | 0,002 |
| ARCH-12 − 14: đối lập, dùng chung backbone | **+0,041** | [+0,030; +0,053] | 10/10 | 0,002 |
| ARCH-10 − 13: thêm đối lập vào bố cục | +0,001 | [−0,002; +0,003] | 6/10 | 0,43 |
| ARCH-10 − 16: thêm bố cục vào đối lập | +0,010 | [+0,001; +0,020] | 7/10 | 0,11 |
| ARCH-18 − 17: thêm đối lập vào bố cục | −0,005 | [−0,016; +0,006] | 5/10 | 0,49 |
| ARCH-18 − 12: thêm bố cục vào đối lập | +0,005 | [−0,006; +0,016] | 6/10 | 0,49 |

Trên tập test, IMI AUPRC tăng ở cả 6 arm có ít nhất một con đường: +0,038 đến +0,053 so với arm
không có gì cùng hàng, p ≤ 0,01.

**Đọc kết quả:**
- **Mỗi con đường đều có tác dụng ở cả hai kiểu định tuyến.** Hai ô trước đây chưa thử, ARCH-16 và
  ARCH-17, đều dương rõ rệt.
- **Hai con đường thay thế nhau, không cộng dồn.** Thêm con đường thứ hai lên trên con đường thứ
  nhất cho Δ ≈ 0 ở cả hai hàng.
- **Bố cục mạnh hơn khối đối lập một chút, ở cả hai hàng (khoảng +0,01).** Đây là lý do ARCH-10 −
  ARCH-16 hơi dương.
- **Ô tốt nhất là ARCH-17** (dùng chung backbone + bố cục, không có khối đối lập). Nó hơn ARCH-10
  +0,007, CI [−0,001; +0,015], p = 0,23: không có ý nghĩa. Vì vậy không có lý do đổi mô hình chính.
- **S/composite không đổi** ở mọi cặp (|Δ| ≤ 0,005).

## 2. Tác dụng phụ trên các nhãn khác (validation)

| Cặp | ILMI | ASMI | AMI | MI macro |
|---|---|---|---|---|
| ARCH-13 − 09 (bố cục) | −0,001 | **+0,017** | +0,008 | **+0,016** |
| ARCH-16 − 09 (đối lập) | −0,002 | +0,003 | −0,015 | +0,004 |
| ARCH-17 − 14 (bố cục) | **−0,031** (p 0,027) | +0,001 | **+0,027** (p 0,037) | **+0,012** |
| ARCH-12 − 14 (đối lập) | −0,004 | +0,004 | +0,001 | +0,011 |

- **Bố cục còn giúp AMI và MI macro.** Khối đối lập thì gần như chỉ giúp IMI.
- **Bố cục khi dùng chung backbone làm ILMI giảm khoảng 0,03.** ARCH-18 − 12 cũng cho −0,029. Nên ghi
  thành hạn chế nếu nhắc đến ARCH-17.

## 3. Cơ chế: che V5–V6 trên tập test, cả 8 arm

Chỉ số là AUROC của đầu IMI khi tách bản ghi chỉ-IMI khỏi bản ghi chỉ-ILMI (320 bản ghi một nhãn MI).

| Arm | Con đường | AUROC | Khi che V5–V6 | Δ | Seed giảm | Che V3–V4 (đối chứng) |
|---|---|---|---|---|---|---|
| ARCH-09 | không có | 0,535 | 0,535 | 0,000 | 0/10 | 0,000 |
| ARCH-14 | không có | 0,557 | 0,538 | −0,019 | 8/10 | +0,005 |
| ARCH-13 | bố cục | 0,632 | 0,456 | **−0,176** | 10/10 | 0,000 |
| ARCH-17 | bố cục | 0,631 | 0,465 | **−0,166** | 10/10 | +0,010 |
| ARCH-16 | đối lập | **0,664** | 0,499 | **−0,165** | 10/10 | 0,000 |
| ARCH-12 | đối lập | 0,631 | 0,461 | **−0,170** | 10/10 | +0,009 |
| ARCH-10 | cả hai | 0,645 | 0,474 | **−0,172** | 10/10 | 0,000 |
| ARCH-18 | cả hai | 0,639 | 0,500 | **−0,139** | 10/10 | +0,014 |

Tỷ lệ ILMI bị đoán thành IMI:
- Hai arm không có con đường: khoảng 61%.
- Sáu arm có con đường: 38–45%.
- Khi che V5–V6, sáu arm này tăng lên 61–73%.

**Đọc kết quả:**
- **Quy luật đúng ở cả 8 arm.** Arm nào có một trong hai con đường thì đầu IMI loại ILMI bằng V5–V6,
  và che V5–V6 xoá lợi thế đó (giảm 0,14–0,18, 10/10 seed). Arm không có con đường thì không phụ
  thuộc V5–V6. Che V3–V4 không có tác dụng ở mọi arm.
- **ARCH-16 chứng minh kênh δ.** Nhánh IMI của ARCH-16 không nhận V5–V6 làm đầu vào, vì nó dùng bố
  cục "shared" của ARCH-09. Vậy mà:
  - đầu IMI đạt AUROC cao nhất cả bảng (0,664)
  - che V5–V6 làm giảm 0,165, ở 10/10 seed

  Thông tin thành bên chỉ có thể đi vào qua δ = x − x_ILMI. Câu "δ mang thông tin thành bên" vì vậy
  không còn là suy luận nữa.

## 4. Đưa vào bài

- **Bảng khảo sát kiến trúc:** thêm hai bảng 2×2. Ghi rõ:
  - cùng máy 5060 Ti
  - ARCH-09/10/12 trùng từng bit với đợt gốc
  - ARCH-13/14/16/17/18 được thêm sau khi chốt giao thức
- **Câu chính:**
  > Two interchangeable routes — a clustered region-token layout, or a projected sibling difference — each raise inferior-MI AUPRC by 0.03–0.05 in both full and partial decoupling, and they do not add up. In all eight configurations the gain appears exactly when the IMI head can use V5–V6 to reject ILMI: occluding V5–V6 removes it (10/10 seeds), occluding anterior leads does not. In ARCH-16 the IMI branch never receives V5–V6, so this lateral evidence can only arrive through the sibling difference.
- **Mô hình chính vẫn là ARCH-10.** ARCH-17 không hơn có ý nghĩa, không rẻ hơn (13,48 triệu so với
  13,24 triệu tham số), và làm ILMI giảm.
- **Hạn chế:**
  - chỉ số chính là validation; tập test chỉ dùng cho phân tích cơ chế
  - ILMI chỉ có 24 bản ghi một nhãn
  - bố cục khi dùng chung backbone làm ILMI giảm
