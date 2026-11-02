# Kết quả ARCH-13/14 và đối chứng cùng máy (RTX 5070 Ti, 30/09/2026)

Nguồn:
- 60 run trong `checkpoints/box_5070ti_b/`: 6 arm × 10 seed, cùng một máy, driver 595.99.02
- báo cáo đầy đủ ở `docs/stat_tests_contrast_5070ti.md` và `docs/stat_tests_prereg_5070ti.md`

Tất cả là chỉ số validation tại epoch được chọn bởi monitor `imi_auprc`. Dấu Δ: dương nghĩa
là arm có khối đối lập (hoặc bước sau trong chuỗi) cao hơn.

## 1. Bốn cặp về khối đối lập — IMI AUPRC, cùng máy

| Cặp | Có | Không | Δ | 95% CI | Seed thắng | p (Wilcoxon chính xác) |
|---|---|---|---|---|---|---|
| **ARCH-12 − ARCH-14**: tách một phần, bật/tắt khối đối lập | 0,6609 | 0,6234 | **+0,0375** | [+0,027; +0,047] | 9/10 | **0,0039** |
| **ARCH-10 − ARCH-13**: tách hoàn toàn, bật/tắt khối đối lập | 0,6578 | 0,6587 | −0,0009 | [−0,004; +0,002] | 4/10 | 0,63 |
| ARCH-14 − ARCH-05: chỉ projection + head | 0,6234 | 0,6281 | −0,0047 | [−0,015; +0,005] | 4/10 | 0,43 |
| ARCH-13 − ARCH-09: chỉ bố cục token `contrast_v2` | 0,6587 | 0,6174 | **+0,0414** | [+0,031; +0,051] | 10/10 | **0,0020** |

Các chỉ số khác:
- **ASMI**: khối đối lập giúp khi tách một phần (+0,014, CI [+0,008; +0,020], p = 0,006).
  Bố cục token giúp khi tách hoàn toàn (+0,018, p = 0,002).
- **ILMI, AMI, S**: không cặp nào khác biệt có ý nghĩa thống kê.
- **MI macro**: chỉ bố cục token có tác dụng (+0,019, p = 0,002).

**Kết luận.** Có hai con đường thay thế nhau để IMI và ASMI tách được khỏi nhãn anh em:
1. Khối đối lập có cổng tường minh, khi các nhãn dùng chung token (tách một phần).
2. Bố cục token theo cụm giải phẫu, khi tách hoàn toàn. Khi đã có bố cục này, khối đối lập
   không thêm gì (Δ = −0,001, CI trong ±0,005).

Projection và head tự chúng không đóng góp gì. Hai con đường đạt cùng mức IMI (0,658–0,661)
và không cộng dồn.

## 2. Giả thuyết đăng ký trước — lặp lại trên GPU thứ hai

| Cặp | 5060 Ti (bài) | 5070 Ti |
|---|---|---|
| Chính: ARCH-10 − ARCH-09 | +0,0396, p = 0,0039 | **+0,0404**, CI [+0,030; +0,050], 10/10, p = 0,0020 |
| Phụ: ARCH-10 − ARCH-05 | (xem bài) | +0,0297, CI [+0,018; +0,041], 10/10, p_holm = 0,0039 |
| Phụ, giả thuyết không: ARCH-09 − ARCH-05 | (xem bài) | −0,0107, CI [−0,020; −0,002], p = 0,065 |

Kết quả chính lặp lại gần như nguyên vẹn trên phần cứng khác.

> **Lưu ý về dòng giả thuyết không.** `stat_tests.py` ghi "inconclusive (CI includes 0 ...)",
> nhưng CI bootstrap ở đây **không** chứa 0. Nhãn đó được gán chỉ dựa trên p ≥ 0,05. Mô tả
> đúng là: ARCH-09 thấp hơn ARCH-05 khoảng 0,01. Bootstrap loại được 0, Wilcoxon thì chưa
> đủ mức ý nghĩa (p = 0,065). Tách hoàn toàn đơn thuần không giúp IMI, và có thể còn kém nhẹ.

## 3. Độ lệch do phần cứng — IMI AUPRC, so từng seed

| So sánh | Arm | Trung bình \|Δ\| | Lớn nhất \|Δ\| | Chênh trung bình arm |
|---|---|---|---|---|
| 5060 Ti so với 5070 Ti | ARCH-05 | 0,012 | 0,026 | −0,001 |
| | ARCH-09 | 0,011 | 0,026 | −0,004 |
| | ARCH-10 | 0,008 | 0,017 | −0,003 |
| | ARCH-12 | 0,007 | 0,016 | +0,003 |
| 5070 Ti máy A so với máy B | ARCH-13 | **0,0000** | 0,0000 | 0 |
| | ARCH-14 | 0,0008 | 0,0015 | — |

Đọc bảng:
- **Đổi loại GPU** làm từng run lệch tới 0,026, nhưng trung bình mỗi arm chỉ dịch ≤ 0,004.
  Vì vậy so sánh giữa các arm phải ghép cặp trên cùng máy. Mức trung bình của từng arm thì
  khá ổn định.
- **Cùng loại GPU, khác máy và khác driver**:
  - Nhánh MI của kiến trúc tách hoàn toàn (ARCH-13) cho **IMI trùng từng bit** qua mọi
    epoch. Chỉ các chỉ số của nhiệm vụ khác (loạn nhịp, dẫn truyền) lệch.
  - Kiến trúc dùng backbone chung (ARCH-14) lệch ≤ 0,0015.
  - Nhánh MI tách rời có tham số và loss riêng, nên không nhận nhiễu tính toán từ phần
    backbone chung.
- 19 run ARCH-13/14 trên máy A (29/09) cho cùng kết luận với máy B:
  - ARCH-12 − ARCH-14: +0,034 (so khác GPU) và +0,0375 (cùng máy)
  - ARCH-10 − ARCH-13: +0,002 và −0,001

  So sánh khác GPU hôm 29/09 vì vậy không bị độ lệch phần cứng làm sai hướng.

## 4. Đưa vào bài

- Bảng khảo sát kiến trúc: thêm ARCH-13 và ARCH-14. Ghi rõ chúng cùng ARCH-05/09/10/12 được
  chạy lại trên một máy 5070 Ti để so ghép cặp. Số 5060 Ti trong các bảng hiện có giữ nguyên.
- ARCH-13/14 là so sánh **sau khi đã chốt giao thức**: báo cáo mô tả, không hiệu chỉnh Holm.
- §3.5/§5.3: viết lại theo hướng "hai con đường" ở mục 1.
- Có thể thêm một câu về khả năng lặp lại: giả thuyết chính lặp lại trên GPU thứ hai
  (+0,040, p = 0,002).
- Phụ lục về tái lập (tuỳ chọn): các số ở mục 3.

## 5. ARCH-15: bỏ cổng, giữ khối đối lập (box C, 5070 Ti, 29/10/2026)

ARCH-15 = ARCH-12, nhưng c̃ = c + W_p δ + b_p (không có σ(W_g ·)). Ít hơn ARCH-12 65.792 tham số.
Box C chạy 10 seed ARCH-15 và chạy lại 10 seed ARCH-12 để cặp "cổng bật/tắt" nằm trong cùng
một máy. Kết quả: `checkpoints/box_5070ti_c/`, bảng đầy đủ:
- `docs/stat_tests_gate_5070ti_c.md`: trong box C
- `docs/stat_tests_gate_crossbox_BC.md`: thêm ARCH-14 của box B

| Cặp (IMI AUPRC, validation) | Δ | 95% CI | thắng | p | máy |
|---|---|---|---|---|---|
| ARCH-12 − ARCH-15 (cổng bật − tắt) | +0,0002 | [−0,0065; +0,0067] | 5/10 | 1,000 | cùng box C |
| ARCH-15 − ARCH-14 (đối lập không cổng − không đối lập) | +0,0358 | [+0,026; +0,046] | 10/10 | 0,002 | C vs B |
| ARCH-12 − ARCH-14 (đối lập có cổng − không đối lập) | +0,0359 | [+0,025; +0,046] | 9/10 | 0,004 | C vs B |

Các chỉ số khác của ARCH-12 − ARCH-15 (cùng box C) đều không có ý nghĩa thống kê:
- ILMI −0,012 (p 0,23)
- ASMI +0,003
- AMI +0,013 (p 0,23)
- MI macro +0,001
- S/composite +0,005 (CI [0,000; +0,010], p 0,11)

Độ lệch ARCH-12 giữa box B và box C, IMI AUPRC, so từng seed:
- trung bình −0,0015, lớn nhất 0,0058
- S/composite trung bình +0,0001

Độ lệch này nhỏ hơn hiệu ứng 0,036 khoảng 6 lần, nên cặp ARCH-15 − ARCH-14 khác máy vẫn tin được.
Cặp này có một điểm phụ: AMI −0,031 (p 0,027, 1/10). AMI chỉ có rất ít ca dương tính và lệch
giữa hai máy khá lớn, nên chỉ ghi lại, không diễn giải.

**Kết luận:**
- Cổng không đóng góp gì. Toàn bộ mức tăng IMI ở nhánh tách một phần đến từ phép chiếu hiệu
  W_p δ.
- Điều này khớp với kết quả kiểm tra cổng trước đó:
  - b_g ≈ −2, nên cổng gần như cố định quanh 0,1–0,17, tức là một hệ số co gần như hằng số
  - W_p học được để bù lại hệ số này

**Đưa vào bài:**
- §3.5 không được nói cổng chọn lọc theo chiều hay là thành phần then chốt.
- Viết "khối đối lập anh em (sibling contrast)". Cổng ghi là chi tiết hiện thực, kèm một câu
  ablation: "Removing the sigmoid gate (ARCH-15) leaves IMI AUPRC unchanged (Δ = +0.0002,
  p = 1.0, same hardware); the gain is carried by the projected sibling difference."
- Thêm hàng ARCH-15 vào bảng khảo sát kiến trúc, ghi rõ chạy trên 5070 Ti, sau khi chốt giao thức.
