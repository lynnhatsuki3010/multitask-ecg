# Runbook — ARCH-13 và ARCH-14 trên 2 × RTX 5070 Ti

Hai arm mới, mỗi arm 10 seed. Cùng trả lời một câu hỏi: **khối đối lập có cổng tự
nó có đóng góp gì không?**

| Arm | Là gì | So với | Đọc ra |
|---|---|---|---|
| **ARCH-14** `abl_arch14_partial_nogate` | ARCH-12 bỏ đúng 2 khối `SiblingContrast` (giữ projection + head) | ARCH-12 | Hiệu ứng khối đối lập khi tách **một phần** — câu hỏi chính |
| **ARCH-13** `abl_arch13_nogate` | ARCH-10 bỏ đúng 2 khối `SiblingContrast` (giữ bố cục token `contrast_v2`) | ARCH-10 | Hiệu ứng khối đối lập khi tách **hoàn toàn** |

Chênh lệch tham số của cả hai cặp đúng bằng 2 khối đối lập: **98 816**.

- ARCH-12 → ARCH-14: 12 809 649 → 12 710 833
- ARCH-10 → ARCH-13: 13 238 832 → 13 140 016

## Đã kiểm tra trước khi viết runbook (29/09/2026, máy local)

| Kiểm tra | Kết quả |
|---|---|
| ARCH-12: 10 checkpoint cũ load `strict=True` sau khi sửa code | OK |
| ARCH-12 IMI AUPRC val sau khi sửa code, seed 42 / 123 | 0,638528 / 0,634881, khớp tới 6 chữ số |
| ARCH-10 load strict sau khi thêm `contrast_v2_nogate` | OK, IMI AUPRC val 0,6685 không đổi |
| `02_train.py --debug` với ARCH-14 và ARCH-13 | exit 0, số tham số đúng như trên |
| `03_calibrate.py --dir` trên run ARCH-14 và ARCH-13 | exit 0, ra đủ `calibration_results.json` và `class_report.json` |
| `stat_tests_extra.py` chạy lại cặp đã biết ARCH-10 − ARCH-09 | +0,0396, p = 0,0039, **khớp số trong bài** |

---

## ⚠ Một điều phải biết trước khi đọc kết quả

Các run ARCH-12 và ARCH-10 trong bài chạy trên **RTX 5060 Ti**. Nếu chỉ chạy ARCH-13/14
trên 5070 Ti rồi so với các run đó, hiệu số sẽ **lẫn độ lệch do đổi GPU**. Dự án đã đo độ
lệch này lên tới **0,018 IMI AUPRC**, cỡ bằng hiệu ứng đang cần đo.

Cách gỡ nằm ở **mục 7 (tuỳ chọn)**: chạy thêm ARCH-12 và ARCH-10 trên chính máy này. Khi đó
cả hai cặp so sánh đều cùng máy. Tốn thêm khoảng 2 giờ, tức khoảng 2,2 USD với giá 1,104
USD/giờ, trong khi môi trường đã dựng sẵn. Để quyết định sau cũng được: runbook giữ đủ
thông tin để một máy 5070 Ti khác chạy nốt, qua cổng `verify_box` ở mục 7.

---

## 1. Đóng gói ở máy local

> Mọi lệnh ở **máy local** trong runbook này viết cho **cmd.exe** (prompt `C:\...>`).
> Trong cmd phải dùng `cd /d` thì mới chuyển được sang ổ E.

Gói đã tạo sẵn ngày 29/09: `E:\KLTN\KL Project\ecg_box.tar.gz` (5,1 MB). Chỉ tạo lại khi
code có thay đổi:

```bat
cd /d "E:\KLTN\KL Project\ECG_HRV" && tar -czf ..\ecg_box.tar.gz src scripts configs requirements.txt data/processed_dir5_imi50_500hz data/splits_dir5_imi50_500hz data/raw/PTB-XL/SHA256SUMS.txt docs/RUNBOOK_ARCH13_14.md && dir ..\ecg_box.tar.gz
```

Gói gồm sẵn `processed` và `splits`: **dùng nguyên, không chạy lại
`01_build_metadata.py`**, để label matrix và split trùng từng byte với các run trong bài.

```bat
cd /d "E:\KLTN\KL Project" && scp -P <PORT> ecg_box.tar.gz root@<HOST>:/workspace/
```

`<PORT>` và `<HOST>` lấy ở nút **Connect** của instance trên vast.ai.

---

## 2. Dựng môi trường trên máy thuê

```bash
apt-get update -qq && apt-get install -y -qq tmux && mkdir -p /workspace/ECG_HRV /workspace/logs && tar -xzf /workspace/ecg_box.tar.gz -C /workspace/ECG_HRV && ls /workspace/ECG_HRV
```

> Template vast đặt Python thật ở `/venv/main`. Shell không tương tác rơi về
> `/usr/bin/python3`, nơi không có torch. **Mọi lệnh bắt đầu bằng
> `export PATH=/venv/main/bin:$PATH`.**

```bash
export PATH=/venv/main/bin:$PATH && python -c "import torch,numpy,scipy,sklearn;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.device_count(), numpy.__version__, scipy.__version__, sklearn.__version__)"
```

Mong đợi, cùng phần mềm với đợt 5060 Ti để chỉ GPU là khác:

```
2.11.0+cu128 12.8 NVIDIA GeForce RTX 5070 Ti 2 2.4.6 1.18.0 1.9.0
```

Nếu có phiên bản khác, cài đúng bản:

```bash
export PATH=/venv/main/bin:$PATH && pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && pip install numpy==2.4.6 scipy==1.18.0 scikit-learn==1.9.0
```

Rồi cài phần còn lại. Pip thấy các bản trên đã thoả nên không đụng lại:

```bash
export PATH=/venv/main/bin:$PATH && cd /workspace/ECG_HRV && pip install -q -r requirements.txt awscli && python -c "import wfdb, yaml, neurokit2; print('deps OK')"
```

Ghi lại cấu hình máy vào bài. Đợt trước thiếu model CPU, lần này ghi đủ:

```bash
export PATH=/venv/main/bin:$PATH && { nvidia-smi; lscpu | head -20; nproc; python -c "import torch,numpy,scipy,sklearn;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.device_count());print('numpy',numpy.__version__,'scipy',scipy.__version__,'sklearn',sklearn.__version__)"; pip freeze; } > /workspace/logs/box_5070ti.txt 2>&1 && head -40 /workspace/logs/box_5070ti.txt
```

---

## 3. Dữ liệu thô — PTB-XL **1.0.1**

Dùng mirror AWS, không dùng physionet.org (bị bóp còn khoảng 140 KB/s). Chỉ cần
`records500`, bỏ `records100`:

```bash
export PATH=/venv/main/bin:$PATH && aws configure set default.s3.max_concurrent_requests 64 && cd /workspace/ECG_HRV/data/raw && aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.1/ PTB-XL/ --exclude 'SHA256SUMS.txt' --exclude '*.zip' --exclude 'records100/*'
```

Đối chiếu với `SHA256SUMS.txt` của **bản local** (có trong gói, lệnh sync cố tình không
ghi đè), rồi đếm số bản ghi:

```bash
cd /workspace/ECG_HRV/data/raw/PTB-XL && sha256sum -c SHA256SUMS.txt --quiet --ignore-missing && echo "CHECKSUM OK" && find records500 -name "*_hr.dat" | wc -l
```

Phải in `CHECKSUM OK` và **21837**. `--ignore-missing` chỉ bỏ qua `records100`; con số
21837 bảo đảm `records500` không thiếu file nào. Sai một trong hai thì dừng.

Kiểm tra dữ liệu đã xử lý trong gói:

```bash
cd /workspace/ECG_HRV && export PATH=/venv/main/bin:$PATH && python -c "import numpy as np;print(np.load('data/processed_dir5_imi50_500hz/label_matrix.npy').shape,[len(np.load(f'data/splits_dir5_imi50_500hz/{s}_indices.npy')) for s in ('train','val','test')])"
```

Phải ra `(21837, 13) [17441, 2193, 2203]`.

---

## 4. Script chạy: mỗi GPU một làn, mỗi làn một config

Mỗi làn chạy tuần tự 10 seed trên một GPU. Mỗi seed train xong thì calibrate luôn trên
đúng thư mục run lấy từ log, không đoán thư mục mới nhất.

Hai làn chạy **hai config khác nhau**, nên không bao giờ có hai run cùng config khởi động
cùng giây, tức lỗi trùng tên thư mục đã làm mất 2 run hồi tháng 8.

```bash
cat > /workspace/lane.sh <<'EOF'
#!/bin/bash
# lane.sh GPU CONFIG SEED...   train, then calibrate, each seed in turn on one GPU
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/venv/main/bin:$PATH
cd /workspace/ECG_HRV
g=$1; c=$2; shift 2
for s in "$@"; do
  log=/workspace/logs/${c}_s${s}.txt
  echo "=== START $c seed=$s gpu=$g $(date -u +%T) ==="
  CUDA_VISIBLE_DEVICES=$g python scripts/02_train.py --config configs/experiments/$c.yaml --seed $s > "$log" 2>&1
  rc=$?
  d=$(grep -m1 "Run directory" "$log" | sed 's/.*Run directory : //')
  if [ $rc -ne 0 ] || [ ! -f "$d/best_model.pth" ]; then echo "!!! TRAIN FAIL $c seed=$s rc=$rc dir=$d"; continue; fi
  if CUDA_VISIBLE_DEVICES=$g python scripts/03_calibrate.py --dir "$d" > /workspace/logs/cal_${c}_s${s}.txt 2>&1; then
    echo "=== DONE  $c seed=$s $(date -u +%T) -> $d"
  else
    echo "!!! CALIB FAIL $c seed=$s dir=$d"
  fi
done
echo "LANE $g ($c) FINISHED $(date -u +%T)"
EOF
chmod +x /workspace/lane.sh && echo OK
```

Không đổi `num_workers: 14` và `monitor_metric: imi_auprc` trong config. Chúng quyết định
luồng ngẫu nhiên và epoch được chọn. Máy có 128 CPU nên 2 × 14 worker là dư.

---

## 5. Chạy ARCH-14 và ARCH-13

Mở `tmux` để rớt SSH không mất việc. Thả ra bằng `Ctrl+B` rồi `D`, vào lại bằng
`tmux attach -t arch`:

```bash
tmux new -s arch
```

Trong tmux:

```bash
S="42 123 2024 7 13 99 3407 12345 2025 31415"; /workspace/lane.sh 0 abl_arch14_partial_nogate $S > /workspace/logs/lane0.txt 2>&1 & sleep 5; /workspace/lane.sh 1 abl_arch13_nogate $S > /workspace/logs/lane1.txt 2>&1 & wait; echo ALL DONE
```

**Sau 3–4 phút**, kiểm tra run đầu tiên đã chạy thật (mở tab tmux mới bằng `Ctrl+B` rồi
`C`):

```bash
grep -h "Trainable parameters\|Epoch \[" /workspace/logs/abl_arch1*_s42.txt | cut -c1-60; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

Phải thấy `12,710,833` (ARCH-14) và `13,140,016` (ARCH-13), cùng các dòng `Epoch [  1/15]`.
Lấy thời gian ở cuối dòng epoch nhân 15 là ra thời gian mỗi run. Ước tính khoảng 10–13
phút mỗi run, **khoảng 2 giờ cho cả 20 run**.

Theo dõi tiến độ:

```bash
cat /workspace/logs/lane0.txt /workspace/logs/lane1.txt | grep "DONE\|FAIL\|FINISHED"
```

Xong khi có đủ 20 dòng `DONE`, không có dòng `FAIL`, và 2 dòng `FINISHED`.

Nếu có `FAIL`, xem đuôi log của seed đó:

```bash
tail -20 /workspace/logs/<config>_s<seed>.txt
```

Chạy lại riêng seed đó trên GPU còn rảnh:

```bash
/workspace/lane.sh 0 <config> <seed>
```

---

## 6. Xem nhanh trên máy thuê

```bash
cd /workspace/ECG_HRV && export PATH=/venv/main/bin:$PATH && python scripts/compare_val_runs.py --glob "checkpoints/run_*abl_arch1[34]*" --aggregate
```

Mỗi arm phải có n = 10.

---

## 7. (Tuỳ chọn, khuyến nghị) Chạy đối chứng trên cùng máy

Chỉ cần làm khi muốn bỏ hẳn độ lệch GPU:

```bash
S="42 123 2024 7 13 99 3407 12345 2025 31415"; /workspace/lane.sh 0 abl_arch12_partial $S > /workspace/logs/lane0_ctrl.txt 2>&1 & sleep 5; /workspace/lane.sh 1 abl_arch10_contrast $S > /workspace/logs/lane1_ctrl.txt 2>&1 & wait; echo CONTROLS DONE
```

Chạy xong, so sánh trong nội bộ máy luôn:

```bash
cd /workspace/ECG_HRV && export PATH=/venv/main/bin:$PATH && python scripts/stat_tests_extra.py --glob "checkpoints/run_*" --out /workspace/logs/stats_5070ti_within.md
```

Hai cặp đầu (ARCH-12 − ARCH-14, ARCH-10 − ARCH-13) là kết quả sạch. Hai cặp sau thiếu
ARCH-05/09 trên máy này nên sẽ báo `no shared seeds`. Điều đó là bình thường.

**Nếu để sau mới chạy đối chứng** trên một máy 5070 Ti khác: chạy ARCH-14 seed 42 ở máy đó
trước, rồi so với run ARCH-14 seed 42 của lần này:

```bash
python scripts/verify_box.py --new <run mới> --ref <run ARCH-14 seed 42 của lần này>
```

Chỉ khi ra `bit-identical` mới được ghép. Vì thế **giữ lại toàn bộ `history.json` và
`config_snapshot.yaml` của lần này** (mục 8 đã làm việc đó).

---

## 8. Kéo kết quả về

Trên máy thuê, đóng gói. Bỏ `.pth` (khoảng 155 MB mỗi file):

```bash
cd /workspace/ECG_HRV && tar -czf /workspace/results_5070ti.tgz --exclude='*.pth' checkpoints && tar -czf /workspace/logs_5070ti.tgz -C /workspace logs && ls -lh /workspace/*.tgz
```

Ở máy local (cmd):

```bat
cd /d "E:\KLTN\KL Project" && scp -P <PORT> root@<HOST>:/workspace/results_5070ti.tgz root@<HOST>:/workspace/logs_5070ti.tgz .
```

Giải nén vào **thư mục con riêng** `checkpoints\box_5070ti\`, **không** vào `checkpoints\`:

```bat
cd /d "E:\KLTN\KL Project\ECG_HRV" && mkdir checkpoints\box_5070ti && tar -xzf ..\results_5070ti.tgz -C checkpoints\box_5070ti --strip-components=1 && tar -xzf ..\logs_5070ti.tgz -C checkpoints\box_5070ti && dir checkpoints\box_5070ti
```

> Lý do: nếu có chạy đối chứng, run ARCH-12/ARCH-10 mới trùng `(exp_id, seed)` với run
> trong bài. `stat_tests.py` và `aggregate_seeds.py` giữ run **mới nhất**, nên sẽ âm thầm
> thay số 5060 Ti trong bài bằng số 5070 Ti. Để ở thư mục con thì glob mặc định
> `checkpoints/run_*` không chạm tới. `checkpoints/` đã nằm trong `.gitignore`.

Muốn giữ checkpoint để phân tích sau thì kéo thêm (khoảng 3 GB cho 20 run):

```bat
cd /d "E:\KLTN\KL Project\ECG_HRV\checkpoints" && scp -P <PORT> -r root@<HOST>:/workspace/ECG_HRV/checkpoints box_5070ti_pth
```

**Xác nhận đã kéo về đủ rồi mới Destroy instance.**

---

## 9. Kiểm định ở máy local

**Có chạy đối chứng (mục 7), so trong cùng máy.** Đây là kết quả để đưa vào bài:

```bat
cd /d "E:\KLTN\KL Project\ECG_HRV" && python scripts/stat_tests_extra.py --glob "checkpoints/box_5070ti/run_*" --out docs/stat_tests_contrast_5070ti.md
```

**Không chạy đối chứng, so khác máy.** Chỉ để tham khảo, phải ghi rõ lẫn độ lệch GPU:

```bat
cd /d "E:\KLTN\KL Project\ECG_HRV" && python scripts/stat_tests_extra.py --glob "checkpoints/box_5070ti/run_*abl_arch1[34]*" --glob "checkpoints/run_*abl_arch*" --out docs/stat_tests_contrast_crossbox.md
```

Script tự chặn nếu cùng một `(exp_id, seed)` xuất hiện ở hai thư mục, nên không thể vô
tình trộn run hai máy của cùng một arm.

Chiều dấu: **delta dương = arm CÓ khối đối lập cao hơn.** Script báo cho IMI, ILMI, ASMI,
AMI, MI macro và S.

---

## 10. Đọc kết quả (cặp cùng máy)

| ARCH-12 − ARCH-14 | ARCH-10 − ARCH-13 | Kết luận cho bài |
|---|---|---|
| > 0, CI không chứa 0 | ≈ 0 | Khối đối lập **có ích** khi token dùng chung. Khi tách hoàn toàn, bố cục token cụm đã làm thay việc đó ("hai đường thay thế nhau") |
| ≈ 0 | ≈ 0 | Mô hình dựa vào khối đối lập (ARCH-12) nhưng không cần nó. Đóng góp nằm ở bố cục token, không ở khối đối lập |
| > 0 | > 0 | Khối đối lập có ích ở cả hai kiến trúc. Giữ nguyên cách trình bày của bài |

Ngưỡng thực tế: CI nằm trong ±0,01 thì coi là không khác biệt, vì nhỏ hơn độ lệch giữa
các GPU.

---

## Nhắc lại chỗ dễ sai

| | |
|---|---|
| `export PATH=/venv/main/bin:$PATH` | Mọi lệnh Python trên máy thuê |
| PTB-XL **1.0.1** | Không phải 1.0.3. Checksum và số 21837 phải đúng |
| Không chạy `01_build_metadata.py` | Dùng `processed` và `splits` có trong gói |
| `03_calibrate.py --dir` | Không phải `--checkpoint`. `lane.sh` đã làm sẵn |
| Giải nén vào `checkpoints/box_5070ti/` | Không vào `checkpoints/` |
