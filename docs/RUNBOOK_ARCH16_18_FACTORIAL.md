# Runbook — ARCH-16/17/18: hoàn tất hai bảng 2×2 (bố cục token × khối đối lập)

## Mục tiêu

Hai con đường dẫn tới mức tăng IMI mới được thử, mỗi con đường trong **một** kiểu định tuyến:
- bố cục token theo cụm: chỉ thử khi tách hoàn toàn
- khối đối lập: chỉ thử khi dùng chung backbone

Đợt này bổ sung các ô còn thiếu để có hai bảng 2×2 đầy đủ.

| | Không có gì | Chỉ bố cục | Chỉ đối lập | Cả hai |
|---|---|---|---|---|
| **Tách hoàn toàn** | ARCH-09 | ARCH-13 | **ARCH-16** (mới) | ARCH-10 |
| **Dùng chung backbone** | ARCH-14 | **ARCH-17** (mới) | ARCH-12 | **ARCH-18** (mới) |

Các arm mới:
- **ARCH-16** `abl_arch16_decoupled_contrast` = ARCH-09 + khối đối lập có cổng (giống ARCH-10).
  - Nhánh IMI **không nhận đạo trình V5–V6**.
  - Nếu ARCH-16 vẫn loại được ILMI thì thông tin thành bên chỉ có thể đi vào qua δ = x − x_ILMI.
  - Đây là phép thử trực tiếp cho giả thuyết "δ là kênh truyền".
- **ARCH-17** `abl_arch17_partial_cluster` = ARCH-14 nhưng phần giải phẫu theo bố cục cụm của
  contrast_v2:
  - hai cụm token [inferior, reciprocal, lateral, inferolateral] và [anterior, lateral, anterior6]
  - một lớp transformer cho mỗi cụm
  - mỗi nhãn pool cả cụm bằng query riêng
- **ARCH-18** `abl_arch18_partial_cluster_contrast` = ARCH-17 + khối đối lập có cổng.

Cả 8 arm chạy trên **cùng một máy**, 10 seed mỗi arm, tổng **80 run**. Các so sánh và tương tác
đều ghép cặp trong cùng máy.

**Cách đọc kết quả** (IMI AUPRC, validation):

| Cặp | Nếu > 0 có ý nghĩa | Nếu ≈ 0 |
|---|---|---|
| ARCH-16 − 09 | Khối đối lập tự đứng được khi tách hoàn toàn | Khối đối lập chỉ có tác dụng khi dùng chung backbone |
| ARCH-17 − 14 | Bố cục token cũng có tác dụng khi dùng chung backbone | Bố cục chỉ có tác dụng khi tách hoàn toàn |
| ARCH-10 − 16, ARCH-18 − 17, ARCH-18 − 12 | Hai con đường cộng dồn | Hai con đường thay thế nhau (đúng như kết luận hiện tại) |

Phép che đạo trình (chạy ngay trên máy thuê) cho biết:
- Ở ARCH-16, đầu IMI có phụ thuộc vào V5–V6 hay không.
- ARCH-14 là đối chứng cho phép che ARCH-12: có δ thì phụ thuộc V5–V6, không có δ thì không.

## Đã kiểm tra ở máy local (02/11/2026)

| Kiểm tra | Kết quả |
|---|---|
| ARCH-05/09/10/12 cũ load `strict=True` sau khi sửa code; IMI AUPRC test seed 42 | trùng số cũ tới 6 chữ số (0,587462 / 0,592521 / 0,652363 / 0,648754) |
| Số tham số | ARCH-16: 13 391 312 (= ARCH-09 + 98 816); ARCH-17: 13 484 561 (ARCH-14 + 773 728); ARCH-18: 13 583 377 |
| Khoá state_dict | ARCH-16 = ARCH-09 + 2 khối đối lập; ARCH-18 = ARCH-17 + 2 khối đối lập |
| Forward/backward | Chạy được. Tham số không nhận gradient đều đã như vậy ở arm gốc |
| `02_train.py --debug` + `03_calibrate.py`, cả 3 arm | exit 0, đủ `best_model.pth`, `calibration_results.json`, `class_report.json`. Lưu ý: chế độ debug chỉ lấy 50 bản ghi validation đầu tiên, không có ca IMI nào, nên lúc thử phải đặt monitor tạm sang `auroc/mi/macro`. Lần chạy thật dùng `imi_auprc` như các arm khác |
| `24_sibling_occlusion.py --arm …` trên run debug của 3 arm | chạy hết, ra đủ 76 chỉ số mỗi run |

**Lưu ý về tham số:** ARCH-17 nhiều hơn ARCH-14 khoảng 6%, vì phải thêm encoder và
cross-attention cho token inferolateral cùng hai transformer theo cụm. ARCH-13 so với ARCH-09 cũng
có chênh lệch kiểu này. Cần ghi rõ trong bài.

## File cần đẩy lên

- `E:\KLTN\KL Project\ecg_box_d.tar.gz`: code mới, config, dữ liệu đã xử lý (~5 MB).

---

## 1. Thuê máy

- **GPU:** RTX 5070 Ti, càng nhiều càng nhanh. Chi phí gần như không đổi theo số GPU.
- **CPU:** ít nhất 16 vCPU **cho mỗi GPU** (mỗi run dùng 14 worker đọc dữ liệu).
- **Ổ đĩa:** ≥ 60 GB (PTB-XL ~ 3 GB, mỗi `best_model.pth` ~ 155 MB × 80 = 12 GB).

Ước lượng, với khoảng 12 phút cho một run:

| Máy | Thời gian | Tiền (~1,4 $/giờ cho 2 GPU) |
|---|---|---|
| 2 × 5070 Ti | ~8 giờ | ~11–12 $ |
| 4 × 5070 Ti | ~4 giờ | ~11–12 $ |

Lấy `<PORT>` và `<HOST>` ở nút **Connect** của instance.

## 2. Đẩy file lên (PowerShell ở máy bạn)

```powershell
scp -P <PORT> "E:\KLTN\KL Project\ecg_box_d.tar.gz" root@<HOST>:/workspace/
```

```powershell
ssh -p <PORT> root@<HOST>
```

## 3. Dựng môi trường (trên máy thuê)

```bash
mkdir -p /workspace/ECG_HRV /workspace/logs && tar -xzf /workspace/ecg_box_d.tar.gz -C /workspace/ECG_HRV && ls /workspace/ECG_HRV
```

```bash
export PATH=/venv/main/bin:$PATH && pip install -q numpy==2.4.6 scipy==1.18.0 scikit-learn==1.9.0 && cd /workspace/ECG_HRV && pip install -q -r requirements.txt awscli && pip install -q numpy==2.4.6 scipy==1.18.0 scikit-learn==1.9.0 && python -c "import torch,numpy,scipy,sklearn,wfdb,neurokit2;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.device_count(), numpy.__version__, scipy.__version__, sklearn.__version__)"
```

Phải ra `2.11.0+cu128 12.8 <tên GPU> <số GPU> 2.4.6 1.18.0 1.9.0`. Nếu torch là bản khác:

```bash
export PATH=/venv/main/bin:$PATH && pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Ghi lại cấu hình máy:

```bash
export PATH=/venv/main/bin:$PATH && { nvidia-smi; lscpu | head -20; nproc; pip freeze; } > /workspace/logs/box_d.txt 2>&1 && grep "Driver Version" /workspace/logs/box_d.txt && grep "Model name" /workspace/logs/box_d.txt && nproc
```

Tải PTB-XL 1.0.1 (chỉ `records500`):

```bash
export PATH=/venv/main/bin:$PATH && aws configure set default.s3.max_concurrent_requests 64 && cd /workspace/ECG_HRV/data/raw && aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.1/ PTB-XL/ --exclude 'SHA256SUMS.txt' --exclude '*.zip' --exclude 'records100/*'
```

```bash
cd /workspace/ECG_HRV/data/raw/PTB-XL && sha256sum -c SHA256SUMS.txt --quiet --ignore-missing && echo "CHECKSUM OK" && find records500 -name "*_hr.dat" | wc -l
```

Phải in `CHECKSUM OK` và `21837`.

## 4. Hàng đợi việc và khởi chạy

Lần này không chia làn cố định. Mỗi GPU có một worker tự lấy việc tiếp theo từ chung một danh
sách, nên máy 2 GPU hay 4 GPU đều dùng cùng một bộ lệnh.

Danh sách 80 việc, xếp theo seed để các bộ 8 arm hoàn chỉnh xong dần:

```bash
for s in 42 123 2024 7 13 99 3407 12345 2025 31415; do for c in abl_arch16_decoupled_contrast abl_arch09_decoupled abl_arch17_partial_cluster abl_arch14_partial_nogate abl_arch18_partial_cluster_contrast abl_arch12_partial abl_arch13_nogate abl_arch10_contrast; do echo "$c $s"; done; done > /workspace/jobs.txt && wc -l < /workspace/jobs.txt
```

Phải in `80`.

Script lấy việc và script worker:

```bash
cat > /workspace/pop.sh <<'EOF'
#!/bin/bash
# pop.sh   print and remove the first job, under a lock shared by all workers
exec 9>/workspace/queue.lock
flock 9
l=$(head -n1 /workspace/jobs.txt)
[ -n "$l" ] && sed -i '1d' /workspace/jobs.txt
echo "$l"
EOF
cat > /workspace/worker.sh <<'EOF'
#!/bin/bash
# worker.sh GPU   take jobs until the queue is empty: train, then calibrate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/venv/main/bin:$PATH
cd /workspace/ECG_HRV
g=$1
while true; do
  job=$(/workspace/pop.sh)
  [ -z "$job" ] && break
  set -- $job; c=$1; s=$2
  log=/workspace/logs/${c}_s${s}.txt
  flock /workspace/launch.lock sleep 5
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
echo "WORKER $g FINISHED $(date -u +%T)"
EOF
chmod +x /workspace/pop.sh /workspace/worker.sh && echo OK
```

Khởi chạy một worker cho mỗi GPU:

```bash
N=$(nvidia-smi -L | wc -l); for g in $(seq 0 $((N-1))); do nohup /workspace/worker.sh $g > /workspace/logs/worker$g.txt 2>&1 & done; echo "LAUNCHED $N workers"
```

Sau 3–4 phút, kiểm tra số tham số và GPU:

```bash
grep -h "Trainable parameters" /workspace/logs/abl_*_s42.txt; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

Phải thấy các số trong danh sách sau (chỉ những run đã bắt đầu):

| Arm | Số tham số |
|---|---|
| ARCH-16 | `13,391,312` |
| ARCH-09 | `13,292,496` |
| ARCH-17 | `13,484,561` |
| ARCH-14 | `12,710,833` |
| ARCH-18 | `13,583,377` |
| ARCH-12 | `12,809,649` |
| ARCH-13 | `13,140,016` |
| ARCH-10 | `13,238,832` |

## 5. Theo dõi

Có thể đóng SSH; `nohup` giữ các worker chạy tiếp.

```bash
echo "DONE: $(cat /workspace/logs/worker*.txt | grep -c '=== DONE')/80  QUEUE: $(wc -l < /workspace/jobs.txt)  RUNNING: $(pgrep -fc '02_train.py|03_calibrate.py')"; grep -h "FAIL\|FINISHED" /workspace/logs/worker*.txt
```

Xem từng epoch của các run đang chạy:

```bash
for f in $(ls -t /workspace/logs/abl_*_s*.txt | head -2); do echo "== $(basename $f)"; tr '\r' '\n' < "$f" | grep "Epoch \[" | cut -c1-150; done
```

**Xong khi** dòng đầu in `DONE: 80/80  QUEUE: 0  RUNNING: 0`, không có dòng `FAIL`, và số dòng
`WORKER … FINISHED` bằng số GPU.

Nếu có dòng `FAIL`: báo tôi kèm tên run, **đừng** chạy lại thủ công.

## 6. Phép che đạo trình ngay trên máy thuê

Chạy trên máy thuê để khỏi tải 12 GB trọng số về. Chia các arm cho 2 GPU:

```bash
cd /workspace/ECG_HRV && export PATH=/venv/main/bin:$PATH && CUDA_VISIBLE_DEVICES=0 nohup python scripts/24_sibling_occlusion.py /workspace/occ_d_a.csv --arm ARCH-09=abl_arch09_decoupled --arm ARCH-16=abl_arch16_decoupled_contrast --arm ARCH-13=abl_arch13_nogate --arm ARCH-10=abl_arch10_contrast > /workspace/logs/occ_a.txt 2>&1 & CUDA_VISIBLE_DEVICES=1 nohup python scripts/24_sibling_occlusion.py /workspace/occ_d_b.csv --arm ARCH-14=abl_arch14_partial_nogate --arm ARCH-17=abl_arch17_partial_cluster --arm ARCH-12=abl_arch12_partial --arm ARCH-18=abl_arch18_partial_cluster_contrast > /workspace/logs/occ_b.txt 2>&1 & echo LAUNCHED
```

Mất khoảng 20–30 phút. Xong khi:

```bash
grep -c "done" /workspace/logs/occ_a.txt /workspace/logs/occ_b.txt; tail -1 /workspace/logs/occ_a.txt /workspace/logs/occ_b.txt
```

Hai dòng đầu phải là `occ_a.txt:40` và `occ_b.txt:40`. Hai dòng `saved /workspace/occ_d_*.csv`
phải có mặt.

## 7. Đóng gói và tải về

```bash
cd /workspace/ECG_HRV && tar -czf /workspace/results_d.tgz --exclude='*.pth' checkpoints && tar -czf /workspace/logs_d.tgz -C /workspace logs occ_d_a.csv occ_d_b.csv && ls -lh /workspace/*_d.tgz && tar -tzf /workspace/results_d.tgz | grep -c history.json
```

Dòng cuối phải là **80**.

Tuỳ chọn: giữ trọng số seed 42 của 3 arm mới (~0,5 GB) để làm IG hay phân tích thêm sau này:

```bash
cd /workspace/ECG_HRV && tar -czf /workspace/weights_d_s42.tgz $(for c in arch16 arch17 arch18; do d=$(grep -lE "^  seed: 42\s*$" checkpoints/run_*abl_${c}_*/config_snapshot.yaml | head -1 | xargs dirname); echo $d/best_model.pth; done) && ls -lh /workspace/weights_d_s42.tgz
```

PowerShell ở máy bạn:

```powershell
scp -P <PORT> root@<HOST>:/workspace/results_d.tgz root@<HOST>:/workspace/logs_d.tgz "E:\KLTN\KL Project"
```

Báo tôi khi tải xong. Tôi sẽ:
1. Giải nén vào `checkpoints\box_5070ti_d\` và kiểm tra đủ 8 arm × 10 seed.
2. Chạy `stat_tests_extra.py --glob "checkpoints/box_5070ti_d/run_*"` cho hai bảng 2×2.
3. Đọc kết quả che đạo trình.

**Chỉ Destroy máy sau khi tôi xác nhận đủ run.**
