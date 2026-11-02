# Runbook — ARCH-15: khối đối lập không cổng

## Mục tiêu

Kiểm chứng riêng **cái cổng** trong khối đối lập. ARCH-13 và ARCH-14 đã bật/tắt **cả khối**
(cổng + phép chiếu W_p); chưa có thí nghiệm nào tách riêng cái cổng.

| Arm | Khối đối lập | Công thức |
|---|---|---|
| ARCH-14 `abl_arch14_partial_nogate` | không có | \(\tilde c_a=c_a\) |
| **ARCH-15** `abl_arch15_partial_ungated` (mới) | có, **không cổng** | \(\tilde c_a=c_a+W_p\delta+b_p\) |
| ARCH-12 `abl_arch12_partial` | có, có cổng | \(\tilde c_a=c_a+g\odot(W_p\delta+b_p)\) |

Cả ba dùng chung backbone (tách một phần). ARCH-15 giữ nguyên khởi tạo W_p = b_p = 0 của
ARCH-12 và chỉ bỏ lớp cổng: ít hơn **65 792** tham số (12 809 649 → 12 743 857).

**Cách đọc kết quả** (cặp ARCH-12 − ARCH-15, IMI AUPRC):

| Kết quả | Kết luận |
|---|---|
| ≈ 0 (CI trong ±0,01) | Cổng không cần; phần có tác dụng là phép hiệu được chiếu \(W_p\delta\) |
| > 0, CI không chứa 0 | Cổng có vai trò thật; giữ tên "gated", bài có thêm bằng chứng |
| < 0 | Cổng làm hại; cách viết của bài phải đổi mạnh |

Cặp ARCH-15 − ARCH-14 cho biết khối đối lập không cổng có tự đứng được không.

## Đã kiểm tra ở máy local (29/10/2026)

| Kiểm tra | Kết quả |
|---|---|
| ARCH-12 cũ load `strict=True` sau khi sửa code; IMI AUPRC seed 42 / 123 | 0,638528 / 0,634881, khớp số cũ |
| ARCH-10 cũ load `strict=True`; IMI AUPRC seed 42 / 123 | 0,668498 / 0,654662, khớp số cũ |
| ARCH-15: số tham số; lớp `gate` không còn, lớp `proj` vẫn có | 12 743 857, ít hơn đúng 65 792 |
| `02_train.py --debug` + `03_calibrate.py --dir` với ARCH-15 | exit 0, ra đủ `calibration_results.json` và `class_report.json` |
| `stat_tests_extra.py` đã có 2 cặp mới | ARCH-12 − ARCH-15, ARCH-15 − ARCH-14 |

## Hai file cần đẩy lên (đã tạo sẵn)

- `E:\KLTN\KL Project\ecg_box.tar.gz`: code, config, dữ liệu đã xử lý (khoảng 5 MB).
- `E:\KLTN\KL Project\ref_arch12_boxB.tgz`: run ARCH-12 seed 42 của **máy B** (máy đã chạy 60
  run hôm 30/09), dùng để nghiệm thu.

---

## 1. Thuê máy

Bất kỳ máy nhiều GPU nào cũng dùng được. Nếu muốn có cơ hội đi **nhánh A** (nhanh và rẻ hơn,
xem mục 5), chọn máy giống máy B nhất có thể:

| | Máy B |
|---|---|
| GPU | 2 × RTX 5070 Ti |
| Driver | 595.99.02 (CUDA 13.2) |
| CPU | AMD EPYC 7B13, 256 luồng |

Lấy `<PORT>` và `<HOST>` ở nút **Connect** của instance.

## 2. Đẩy file lên (PowerShell ở máy bạn)

```powershell
scp -P <PORT> "E:\KLTN\KL Project\ecg_box.tar.gz" "E:\KLTN\KL Project\ref_arch12_boxB.tgz" root@<HOST>:/workspace/
```

```powershell
ssh -p <PORT> root@<HOST>
```

## 3. Dựng môi trường (trên máy thuê)

```bash
mkdir -p /workspace/ECG_HRV /workspace/logs /workspace/ref /workspace/verify_runs && tar -xzf /workspace/ecg_box.tar.gz -C /workspace/ECG_HRV && tar -xzf /workspace/ref_arch12_boxB.tgz -C /workspace/ref && ls /workspace/ECG_HRV /workspace/ref
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
export PATH=/venv/main/bin:$PATH && { nvidia-smi; lscpu | head -20; nproc; pip freeze; } > /workspace/logs/box_c.txt 2>&1 && grep "Driver Version" /workspace/logs/box_c.txt && grep "Model name" /workspace/logs/box_c.txt
```

Tải PTB-XL 1.0.1 (chỉ `records500`):

```bash
export PATH=/venv/main/bin:$PATH && aws configure set default.s3.max_concurrent_requests 64 && cd /workspace/ECG_HRV/data/raw && aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.1/ PTB-XL/ --exclude 'SHA256SUMS.txt' --exclude '*.zip' --exclude 'records100/*'
```

```bash
cd /workspace/ECG_HRV/data/raw/PTB-XL && sha256sum -c SHA256SUMS.txt --quiet --ignore-missing && echo "CHECKSUM OK" && find records500 -name "*_hr.dat" | wc -l
```

Phải in `CHECKSUM OK` và `21837`.

## 4. Script làn và khởi chạy pha 1

Khóa `flock` bảo đảm mọi lần khởi chạy (kể cả từ hai GPU) cách nhau ít nhất 5 giây. Nhờ vậy hai
run cùng config không bao giờ trùng tên thư mục.

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
echo "LANE $g ($c) FINISHED $(date -u +%T)"
EOF
chmod +x /workspace/lane.sh && echo OK
```

Pha 1 chạy hai việc cùng lúc:
- **GPU 0:** chạy lại ARCH-12 seed 42 để nghiệm thu, mất khoảng 12 phút.
- **GPU 1:** bắt đầu ngay 5 seed đầu của ARCH-15. Cả hai nhánh đều cần các run này.

```bash
nohup /workspace/lane.sh 0 abl_arch12_partial 42 > /workspace/logs/lane0_verify.txt 2>&1 & nohup /workspace/lane.sh 1 abl_arch15_partial_ungated 42 123 2024 7 13 > /workspace/logs/lane1.txt 2>&1 & echo LAUNCHED
```

Sau 3–4 phút, kiểm tra số tham số và GPU:

```bash
grep -h "Trainable parameters" /workspace/logs/*_s42.txt; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

Phải thấy `12,809,649` (ARCH-12) và `12,743,857` (ARCH-15).

## 5. ⛔ Nghiệm thu, khi `lane0_verify.txt` có dòng `FINISHED`

```bash
cat /workspace/logs/lane0_verify.txt; cd /workspace/ECG_HRV && export PATH=/venv/main/bin:$PATH && python scripts/verify_box.py --new "$(ls -d checkpoints/run_*abl_arch12_partial* | head -1)" --ref /workspace/ref/run_20260930_020011_abl_arch12_partial-multi_branch-tf-aug
```

### Nhánh A: `VERDICT: bit-identical` (hoặc lệch trong 1e-6)

Máy này cho kết quả trùng máy B, nên ARCH-12 và ARCH-14 của máy B dùng lại được. Chỉ cần đủ 10
seed ARCH-15.

Chuyển run nghiệm thu ra ngoài (nó là bản sao của run máy B), rồi cho GPU 0 chạy 5 seed còn lại
của ARCH-15:

```bash
mv /workspace/ECG_HRV/checkpoints/run_*abl_arch12_partial* /workspace/verify_runs/ && nohup /workspace/lane.sh 0 abl_arch15_partial_ungated 99 3407 12345 2025 31415 > /workspace/logs/lane0.txt 2>&1 & echo LAUNCHED
```

Tổng khoảng **1 giờ 15 phút**.

**Xong khi:**
- `grep -c "=== DONE" /workspace/logs/lane*.txt` cho `lane0.txt:5`, `lane1.txt:5`, `lane0_verify.txt:1`;
- có 3 dòng `FINISHED`;
- `pgrep -fc "02_train.py|03_calibrate.py"` in `0`.

### Nhánh B: `VERDICT: DIVERGED`

Phải chạy cả ARCH-12, ARCH-14 và ARCH-15 trên máy này. Run nghiệm thu **giữ lại**: nó chính là
ARCH-12 seed 42 của máy này.

GPU 0 chạy 9 seed còn lại của ARCH-12, rồi 5 seed ARCH-14:

```bash
nohup bash -c "/workspace/lane.sh 0 abl_arch12_partial 123 2024 7 13 99 3407 12345 2025 31415; /workspace/lane.sh 0 abl_arch14_partial_nogate 42 123 2024 7 13" > /workspace/logs/lane0.txt 2>&1 & echo LAUNCHED
```

GPU 1 tự chờ 5 seed ARCH-15 đầu xong, rồi chạy 5 seed ARCH-15 còn lại và 5 seed ARCH-14 còn lại:

```bash
nohup bash -c "while pgrep -f '[l]ane.sh 1 abl_arch15_partial_ungated 42' > /dev/null; do sleep 60; done; /workspace/lane.sh 1 abl_arch15_partial_ungated 99 3407 12345 2025 31415; /workspace/lane.sh 1 abl_arch14_partial_nogate 99 3407 12345 2025 31415" > /workspace/logs/lane1b.txt 2>&1 & echo LAUNCHED
```

Tổng khoảng **3 giờ**.

**Xong khi** `grep -c "=== DONE" /workspace/logs/lane*.txt` cho `lane0_verify.txt:1`,
`lane0.txt:14`, `lane1.txt:5`, `lane1b.txt:10` (tổng 30), có 5 dòng `FINISHED`, và
`pgrep -fc "02_train.py|03_calibrate.py"` in `0`.

## 6. Theo dõi

Có thể đóng SSH; `nohup` giữ các làn chạy tiếp.

```bash
grep -c "=== DONE" /workspace/logs/lane*.txt; grep -h "FAIL\|FINISHED" /workspace/logs/lane*.txt
```

Xem từng epoch của các run đang chạy:

```bash
for f in $(ls -t /workspace/logs/abl_*_s*.txt | head -2); do echo "== $(basename $f)"; tr '\r' '\n' < "$f" | grep "Epoch \[" | cut -c1-150; done
```

## 7. Đóng gói và tải về — chỉ khi đã đạt điều kiện "xong"

```bash
cd /workspace/ECG_HRV && tar -czf /workspace/results_c.tgz --exclude='*.pth' checkpoints && tar -czf /workspace/logs_c.tgz --exclude='*.pth' -C /workspace logs verify_runs && ls -lh /workspace/*_c.tgz && tar -tzf /workspace/results_c.tgz | grep -c history.json
```

Dòng cuối phải là **10** (nhánh A) hoặc **30** (nhánh B).

PowerShell ở máy bạn:

```powershell
scp -P <PORT> root@<HOST>:/workspace/results_c.tgz root@<HOST>:/workspace/logs_c.tgz "E:\KLTN\KL Project"
```

Báo tôi khi tải xong. Tôi giải nén vào `checkpoints\box_5070ti_c\`, kiểm tra từng run rồi chạy
kiểm định:
- Nhánh A: ghép với máy B (`--glob checkpoints/box_5070ti_b/run_* --glob checkpoints/box_5070ti_c/run_*`).
- Nhánh B: chỉ dùng máy C.

**Chỉ Destroy máy sau khi tôi xác nhận đủ run.**
