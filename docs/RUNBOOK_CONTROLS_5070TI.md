# Runbook — đối chứng trên cùng máy 5070 Ti

Mục tiêu: đưa **cả 4 cặp so sánh** của ARCH-13/14 về cùng một phần cứng, để chúng thành
so sánh ghép cặp sạch, không lẫn độ lệch do đổi GPU.

| Arm | Seed | Vì sao |
|---|---|---|
| ARCH-14 seed 42 (chạy lại) | 42 | **Nghiệm thu máy**: phải trùng từng bit với run hôm 29/09 |
| ARCH-14 | 31415 | Seed bị mất hôm 29/09, cho đủ n = 10 |
| ARCH-12 `abl_arch12_partial` | 10 seed | Đối chứng cho ARCH-14 |
| ARCH-10 `abl_arch10_contrast` | 10 seed | Đối chứng cho ARCH-13 |
| ARCH-09 `abl_arch09_decoupled` | 10 seed | Đối chứng cho ARCH-13 (bố cục token) |
| ARCH-05 `abl_arch05_shared_tf` | 10 seed | Đối chứng cho ARCH-14 (projection + head) |

Tổng cộng 42 run trên 2 GPU, **khoảng 4 giờ, khoảng 4,5 USD** ở mức 1,104 USD/giờ.

Máy hôm 29/09 (lưu ở `checkpoints/box_5070ti/logs/box_5070ti.txt`): 2 × RTX 5070 Ti,
**driver 595.91.07**, CUDA 13.2, torch 2.11.0+cu128, numpy 2.4.6, scipy 1.18.0,
sklearn 1.9.0.

Các lệnh **ở máy bạn** viết cho **PowerShell**. Các lệnh **trên máy thuê** viết cho bash, chạy
trong cửa sổ SSH.

---

## 1. Đẩy code và file tham chiếu lên (PowerShell ở máy bạn)

Hai file đã có sẵn:
- `E:\KLTN\KL Project\ecg_box.tar.gz`: code và dữ liệu đã xử lý, như hôm 29/09.
- `E:\KLTN\KL Project\ref_5070ti.tgz`: `history.json` và `config_snapshot.yaml` của run ARCH-14
  seed 42 hôm 29/09, dùng để nghiệm thu.

Đổi `<PORT>` và `<HOST>` theo nút **Connect** của instance mới (có thể khác hôm qua):

```powershell
scp -P <PORT> "E:\KLTN\KL Project\ecg_box.tar.gz" "E:\KLTN\KL Project\ref_5070ti.tgz" root@<HOST>:/workspace/
```

```powershell
ssh -p <PORT> root@<HOST>
```

---

## 2. Dựng môi trường (trên máy thuê)

```bash
apt-get update -qq && apt-get install -y -qq tmux && mkdir -p /workspace/ECG_HRV /workspace/logs /workspace/ref /workspace/verify_runs && tar -xzf /workspace/ecg_box.tar.gz -C /workspace/ECG_HRV && tar -xzf /workspace/ref_5070ti.tgz -C /workspace/ref && ls /workspace/ECG_HRV /workspace/ref
```

```bash
export PATH=/venv/main/bin:$PATH && pip install numpy==2.4.6 scipy==1.18.0 scikit-learn==1.9.0 && cd /workspace/ECG_HRV && pip install -q -r requirements.txt awscli && pip install -q numpy==2.4.6 scipy==1.18.0 scikit-learn==1.9.0 && python -c "import torch,numpy,scipy,sklearn,wfdb,neurokit2;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.device_count(), numpy.__version__, scipy.__version__, sklearn.__version__)"
```

Phải ra đúng:

```
2.11.0+cu128 12.8 NVIDIA GeForce RTX 5070 Ti 2 2.4.6 1.18.0 1.9.0
```

Nếu torch là bản khác, cài lại đúng bản rồi chạy lại lệnh trên:

```bash
export PATH=/venv/main/bin:$PATH && pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Ghi lại cấu hình máy, rồi so driver với hôm 29/09:

```bash
export PATH=/venv/main/bin:$PATH && { nvidia-smi; lscpu | head -20; nproc; pip freeze; } > /workspace/logs/box_5070ti_b.txt 2>&1 && grep "Driver Version" /workspace/logs/box_5070ti_b.txt
```

Hôm 29/09 là `595.91.07`. Nếu khác, vẫn chạy tiếp; bước nghiệm thu ở mục 4 sẽ cho biết có
ảnh hưởng hay không.

---

## 3. Dữ liệu thô PTB-XL 1.0.1

```bash
export PATH=/venv/main/bin:$PATH && aws configure set default.s3.max_concurrent_requests 64 && cd /workspace/ECG_HRV/data/raw && aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.1/ PTB-XL/ --exclude 'SHA256SUMS.txt' --exclude '*.zip' --exclude 'records100/*'
```

```bash
cd /workspace/ECG_HRV/data/raw/PTB-XL && sha256sum -c SHA256SUMS.txt --quiet --ignore-missing && echo "CHECKSUM OK" && find records500 -name "*_hr.dat" | wc -l
```

Phải in `CHECKSUM OK` và `21837`.

---

## 4. Script làn và khởi chạy

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

Khởi chạy cùng lúc hai việc:
- **GPU 1** bắt đầu luôn chuỗi ARCH-12 rồi ARCH-09. Hai arm này có ích dù nghiệm thu
  thành công hay thất bại.
- **GPU 0** chạy lại ARCH-14 seed 42 để nghiệm thu.

```bash
S="42 123 2024 7 13 99 3407 12345 2025 31415"; nohup bash -c "/workspace/lane.sh 1 abl_arch12_partial $S; /workspace/lane.sh 1 abl_arch09_decoupled $S" > /workspace/logs/lane1.txt 2>&1 & sleep 5; nohup /workspace/lane.sh 0 abl_arch14_partial_nogate 42 > /workspace/logs/lane0_verify.txt 2>&1 & echo LAUNCHED
```

Hai GPU không bao giờ chạy cùng một config cùng lúc, nên không có nguy cơ trùng tên thư mục.

Sau 3–4 phút kiểm tra:

```bash
grep -h "Trainable parameters" /workspace/logs/*_s42.txt; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

Phải thấy `12,710,833` (ARCH-14) và `12,809,649` (ARCH-12), cả 2 GPU đều đang chạy.

---

## 5. ⛔ Nghiệm thu (khoảng 12 phút sau khi khởi chạy)

Đợi đến khi `lane0_verify.txt` có dòng `FINISHED`:

```bash
cat /workspace/logs/lane0_verify.txt
```

So với run hôm 29/09:

```bash
cd /workspace/ECG_HRV && export PATH=/venv/main/bin:$PATH && python scripts/verify_box.py --new "$(ls -d checkpoints/run_*abl_arch14_partial_nogate* | head -1)" --ref /workspace/ref/run_20260929_090930_abl_arch14_partial_nogate-multi_branch-tf-aug
```

Rồi **luôn** chuyển run nghiệm thu ra khỏi `checkpoints/`. Nó là bản sao của seed 42, không
được lọt vào gói kết quả:

```bash
mv /workspace/ECG_HRV/checkpoints/run_*abl_arch14_partial_nogate* /workspace/verify_runs/ && ls /workspace/verify_runs
```

### 5a. Nếu ra `VERDICT: bit-identical`, hoặc lệch trong 1e-6

Máy cho kết quả trùng với hôm 29/09, nên ARCH-13/14 cũ được dùng lại nguyên vẹn. Cho GPU 0
chạy tiếp ARCH-14 seed 31415, ARCH-10 rồi ARCH-05:

```bash
S="42 123 2024 7 13 99 3407 12345 2025 31415"; nohup bash -c "/workspace/lane.sh 0 abl_arch14_partial_nogate 31415; /workspace/lane.sh 0 abl_arch10_contrast $S; /workspace/lane.sh 0 abl_arch05_shared_tf $S" > /workspace/logs/lane0.txt 2>&1 & echo LAUNCHED
```

Xong khi `grep -c "=== DONE" /workspace/logs/lane0.txt /workspace/logs/lane1.txt` in
`lane0.txt:21` và `lane1.txt:20`. Khoảng 4 giờ kể từ khi khởi chạy.

### 5b. Nếu ra `VERDICT: DIVERGED`

ARCH-13/14 hôm 29/09 **không** ghép được với các run mới. Phải chạy lại cả hai arm trên máy
này, thêm 20 run, khoảng 2 giờ nữa (tổng khoảng 6 giờ, khoảng 6,6 USD):

```bash
S="42 123 2024 7 13 99 3407 12345 2025 31415"; nohup bash -c "/workspace/lane.sh 0 abl_arch10_contrast $S; /workspace/lane.sh 0 abl_arch05_shared_tf $S; /workspace/lane.sh 0 abl_arch14_partial_nogate $S" > /workspace/logs/lane0.txt 2>&1 & echo LAUNCHED
```

Khi `lane1.txt` đã có 2 dòng `FINISHED` (ARCH-12 và ARCH-09 xong), cho GPU 1 chạy ARCH-13:

```bash
S="42 123 2024 7 13 99 3407 12345 2025 31415"; nohup /workspace/lane.sh 1 abl_arch13_nogate $S > /workspace/logs/lane1b.txt 2>&1 & echo LAUNCHED
```

Trong trường hợp này, ở mục 7 giải nén vào **`checkpoints\box_5070ti_b`**, không vào
`box_5070ti`.

---

## 6. Theo dõi

Có thể đóng SSH và tắt máy tính; `nohup` giữ các lần chạy tiếp tục.

```bash
grep -h "DONE\|FAIL\|FINISHED" /workspace/logs/lane*.txt | tail -8; grep -c "=== DONE" /workspace/logs/lane0.txt /workspace/logs/lane1.txt
```

Chạy lại một seed bị `FAIL` (xem nguyên nhân trước bằng
`tail -20 /workspace/logs/<config>_s<seed>.txt`):

```bash
nohup /workspace/lane.sh <GPU rảnh> <config> <seed> > /workspace/logs/lane_retry.txt 2>&1 &
```

---

## 7. Kéo kết quả về — CHỈ khi mọi làn đã `FINISHED`

**Hôm 29/09 mất một run vì đóng gói khi làn chưa xong.** Kiểm tra trước:

```bash
grep -h "FINISHED" /workspace/logs/lane*.txt; ps aux | grep -c "[0]2_train.py"
```

Trường hợp 5a phải thấy **6 dòng** `FINISHED`:
- `lane0_verify.txt`: 1 dòng (run nghiệm thu)
- `lane1.txt`: 2 dòng (ARCH-12, ARCH-09)
- `lane0.txt`: 3 dòng (ARCH-14, ARCH-10, ARCH-05)

Trường hợp 5b: **7 dòng**, gồm `lane0.txt` 3 dòng (ARCH-10, ARCH-05, ARCH-14) và thêm
`lane1b.txt` 1 dòng (ARCH-13). Số tiến trình `02_train.py` phải là **0**.

Đóng gói:

```bash
cd /workspace/ECG_HRV && tar -czf /workspace/results_5070ti_b.tgz --exclude='*.pth' checkpoints && tar -czf /workspace/logs_5070ti_b.tgz --exclude='*.pth' -C /workspace logs verify_runs && ls -lh /workspace/*_b.tgz && tar -tzf /workspace/results_5070ti_b.tgz | grep -c history.json
```

Dòng cuối phải là **41** trong trường hợp 5a, hoặc **60** trong trường hợp 5b.

Ở máy bạn (PowerShell):

```powershell
scp -P <PORT> root@<HOST>:/workspace/results_5070ti_b.tgz root@<HOST>:/workspace/logs_5070ti_b.tgz "E:\KLTN\KL Project"
```

Báo tôi khi tải xong: tôi sẽ giải nén, kiểm tra từng run và chạy kiểm định. **Chỉ Destroy
sau khi tôi xác nhận đủ run.**

---

## 8. Sau khi có kết quả

- **4 cặp của ARCH-13/14 cùng máy**:
  `stat_tests_extra.py --glob "checkpoints/box_5070ti/run_*"`.
- **Thêm một kết quả cho bài**: `stat_tests.py` trên cùng thư mục sẽ cho cặp chính
  ARCH-10 − ARCH-09 và cặp ARCH-05 − ARCH-10 **trên GPU thứ hai**. Nếu vẫn có ý nghĩa thống
  kê, bài có thêm bằng chứng lặp lại trên phần cứng khác.
- Số trong các bảng hiện có của bài (5060 Ti) giữ nguyên. Kết quả 5070 Ti là một khối
  riêng, ghi rõ phần cứng.

---

## Phụ lục — kiểm tra tất cả các làn bằng một lệnh

Tạo một lần:

```bash
cat > /workspace/check.sh <<'EOF'
#!/bin/bash
# check.sh [ROOT]  - is every arm complete (trained + calibrated, 10 distinct seeds)?
ROOT=${1:-/workspace/ECG_HRV}
LOGS=${2:-/workspace/logs}
cd "$ROOT" || exit 1
EXP="abl_arch05_shared_tf abl_arch09_decoupled abl_arch10_contrast abl_arch12_partial abl_arch13_nogate abl_arch14_partial_nogate"
total=0; ok=1
echo "arm                          done   seeds"
for e in $EXP; do
  seeds=""
  for d in checkpoints/run_*"${e}"-*; do
    [ -d "$d" ] || continue
    [ -f "$d/history.json" ] && [ -f "$d/calibration_results.json" ] || continue
    s=$(grep -A40 "^training:" "$d/config_snapshot.yaml" | grep -m1 "  seed:" | awk '{print $2}' | tr -d '\r')
    seeds="$seeds $s"
  done
  n=$(echo $seeds | wc -w)
  uniq=$(echo $seeds | tr ' ' '\n' | grep -v '^$' | sort -u | wc -l)
  flag=""
  [ "$n" -ne "$uniq" ] && flag="  !! DUPLICATE SEED" && ok=0
  [ "$uniq" -ne 10 ] && ok=0
  printf "%-28s %2d/10 %s%s\n" "$e" "$uniq" "$seeds" "$flag"
  total=$((total + uniq))
done
unfinished=$(for d in checkpoints/run_*; do [ -d "$d" ] && { [ -f "$d/history.json" ] && [ -f "$d/calibration_results.json" ] || echo "$d"; }; done)
running=$(pgrep -fc "02_train.py|03_calibrate.py" 2>/dev/null || echo 0)
lanes=$(pgrep -fc "[l]ane.sh" 2>/dev/null || echo 0)
fails=$(grep -h "FAIL" "$LOGS"/lane*.txt 2>/dev/null)
echo
echo "total complete runs : $total / 60"
echo "train/calib running : $running"
echo "lanes still alive   : $lanes"
[ -n "$unfinished" ] && echo "incomplete folders  :" && echo "$unfinished" | sed 's/^/  /'
[ -n "$fails" ] && echo "FAIL lines:" && echo "$fails" | sed 's/^/  /' && ok=0
echo
if [ $ok -eq 1 ] && [ "$total" -eq 60 ] && [ "$running" -eq 0 ] && [ "$lanes" -eq 0 ] && [ -z "$unfinished" ]; then
  echo ">>> ALL DONE - safe to pack (section 7)"
else
  echo ">>> NOT DONE - do not pack yet"
fi
EOF
chmod +x /workspace/check.sh && echo OK
```

Chạy bất cứ lúc nào:

```bash
/workspace/check.sh
```

Chỉ đóng gói khi dòng cuối là `>>> ALL DONE - safe to pack (section 7)`.
