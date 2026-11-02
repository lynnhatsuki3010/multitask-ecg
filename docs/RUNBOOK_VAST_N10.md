# Runbook — nâng Tier 1 lên n=10 trên vast.ai

Mục tiêu: thêm 7 seed cho 4 cấu hình, để claim significance không còn dựa trên n=3.

| | |
|---|---|
| Config | `abl_arch05_shared_tf`, `abl_arch09_decoupled`, `abl_arch10_contrast`, `abl_arch10b_cond_dim` |
| Seed đã có | 42, 123, 2024 |
| Seed thêm | 7, 13, 99, 3407, 12345, 2025, 31415 |
| Số run | 28 |
| Thời gian | ~1,5 giờ ở 4 GPU (~12 phút/run) |

---

## 1. Thông số máy đã chạy chiến dịch 02/08/2026

Không có trường nào trong `checkpoints/run_*` ghi phần cứng. Suy ra từ log sinh cùng đợt:

| | |
|---|---|
| GPU | **NVIDIA GeForce RTX 5070** (Blackwell, sm_120, 12 GB) |
| Số GPU | 4 |
| CPU | 61 core |
| torch | **2.11.0+cu128** |
| CUDA | **12.8** |
| Đường dẫn | `/workspace/ECG_HRV` |

Nguồn: `calibration_all.txt` in `Using GPU: NVIDIA GeForce RTX 5070` 64 lần, và chứa đúng tên
24 run `run_20260802_*_abl_arch*` — tức là box đã hiệu chỉnh chính những run này.
`complexity_all.txt` và `env.txt` (cùng mốc 03/08 04:01) ghi `RTX 5070 / torch 2.11.0+cu128 /
cuda 12.8`. Số core lấy từ `RUNBOOK_FINAL.md`.

**Cảnh báo:** `complexity.txt` (01/08 23:35) ghi **RTX 4090** — đó là box của đợt trước, không
phải box của chiến dịch ARCH. Đừng lấy nhầm 4090 làm mục tiêu.

Đây là suy luận từ log, không phải trường phần cứng được ghi lại. Mục 4 kiểm chứng bằng thực
nghiệm, không cần tin suy luận này.

---

## 2. Thuê máy

Ưu tiên: **4 × RTX 5070**. Không có thì 2 × hoặc 1 × RTX 5070 vẫn chạy được, chỉ chậm hơn theo
tỉ lệ — điều **không** được đổi là `num_workers: 14`, vì nó đổi luồng ngẫu nhiên numpy của
augmentation (bất biến 1, `RUNBOOK_FINAL.md`). Máy ít core hơn thì worker tranh core và chậm,
nhưng số vẫn ghép được với 3 seed cũ. Hạ `nw` để "vừa máy" là hỏng cả bảng.

```bash
vastai search offers 'gpu_name=RTX_5070 num_gpus>=1 cuda_vers>=12.8 disk_space>30 inet_down>300' -o dph
```

Driver phải hỗ trợ CUDA 12.8. Đĩa **30 GB** là đủ rộng rãi — vast tính tiền theo dung lượng nên
đừng lấy dư:

| | |
|---|---|
| PTB-XL giải nén | 3,1 GB |
| zip PTB-XL (xoá ngay sau khi giải nén) | 1,7 GB |
| torch + CUDA runtime | ~5 GB |
| 28 checkpoint (`best_model.pth` ~50 MB mỗi cái) | ~1,4 GB |
| mã nguồn + processed + splits | ~50 MB |

Ước tính: 4 GPU ~1,5 giờ; 2 GPU ~3 giờ; 1 GPU ~5,6 giờ.

---

## 3. Dựng box

```bash
apt-get update -qq && apt-get install -y -qq git tmux rsync unzip zip
mkdir -p /workspace && cd /workspace
```

**Cẩn thận môi trường Python.** Template vast đặt env thật ở `/venv/main`, nhưng shell
non-interactive (tức mọi lệnh chạy qua `ssh host "..."`) không nạp nó và rơi về
`/usr/bin/python3` — cài `pip install` kiểu đó là đổ vào system python, nơi không có torch cu128.
Prompt tương tác hiện `(main)` nên nhìn thì tưởng đã đúng env.

Mở đầu **mọi** lệnh non-interactive bằng:

```bash
export PATH=/venv/main/bin:$PATH
```

Kiểm tra torch có sẵn chưa — template thường ship đúng bản, khỏi tải 3 GB:

```bash
/venv/main/bin/python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.device_count())"
```

Ra `2.11.0+cu128 12.8 ... 4` là khớp chiến dịch cũ. Khác bản thì mới cài:

```bash
export PATH=/venv/main/bin:$PATH && pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Mã nguồn — `configs/experiments/abl_arch*.yaml` hiện **untracked**, nên đẩy cây làm việc chứ
đừng `git clone`.

Windows không có `rsync`, và PowerShell nối dòng bằng backtick chứ không phải `\`. Dùng
tarball + `scp` (Windows có sẵn OpenSSH). Chạy trong **Git Bash** ở máy local:

```bash
cd "E:/KLTN/KL Project/ECG_HRV" && tar -czf /tmp/ecg_box.tar.gz src scripts configs requirements.txt data/processed_dir5_imi50_500hz data/splits_dir5_imi50_500hz data/raw/PTB-XL/SHA256SUMS.txt scratch/ref_n3 docs/RUNBOOK_VAST_N10.md docs/RUNBOOK_FINAL.md
```

Danh sách **liệt kê thứ cần**, không phải loại thứ không cần — thư mục gốc có `.git` 1,6 GB,
`artifacts` 726 MB và ~18 GB tarball mô hình, dùng `--exclude` là sớm muộn cũng lọt. Gói ra
khoảng 6 MB.

```bash
scp -P <PORT> /tmp/ecg_box.tar.gz root@<HOST>:/workspace/
```

Trên box:

```bash
mkdir -p /workspace/ECG_HRV && tar -xzf /workspace/ecg_box.tar.gz -C /workspace/ECG_HRV && ls /workspace/ECG_HRV
```

Gói này đã gồm `data/processed_dir5_imi50_500hz/` và `data/splits_dir5_imi50_500hz/`.
**Dùng nguyên chúng, đừng sinh lại bằng `01_build_metadata.py`** — label matrix và split phải
trùng từng byte với 3 seed cũ, sinh lại là mở cửa cho sai lệch không ai kiểm được. Gói cũng đã
kèm `scratch/ref_n3/` cho bước nghiệm thu, và `SHA256SUMS.txt` bản local để đối chiếu PTB-XL.

Rồi trên box:

```bash
export PATH=/venv/main/bin:$PATH && cd /workspace/ECG_HRV && pip install -q -r requirements.txt
```

`requirements.txt` ghi `torch>=2.0.0`, pip thấy 2.11.0+cu128 đã thoả nên không đụng vào — bản
cu128 giữ nguyên.

### Dữ liệu thô

Dự án dùng **PTB-XL 1.0.1**, không phải bản mới nhất. Xác định bằng kích thước file, khớp
chính xác `s3://physionet-open/ptb-xl/1.0.1/`:

| file | local = 1.0.1 | 1.0.2 | 1.0.3 |
|---|---|---|---|
| `RECORDS` | 1 135 522 | 1 133 650 | 1 133 547 |
| `ptbxl_database.csv` | 6 602 796 | 9 619 646 | 6 594 879 |
| `SHA256SUMS.txt` | 8 298 464 | 8 284 874 | 8 284 204 |

Local cũng không có `ptbxl_v102_changelog.txt` / `ptbxl_v103_changelog.txt` — dấu hiệu thứ hai
của 1.0.1. Tải nhầm 1.0.3 là đổi metadata dưới chân 3 seed cũ.

**Đừng tải từ `physionet.org/static/`** — nó bóp còn ~140 KB/s, 3 GB mất gần 5 giờ. Dùng mirror
AWS Open Data, không cần credential:

```bash
export PATH=/venv/main/bin:$PATH && pip install -q awscli && aws configure set default.s3.max_concurrent_requests 64
```

```bash
cd /workspace/ECG_HRV/data/raw && aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.1/ PTB-XL/ --exclude 'SHA256SUMS.txt' --exclude '*.zip'
```

Loại `SHA256SUMS.txt` để giữ bản local trong tarball làm mốc đối chiếu độc lập.

Verify bằng `SHA256SUMS.txt` **của bản local** (đã có sẵn trong tarball mục 3, và lệnh sync ở
trên cố tình không đè lên nó). Verify bằng file checksum do chính nguồn tải về chỉ chứng minh
nguồn tự nhất quán, không chứng minh trùng bản local:

```bash
cd /workspace/ECG_HRV/data/raw/PTB-XL && sha256sum -c SHA256SUMS.txt --quiet && echo "PTB-XL OK - byte-identical voi ban local"
```

Checksum lệch là dừng — sai bản dataset thì mọi thứ sau đó vô nghĩa.

### Ghi lại phần cứng (việc mà đợt trước đã bỏ sót)

```bash
cd /workspace/ECG_HRV && nvidia-smi > /workspace/box_nvidia_smi.txt && \
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.device_count())" >> /workspace/box_nvidia_smi.txt && \
nproc >> /workspace/box_nvidia_smi.txt && cat /workspace/box_nvidia_smi.txt
```

---

## 4. Nghiệm thu box — làm TRƯỚC khi tiêu tiền cho 28 run

Chạy lại đúng một cấu hình ở seed **đã có**, rồi so với run cũ. Cùng phần cứng thì kết quả
trùng từng bit (đo được: max |Δ| = 0 trên 87 chỉ số). Khác kiến trúc GPU thì IMI AUPRC lệch
~0,0099 — bằng một phần tư hiệu ứng bài đang muốn chứng minh.

Bộ tham chiếu `scratch/ref_n3/` (4,2 MB, chỉ `history.json` + `config_snapshot.yaml` + metrics,
không có `.pth`) đã nằm trong tarball ở mục 3.

Trên box — một run, ~12 phút:

```bash
cd /workspace/ECG_HRV && python scripts/02_train.py --config configs/experiments/abl_arch10_contrast.yaml --seed 42
```

```bash
cd /workspace/ECG_HRV && python scripts/verify_box.py --new $(ls -dt checkpoints/run_*abl_arch10_contrast* | head -1)
```

| Verdict | Làm gì |
|---|---|
| `bit-identical` | Chạy tiếp mục 5 |
| trong `1e-6` | Chạy tiếp, ghi lại độ lệch vào `EXPERIMENT_LOG.md` |
| `DIVERGED` | **Dừng.** Không ghép seed mới với seed cũ được. Hoặc đổi box, hoặc chấp nhận chạy lại cả 10 seed × 4 config (40 run) trên box này |

Run nghiệm thu này là bản sao của seed 42 — xoá đi sau khi verify, để nó không lọt vào bảng
gộp (`aggregate_seeds` và `stat_tests` đều giữ run mới nhất theo `(exp_id, seed)`, nên bỏ quên
nó sẽ âm thầm thay thế run gốc):

```bash
rm -rf $(ls -dt /workspace/ECG_HRV/checkpoints/run_*abl_arch10_contrast* | head -1)
```

---

## 5. Chạy 28 run

`run4.sh` như `RUNBOOK_FINAL.md` mục 0.1. `num_workers` và `monitor_metric` nằm trong config
(`02_train.py` không có cờ tương ứng), và config trong repo đã được đối chiếu trùng khít
`config_snapshot.yaml` của cả 12 run cũ, nên không cần override gì.

Máy 4 GPU:

```bash
cat > /workspace/run4.sh <<'EOF'
#!/bin/bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/venv/main/bin:$PATH
cd /workspace/ECG_HRV
S=$1; shift
i=0
for c in "$@"; do
  g=$((i % 4))
  echo "=== START $c seed=$S gpu=$g $(date -u) ==="
  CUDA_VISIBLE_DEVICES=$g python scripts/02_train.py --config configs/experiments/$c.yaml --seed $S \
    > /workspace/log_${c}_s${S}.txt 2>&1 &
  i=$((i + 1))
  [ $((i % 4)) -eq 0 ] && wait && echo "--- wave done $(date -u)"
done
wait
echo "BATCH DONE seed=$S $(date -u)"
EOF
chmod +x /workspace/run4.sh && echo OK
```

Trong `tmux` để rớt mạng không mất việc (`Ctrl+B` rồi `D` để thả):

```bash
tmux new -s n10
```

```bash
for s in 7 13 99 3407 12345 2025 31415; do /workspace/run4.sh $s abl_arch05_shared_tf abl_arch09_decoupled abl_arch10_contrast abl_arch10b_cond_dim; done
```

Máy 1 GPU thì thay bằng:

```bash
cd /workspace/ECG_HRV && for s in 7 13 99 3407 12345 2025 31415; do for c in abl_arch05_shared_tf abl_arch09_decoupled abl_arch10_contrast abl_arch10b_cond_dim; do python scripts/run_seed.py --config configs/experiments/$c.yaml --seed $s --no-calibrate; done; done
```

---

## 6. Hiệu chỉnh

`run4.sh` chỉ train. Hiệu chỉnh toàn bộ run mới (cần cho bảng test của `abl_arch10b`):

```bash
cd /workspace/ECG_HRV && for d in $(ls -d checkpoints/run_*abl_arch* | grep -v 20260802); do python scripts/03_calibrate.py --dir $d; done
```

---

## 7. Kết quả — kiểm tra trên box rồi mới tải về

```bash
cd /workspace/ECG_HRV && python scripts/stat_tests.py --out /workspace/stat_tests_n10.md
```

Mỗi arm phải in `n=10` và không còn dòng `missing declared seeds`. Còn dòng đó là thiếu run.

Gói kết quả trên box, bỏ `.pth` (~50 MB × 28):

```bash
cd /workspace/ECG_HRV && tar -czf /workspace/ecg_results.tar.gz --exclude='*.pth' checkpoints/ && cp /workspace/box_nvidia_smi.txt /workspace/stat_tests_n10.md /workspace/ 2>/dev/null; ls -lh /workspace/ecg_results.tar.gz
```

Tải về, chạy ở Git Bash local:

```bash
scp -P <PORT> root@<HOST>:/workspace/ecg_results.tar.gz root@<HOST>:/workspace/box_nvidia_smi.txt root@<HOST>:/workspace/stat_tests_n10.md /tmp/
```

```bash
cd "E:/KLTN/KL Project/ECG_HRV" && tar -xzf /tmp/ecg_results.tar.gz && cp /tmp/box_nvidia_smi.txt /tmp/stat_tests_n10.md docs/
```

Giữ `best_model.pth` của 7 seed mới cho `abl_arch10b` nếu còn định làm ensemble hoặc XAI; các
arm khác không cần.

---

## 8. Sau khi có n=10

1. `python scripts/stat_tests.py --out docs/stat_tests_n10.md`
2. Sửa `stats.pstdev` → `stats.stdev` ở `scripts/aggregate_seeds.py:92` (bảng test đang dùng
   population SD, còn `compare_val_runs.py:223` dùng sample SD — hai bảng trong bài không cùng
   định nghĩa "±")
3. `python scripts/aggregate_seeds.py abl_arch10b_cond_dim` cho bảng test
4. Thay 4 chỗ "4,3 SE" bằng `Δ ± 95% CI, p` — grep `4,3 SE` để không sót
5. Ghi n từng dòng vào caption Bảng 12: 4 dòng n=10, 4 dòng n=3
