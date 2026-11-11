# Runbook: 10-fold CV, protocol v2 (chạy tự động trên máy thuê)

## Mục tiêu

Chạy toàn bộ nghiên cứu CV theo `docs/PROTOCOL_CV_V2.md` bằng **một lệnh**. Script
`scripts/cloud/run_cv.sh` tự làm lần lượt:

| Bước | Việc | Số lần train |
|---|---|---|
| 0 | Dựng dữ liệu CV (LBBB = CLBBB + ILBBB) và kiểm tra checksum với `configs/cv/data_sha256.txt` | – |
| 1 | Lưới lr: 10 arm × lr {1e-4, 3e-4, 1e-3} trên fold 1, seed 3001 | 30 |
| 2 | `29_select_lr.py`: mỗi arm giữ lr có validation loss thấp nhất. Run được chọn chính là run fold 1 của CV; 2 run còn lại bị dời sang `checkpoints/cv_v2_lr_grid/` | – |
| 3 | Fold 2–10 với lr đã chọn, seed 3000+k; calibrate và dump dự đoán cho cả 100 run | 90 |
| 4 | `28_cv_aggregate.py`: OOF, bootstrap theo bệnh nhân, H1–H3 có Holm; chạy thêm bản bỏ fold 2 | – |

Tổng cộng **120 lần train**, trong đó **100 run** tạo thành nghiên cứu CV (10 arm × 10 fold).
Số không phải 130: run lr được chọn ở fold 1 được dùng lại làm run CV fold 1, vì nó có đúng config,
dữ liệu và seed của run đó.

Mỗi GPU có một worker tự lấy việc từ hàng đợi chung. Mỗi việc ghi vào một thư mục cố định
`checkpoints/cv_v2/<arm>_f<fold>_lr<lr>`. Nếu máy treo hay reboot, **chạy lại đúng lệnh cũ**: việc
đã xong được bỏ qua; việc dở dang được train lại từ đầu với cùng seed.

Script sẽ **từ chối chạy** nếu code trong `src/`, `scripts/`, `configs/` khác với commit đã clone.
Commit đó là bản đăng ký protocol.

## Ước lượng (4 × RTX 5060 Ti)

| Loại run | Số run | Thời gian một run (ước lượng) |
|---|---|---|
| 7 arm của dự án, 15 epoch | 84 | ~17 phút, cộng ~3 phút calibrate và dump |
| 3 backbone công bố, 30 epoch | 36 | ~15–35 phút (mô hình nhỏ, phụ thuộc tốc độ đọc dữ liệu) |

Tổng khoảng **10–14 giờ**, ~5–7 $ ở mức 0,48 $/giờ. Sau khi pha 1 xong, tôi sẽ ước lượng lại từ
log thật.

Yêu cầu máy:
- ≥ 8 vCPU cho mỗi GPU
- RAM ≥ 64 GB
- đĩa ≥ 60 GB (khoảng 120 × 160 MB trọng số cho các arm lớn, cộng PTB-XL 3 GB)

---

## 1. Môi trường (trên máy thuê)

```bash
mkdir -p /workspace/logs && export PATH=/venv/main/bin:$PATH && pip install -q numpy==2.4.6 scipy==1.18.0 scikit-learn==1.9.0 pandas==2.3.2 wfdb neurokit2 matplotlib tqdm PyYAML optuna iterative-stratification awscli && pip install -q numpy==2.4.6 scipy==1.18.0 scikit-learn==1.9.0 && python -c "import torch,numpy,scipy,sklearn,wfdb,neurokit2;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.device_count(), numpy.__version__, scipy.__version__, sklearn.__version__)"
```

Phải in `2.11.0+cu128 12.8 <tên GPU> <số GPU> 2.4.6 1.18.0 1.9.0`. Nếu torch ra bản khác:

```bash
export PATH=/venv/main/bin:$PATH && pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

## 2. PTB-XL 1.0.1 (chỉ `records500`)

```bash
export PATH=/venv/main/bin:$PATH && aws configure set default.s3.max_concurrent_requests 64 && aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.1/ /workspace/PTB-XL/ --exclude '*.zip' --exclude 'records100/*'
```

```bash
cd /workspace/PTB-XL && sha256sum -c SHA256SUMS.txt --quiet --ignore-missing && echo "CHECKSUM OK" && find records500 -name "*_hr.dat" | wc -l
```

Phải in `CHECKSUM OK` và `21837`.

## 3. Lấy code

```bash
cd /workspace && git clone -b CV-PROTOCOL-V2 https://github.com/lynnhatsuki3010/multitask-ecg.git ECG_HRV && cd ECG_HRV && git log -1 --format='%h %s' && mkdir -p data/raw && ln -s /workspace/PTB-XL data/raw/PTB-XL && ls data/raw/PTB-XL/ptbxl_database.csv
```

Mã commit in ra phải trùng mã tôi báo trong chat.

## 4. Chạy thử (smoke test, ~10 phút)

Chạy đủ mọi bước với 2 arm, fold 1–2, 3 epoch, 200 bản ghi, trong thư mục riêng `*_debug`. Lần
chạy này cũng dựng dữ liệu CV và kiểm tra checksum, nên lần chạy thật không phải dựng lại.

```bash
cd /workspace/ECG_HRV && export PATH=/venv/main/bin:$PATH && DEBUG=1 nohup bash scripts/cloud/run_cv.sh > /workspace/logs/smoke.out 2>&1 & echo STARTED
```

```bash
tail -5 /workspace/ECG_HRV/logs/cv_v2_debug/status.txt
```

Xong khi dòng cuối là `=== ALL FINISHED: 4/4 runs, FAIL lines in this attempt: 0 ===`. Nếu thấy
`ABORT` hay `FAIL`, gửi tôi:

```bash
cd /workspace/ECG_HRV && cat logs/cv_v2_debug/status.txt; tail -30 $(ls -t logs/cv_v2_debug/*.log | head -1)
```

## 5. Chạy thật

```bash
cd /workspace/ECG_HRV && export PATH=/venv/main/bin:$PATH && nohup bash scripts/cloud/run_cv.sh > /workspace/logs/run_cv.out 2>&1 & echo STARTED
```

Có thể đóng SSH. Sau 3–4 phút kiểm tra số tham số:

```bash
grep "TRAINED\|START" /workspace/ECG_HRV/logs/cv_v2/status.txt | tail -8; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

Mỗi dòng `TRAINED` có số tham số. Phải khớp với bảng sau:

| Arm | Số tham số | Epoch |
|---|---|---|
| `cv_layout` (đề xuất) | 13,140,016 | 15 |
| `cv_decoupled`, `cv_pairsoft`, `cv_hier` | 13,292,496 | 15 |
| `cv_shared` | 12,644,785 | 15 |
| `cv_partial_layout` | 13,484,561 | 15 |
| `cv_contrast` | 13,391,312 | 15 |
| `cv_ecgtransform` | 1,241,397 | 30 |
| `cv_xresnet1d101` | 2,267,534 | 30 |
| `cv_inception1d` | 640,270 | 30 |

## 6. Theo dõi

Tiến độ tổng:

```bash
cd /workspace/ECG_HRV && S=logs/cv_v2/status.txt && echo "TRAINED $(grep -c ' TRAINED ' $S)/120  DONE $(grep -c ' DONE ' $S)/100  FAIL $(grep -c ' FAIL ' $S)  RUNNING $(pgrep -fc 'scripts/02_train|scripts/03_calibrate|scripts/27_dump')" && tail -4 $S
```

Từng epoch của các run đang chạy:

```bash
cd /workspace/ECG_HRV && for f in $(ls -t logs/cv_v2/*.train.log | head -4); do echo "== $(basename $f)"; tr '\r' '\n' < "$f" | grep "Epoch \[" | tail -2 | cut -c1-160; done
```

Kết quả chọn lr (có sau khi pha 1 xong, khoảng 2–3 giờ):

```bash
cd /workspace/ECG_HRV && grep -A12 "lr selection" logs/cv_v2/status.txt
```

Bảng này in validation loss của 3 lr cho từng arm và lr được chọn. Gửi tôi bảng này khi có.

**Xong khi** dòng cuối của `status.txt` là:

```
=== ALL FINISHED: 100/100 runs, FAIL lines in this attempt: 0 ===
```

Đồng thời phải có `results/cv_v2/cv_results.md` và `results/cv_v2/cv_results_no_fold2.md`.

**Nếu có `FAIL`:** đừng sửa gì. Gửi tôi dòng FAIL và đuôi log mà dòng đó chỉ tới. Thường chỉ cần chạy
lại đúng lệnh ở bước 5: script chỉ chạy các việc còn thiếu.

**Nếu máy reboot:** chạy lại đúng lệnh ở bước 5.

## 7. Đóng gói và tải về

Không lấy trọng số (`*.pth`). Gói gồm config, history, calibration, `predictions.npz`, log, kết
quả tổng hợp:

```bash
cd /workspace/ECG_HRV && tar -czf /workspace/cv_v2_results.tgz --exclude='*.pth' checkpoints/cv_v2 checkpoints/cv_v2_lr_grid logs/cv_v2 results/cv_v2 && ls -lh /workspace/cv_v2_results.tgz && tar -tzf /workspace/cv_v2_results.tgz | grep -c "cv_v2/.*predictions.npz"
```

Dòng cuối phải là **100**.

Tuỳ chọn: giữ trọng số 10 fold của 4 arm chính (`cv_layout`, `cv_decoupled`, `cv_shared`,
`cv_partial_layout`), chỉ phần mô hình, không có optimizer (~2 GB). Dùng cho ngoại kiểm PTB
Diagnostic hay phân tích attribution sau này:

```bash
cd /workspace/ECG_HRV && export PATH=/venv/main/bin:$PATH && python -c "
import glob, torch, os
for p in glob.glob('checkpoints/cv_v2/cv_*_f*_lr*/best_model.pth'):
    if os.path.basename(os.path.dirname(p)).rsplit('_f', 1)[0] in ('cv_layout', 'cv_decoupled', 'cv_shared', 'cv_partial_layout'):
        c = torch.load(p, map_location='cpu', weights_only=False)
        torch.save({'epoch': c['epoch'], 'model': c['model']}, p.replace('best_model.pth', 'model_only.pth'))
" && tar -cf /workspace/cv_v2_weights.tar checkpoints/cv_v2/*/model_only.pth && ls -lh /workspace/cv_v2_weights.tar
```

PowerShell ở máy bạn:

```powershell
scp -P <PORT> root@<HOST>:/workspace/cv_v2_results.tgz "E:\KLTN\KL Project"
```

Báo tôi khi tải xong. Tôi sẽ:
1. Giải nén vào `checkpoints\` và chạy lại `28_cv_aggregate.py` ở máy local, đối chiếu với kết quả
   trên máy thuê.
2. Đọc bảng H1–H3, bản bỏ fold 2, phép che đạo trình và các luật hậu kiểm.

**Chỉ Destroy máy sau khi tôi xác nhận đủ 100 run.**
