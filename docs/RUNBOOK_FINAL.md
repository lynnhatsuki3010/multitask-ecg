# Runbook cuối — chiến dịch trên kiến trúc bài

Nền: **`branche10_05_cond_depth`** = ARCH-11 (`decoupled_multitask` + `contrast_v2` +
`cond_layers: 3` + `cd_lead_out_dim: 192`).

## Bốn bất biến — vi phạm là hỏng cả chiến dịch

**1. `--num-workers 14` cho mọi run.** Nó đổi luồng ngẫu nhiên numpy của augmentation nên không
trung tính.

> **Đo được trên máy 61 core / 4 GPU:** workload này **nghẽn GPU, không nghẽn dataloader**.
> GPU 0 chạy 83–99% trong khi load average chỉ 6,05 — worker ngồi không, chờ tiến trình chính.
> Nâng `nw` 18 → 48 làm epoch **chậm đi** (42s → 46,5s) vì thêm chi phí IPC.
> Kiến trúc `decoupled_multitask` nặng 6,14 GFLOPs so với 4,44 của `multi_branch`.
>
> Đòn bẩy thật là **4 GPU**: chạy 4 config song song, mỗi cái `nw=14` (4 × 14 = 56 trên 61 core).
> Thông lượng gấp 4, chiến dịch từ ~12,5 giờ xuống **~3,2 giờ**.

**2. `--monitor imi_auprc` cho mọi run.** Config nền `branche10_05_cond_depth` mang
`f1/cond/macro` (tàn dư của thí nghiệm tinh chỉnh Conduction), còn `dir5_p3a_imi50` mang
`imi_auprc` — chuỗi ARCH cũ vì thế so checkpoint chọn theo hai luật khác nhau. Hai script sinh
config nay **ghi đè** thay vì kế thừa.

**3. Một GPU instance duy nhất** từ đầu đến cuối. Đo được: đổi kiến trúc GPU làm IMI AUPRC lệch
0,0099 và AMI F1 lệch 0,157 ở cùng seed. Đổi máy giữa chừng thì ghi lại và **không trộn seed
giữa hai máy**. (Bốn GPU cùng model trong một instance thì không sao.)

**4. Config sinh trên máy chạy**, không lấy từ git. `configs/experiments/gp_*.yaml` và
`gpb10_*.yaml` đã gitignore chính vì lỗi `nw=28` lọt vào repo.

## Hai tiêu chí, khai trước

| | Chỉ số | Dùng để |
|---|---|---|
| `monitor_metric` | `imi_auprc` | chọn epoch nào trong một run thành `best_model.pth` |
| **S** | **trung bình 3 macro AUPRC** | chọn kỹ thuật nào thắng một stage |

Hai quyết định độc lập. Báo cáo **cả hai** cách đọc trong bài.

Bảng 5 của bài ghi *"Tiêu chí chọn mô hình | IMI AUPRC"* — đúng với chiến dịch này, nhưng **sai với
mô hình đang báo cáo** (`branche10_05` dùng `f1/cond/macro`). Sửa sau khi chạy xong.

---

# Nhật ký quyết định

Ghi lại ngay khi chốt từng stage, để bảng trong bài truy ngược được.

## Stage 1 — NORM: **zscore** (02/08/2026, 3 seed, kiến trúc bài)

| | minmax | robust | **zscore** |
|---|---|---|---|
| **S** (tiêu chí chốt) | 0,637±0,016 | 0,670±0,014 | **0,685±0,009** |
| IMI AUPRC | 0,613±0,026 | **0,671±0,006** | 0,661±0,005 |
| Cond macro F1 | 0,579±0,024 | 0,664±0,036 | **0,674±0,032** |
| Arrhy macro F1 | 0,720±0,040 | 0,789±0,048 | **0,798±0,041** |

zscore thắng 7/11 chỉ số. `minmax` thua rõ mọi mặt, loại.

**Hai cách đọc chênh nhau, và phải nêu cả hai trong bài:**

- S: zscore hơn robust 0,015 → **1,6 SE**
- IMI AUPRC: robust hơn zscore 0,010 → **2,2 SE**

Robust thắng `imi_auprc` chắc hơn zscore thắng S. Chọn zscore vì S là tiêu chí **khai trước**, không
phải vì nó thắng rõ hơn.

**Hệ quả cho bản thảo:** Mục 3.3 hiện biện luận lý thuyết cho robust (*"ít nhạy cảm hơn với ngoại
lai và spike nhiễu"*). Phải viết lại: trung vị/IQR bền với ngoại lai nhưng làm mất biên độ tương
đối mà mô hình cần. Bảng 11 đổi dòng Chuẩn hoá sang zscore.

## Stage 2 — AUG: **lead_dropout** (hoà với `none`)

| | S | IMI AUPRC |
|---|---|---|
| **lead_dropout** | **0,689 ± 0,009** | 0,668 ± 0,008 |
| none | 0,685 ± 0,009 | 0,661 ± 0,006 |
| mixup | 0,683 ± 0,016 | 0,664 ± 0,008 |
| noise_warp | 0,669 ± 0,016 | **0,670 ± 0,009** |
| random_crop | 0,607 ± 0,011 | 0,644 ± 0,015 |

**Không phải thắng — là hoà.** lead_dropout hơn `none` đúng 0,004 = **0,5 SE**. Bốn nhánh đầu trên
`imi_auprc` trải 0,661–0,670 với SD ~0,008, thứ tự còn đảo.

**Augmentation không đem lại lợi ích đo được trên kiến trúc này.** Phải viết đúng như vậy, không
được trình bày lead_dropout như một cải thiện. Giữ nó vì xếp nhất theo S, nhì theo `imi_auprc` —
vị trí nhất quán nhất giữa hai cách đọc — và vì mô phỏng mất điện cực là tính chất bền vững muốn có
bất kể hiệu quả đo được.

Kết luận chắc chắn duy nhất: **`random_crop` thua dứt khoát** (S 0,607, kém 0,08 ≈ 9 SD). Cắt ngẫu
nhiên phá mốc thời gian mà chẩn đoán phụ thuộc.

## Stage 3 — SAMP: **uniform** (hai cách đọc đồng thuận)

| | S | IMI AUPRC |
|---|---|---|
| **uniform** | **0,689 ± 0,009** | **0,668 ± 0,008** |
| weighted | 0,673 ± 0,012 | 0,627 ± 0,003 |

S hơn 0,016 = **1,9 SE**; IMI AUPRC hơn 0,041 = **8,3 SE**. Dứt khoát.

Kết luận này lặp lại lần thứ tư trên bốn nền khác nhau (sweep cũ, nền noise_warp, nền multi_branch,
kiến trúc bài). Đủ vững để viết mạnh.

## Stage 4 — LOSS: **ASL** — đánh đổi có ý thức

| | S | IMI AUPRC | ILMI F1 | AMI F1 | Cond F1 |
|---|---|---|---|---|---|
| **asl** | **0,700 ± 0,017** | 0,660 ± 0,007 | **0,453 ± 0,042** | **0,214 ± 0,083** | 0,649 ± 0,005 |
| focal | 0,691 ± 0,020 | 0,654 ± 0,008 | 0,173 ± 0,074 | 0,000 | 0,522 ± 0,055 |
| posweight | 0,690 ± 0,010 | 0,644 ± 0,009 | 0,454 ± 0,058 | 0,170 ± 0,044 | 0,615 ± 0,034 |
| bce | 0,689 ± 0,009 | **0,668 ± 0,008** | 0,340 ± 0,006 | 0,056 ± 0,048 | **0,683 ± 0,019** |

S và `imi_auprc` đều **hoà** (1,0 SE và 1,3 SE). Nhưng ba khác biệt vượt nhiễu rõ ràng:

| asl vs bce | Biên | |
|---|---|---|
| ILMI F1 | +0,113 | **4,6 SE** |
| AMI F1 | +0,158 | **2,9 SE** |
| Cond macro F1 | −0,034 | **3,0 SE** |

ASL nâng AMI F1 từ 0,056 lên 0,214 — gần 4 lần — và ILMI từ 0,340 lên 0,453. Đúng hai nhãn định vị
MI khó nhất, đúng trọng tâm bài. Đổi lại mất 0,034 ở Conduction.

Chọn ASL theo S (tiêu chí khai trước), nhất quán với quyết định stage 1 nơi S cũng thắng `imi_auprc`.

`focal` loại dứt khoát: AMI F1 = **0,000**, Cond F1 0,522.

---

# Golden pipeline chốt

**zscore → lead_dropout → uniform → ASL** = `gpb10_s4_loss_asl`

## Bản thảo phải sửa

| Mục | Hiện tại | Đổi thành |
|---|---|---|
| Bảng 11, dòng Chuẩn hoá | Robust median/IQR | **zscore** |
| Bảng 11, dòng Hàm mất mát | BCEWithLogitsLoss | **ASL** |
| Bảng 5, Hàm mất mát | BCEWithLogitsLoss | **ASL** |
| Mục 3.3 | biện luận robust bền với ngoại lai | viết lại: trung vị/IQR bền với ngoại lai nhưng làm mất biên độ tương đối mô hình cần |
| Mục 3.6 | công thức BCE | thêm công thức ASL |
| Mục 4.2 | *"cơ chế tái cân bằng ở tầng loss dễ gây điều chỉnh dư thừa"* | **bác lại** — ASL không điều chỉnh dư thừa, nó cải thiện lớp hiếm rõ rệt |
| Bảng 11, dòng Tăng cường | Lead Dropout | giữ, nhưng ghi rõ **hoà với không augmentation** |
| Bảng 11, dòng Lấy mẫu | Uniform | giữ, kết luận mạnh hơn (8,3 SE) |

---

# GIAI ĐOẠN 0 — Setup

Bỏ qua nếu máy đã có dataset và repo.

```bash
pip install -q awscli && aws configure set default.s3.max_concurrent_requests 128 && aws configure set default.s3.max_queue_size 10000 && echo READY
```

```bash
time aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/records500/00000/ /tmp/s3test/ && du -sh /tmp/s3test
```

**Quá 60 giây → destroy, đổi máy.**

```bash
cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us; nvidia-smi --query-gpu=index,name --format=csv,noheader
```

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.zeros(1).cuda()+1)" | tee /workspace/env.txt
```

```bash
rm -rf /tmp/s3test && cd /workspace && git clone --depth 1 --single-branch -b GOLDEN13-IMI50 https://github.com/lynnhatsuki3010/multitask-ecg.git ECG_HRV && cd ECG_HRV && pip install wfdb neurokit2 scikit-learn scipy matplotlib tqdm PyYAML optuna iterative-stratification pandas
```

```bash
cd /workspace/ECG_HRV && mkdir -p data/raw/PTB-XL && time aws s3 sync --no-sign-request --exclude "records100/*" s3://physionet-open/ptb-xl/1.0.3/ data/raw/PTB-XL/ && du -sh data/raw/PTB-XL
```

## 0.1 Script chạy 4 GPU song song

```bash
cat > /workspace/run4.sh <<'EOF'
#!/bin/bash
# usage: run4.sh <seed> <config...>   -- 4 cau hinh cung luc, moi cai mot GPU
# OMP_NUM_THREADS=1: moi worker chay scipy filtfilt; de scipy tu mo luong OpenMP thi
# 56 worker x N luong tranh nhau 61 core.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
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

Chạy song song **không đổi kết quả**: mỗi run là tiến trình riêng, seed riêng, RNG độc lập — đã
kiểm chứng bit-identical trên cùng phần cứng.

---

# GIAI ĐOẠN 1 — Recipe sweep (33 run, ~1 giờ 40)

## 1.1 Sinh 11 config

```bash
cd /workspace/ECG_HRV && for s in norm aug samp loss; do
  case $s in
    norm) A="";;
    aug)  A="--norm robust";;
    samp) A="--norm robust --aug lead_dropout";;
    loss) A="--norm robust --aug lead_dropout --samp uniform";;
  esac
  python scripts/make_golden_stage.py --stage $s --base branche10_05_cond_depth --num-workers 14 --monitor imi_auprc $A
done && ls configs/experiments/gpb10_*.yaml | wc -l
```

Winner truyền vào lấy từ Bảng 11 hiện tại của bài. Chúng chỉ quyết định **nền** của stage sau;
kết luận đọc lại từ bảng mới.

## 1.2 Cửa ải — kiểm cả 11 config một lượt

```bash
cd /workspace/ECG_HRV && python -c "
import yaml, glob
want  = {'architecture':'decoupled_multitask','mi_head_mode':'contrast_v2','cond_layers':3,'cd_lead_out_dim':192}
wantt = {'num_workers':14,'monitor_metric':'imi_auprc','batch_size':96,'epochs':15}
bad = 0
files = sorted(glob.glob('configs/experiments/gpb10_*.yaml'))
for f in files:
    c = yaml.safe_load(open(f, encoding='utf-8'))
    for k, v in want.items():
        if c['model'].get(k) != v: print(f'{f}: model.{k}={c[\"model\"].get(k)!r} (can {v!r})'); bad += 1
    for k, v in wantt.items():
        if c['training'].get(k) != v: print(f'{f}: training.{k}={c[\"training\"].get(k)!r} (can {v!r})'); bad += 1
    if c['dataset'].get('label_threshold') != 50 or c['dataset'].get('label_threshold_overrides'):
        print(f'{f}: threshold lech'); bad += 1
print(f'{len(files)} config - ' + ('TAT CA DUNG' if bad == 0 else f'{bad} loi, DUNG LAI'))
"
```

Phải ra `11 config - TAT CA DUNG`.

## 1.3 Metadata

```bash
cd /workspace/ECG_HRV && ls -la data/processed_dir5_imi50_500hz/ 2>/dev/null || python scripts/01_build_metadata.py --config configs/experiments/gpb10_s1_norm_zscore.yaml --no-hrv
```

## 1.4 Smoke test

```bash
cd /workspace/ECG_HRV && python scripts/02_train.py --config configs/experiments/gpb10_s1_norm_zscore.yaml --debug && rm -rf checkpoints/*-debug && echo SMOKE_OK
```

`--debug` ép `num_workers=0` nên không đo được tốc độ thật — chỉ kiểm môi trường.

## 1.5 Chạy

```bash
CFG1="gpb10_s1_norm_zscore gpb10_s1_norm_robust gpb10_s1_norm_minmax gpb10_s2_aug_lead_dropout gpb10_s2_aug_mixup gpb10_s2_aug_noise_warp gpb10_s2_aug_random_crop gpb10_s3_samp_weighted gpb10_s4_loss_posweight gpb10_s4_loss_focal gpb10_s4_loss_asl"
```

```bash
/workspace/run4.sh 42 $CFG1
```

```bash
/workspace/run4.sh 123 $CFG1
```

```bash
/workspace/run4.sh 2024 $CFG1
```

Mỗi seed 11 config chia 3 đợt (4+4+3), ~33 phút. Ngắt kết nối vẫn chạy: `Ctrl+B`, thả, `D`.

Theo dõi — cả 4 GPU phải cùng bận:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

## 1.6 Đọc kết quả theo cả hai tiêu chí

```bash
cd /workspace/ECG_HRV && python scripts/compare_val_runs.py --glob "checkpoints/run_*gpb10_s*" --aggregate
```

Dòng đầu là **S**, dòng `IMI AUPRC` là cách đọc thứ hai. Chốt winner 4 stage.

## 1.7 Sao lưu — làm sau mỗi giai đoạn

```bash
cd /workspace/ECG_HRV && tar czf /workspace/metrics.tar.gz --exclude='*.pth' checkpoints/run_* configs/experiments/*.yaml /workspace/env.txt 2>/dev/null; ls -lh /workspace/metrics.tar.gz
```

---

# GIAI ĐOẠN 2 — Ablation kiến trúc ARCH-5→11 (24 run, ~1 giờ 12)

Chuỗi 8 bước, mỗi bước đổi **đúng một** thứ, recipe giữ nguyên từ Giai đoạn 1.

| Config | Bước | Đổi gì so với dòng trên |
|---|---|---|
| `abl_arch05_shared_tf` | ARCH-5 | Shared CNN + Shared TF ×2 (`soft`, `shared=2`) |
| `abl_arch06_no_shared_tf` | ARCH-6 | bỏ Shared TF (`shared=0`) |
| `abl_arch07_hard_graph` | ARCH-7 | `routing_mode: hard_graph` |
| `abl_arch08_subset_lead` | ARCH-8 | `routing_mode: subset_lead`, tắt cross-attention |
| `abl_arch09_decoupled` | ARCH-9 | `decoupled_multitask` — bỏ hẳn Shared CNN |
| `abl_arch10_contrast` | ARCH-10 | `mi_head_mode: contrast_v2` |
| `abl_arch10b_cond_dim` | ARCH-10b | `cd_lead_out_dim` 128 → 192 |
| `abl_arch11_cond_depth` | ARCH-11 | `cond_layers` 2 → 3 |

**ARCH-10b là bước bài đang gộp.** Tách ra vì `branche10_04` làm MI macro F1 tụt 0,511 → 0,495
trước khi `branche10_05` kéo lên — attribution mà dòng gộp che mất.

## 2.1 Sinh config — nền là winner AUG của Giai đoạn 1

Thay `<W2>` bằng winner stage AUG vừa chốt:

```bash
cd /workspace/ECG_HRV && python scripts/make_arch_ablation.py --base gpb10_s2_aug_<W2> --num-workers 14 --monitor imi_auprc
```

## 2.2 Cửa ải — điểm cuối phải trùng kiến trúc bài

```bash
cd /workspace/ECG_HRV && python -c "
import yaml
IGNORE = {'num_shared_layers','routing_mode'}
a = yaml.safe_load(open('configs/experiments/abl_arch11_cond_depth.yaml', encoding='utf-8'))['model']
b = yaml.safe_load(open('configs/experiments/branche10_05_cond_depth.yaml', encoding='utf-8'))['model']
bad = [k for k in sorted(set(a)|set(b)) if k not in IGNORE and a.get(k) != b.get(k)]
for k in bad: print(f'DIFF {k}: abl={a.get(k)!r}  paper={b.get(k)!r}')
print('ARCH-11 == kien truc bai' if not bad else f'{len(bad)} khac biet - DUNG LAI')
"
```

## 2.3 Chạy

```bash
CFG2="abl_arch05_shared_tf abl_arch06_no_shared_tf abl_arch07_hard_graph abl_arch08_subset_lead abl_arch09_decoupled abl_arch10_contrast abl_arch10b_cond_dim abl_arch11_cond_depth"
```

```bash
/workspace/run4.sh 42 $CFG2
```

```bash
/workspace/run4.sh 123 $CFG2
```

```bash
/workspace/run4.sh 2024 $CFG2
```

## 2.4 Bảng ablation

```bash
cd /workspace/ECG_HRV && python scripts/compare_val_runs.py --glob "checkpoints/run_*abl_arch*" --aggregate
```

**Tặng kèm:** ARCH-5 (shared CNN) và ARCH-9 (decoupled) chạy cùng recipe, cùng GPU, 3 seed — bằng
chứng trực tiếp cho luận điểm gradient conflict ở Mục 6, thứ review dòng 363 đòi.

---

# GIAI ĐOẠN 3 — Threshold sensitivity (6 run, ~18 phút)

Nhánh 50 lấy từ Giai đoạn 2, chỉ chạy 35 và 80.

```bash
cd /workspace/ECG_HRV && for T in 35 80; do
  sed -e "s/^  label_threshold: 50/  label_threshold: $T/" \
      -e "s#processed_dir5_imi50#processed_dir5_imi$T#g" \
      -e "s#splits_dir5_imi50#splits_dir5_imi$T#g" \
      -e "s/^  id: .*/  id: gpb10_thr_$T/" \
      configs/experiments/abl_arch11_cond_depth.yaml > configs/experiments/gpb10_thr_$T.yaml
  python scripts/01_build_metadata.py --config configs/experiments/gpb10_thr_$T.yaml --no-hrv
done && ls -d data/processed_dir5_imi*_500hz
```

```bash
for s in 42 123 2024; do /workspace/run4.sh $s gpb10_thr_35 gpb10_thr_80; done
```

Bảng đếm mẫu dương — số liệu bắt buộc cho bài:

```bash
cd /workspace/ECG_HRV && python -c "
import numpy as np, yaml
cfg = yaml.safe_load(open('configs/experiments/abl_arch11_cond_depth.yaml', encoding='utf-8'))
names = [l['name'] for l in cfg['labels']]
cols = {t: np.load(f'data/processed_dir5_imi{t}_500hz/label_matrix.npy').sum(axis=0).astype(int) for t in [35,50,80]}
print('label   ' + ''.join(f'{t:>10}' for t in [35,50,80]))
for i, n in enumerate(names): print(f'{n:8}' + ''.join(f'{cols[t][i]:>10}' for t in [35,50,80]))
" | tee /workspace/threshold_counts.txt
```

---

# GIAI ĐOẠN 4 — Độ phức tạp toàn chuỗi (0 run, ~5 phút)

Params / FLOPs / latency / VRAM cho **cả 8 kiến trúc** — bảng đánh đổi chính xác/chi phí mà review
dòng 296 đòi.

```bash
pip install -q thop && cd /workspace/ECG_HRV && python -c "
import sys, time, yaml, torch, glob; sys.path.insert(0,'/workspace/ECG_HRV')
from src.models.factory import build_model
try:
    from thop import profile; HAS=True
except ImportError: HAS=False
files = sorted(glob.glob('configs/experiments/abl_arch*.yaml'))
print(f\"{'config':26}{'params_M':>10}{'FLOPs_G':>10}{'lat_ms':>9}{'vram_MB':>10}\")
for f in files:
    cfg = yaml.safe_load(open(f, encoding='utf-8')); g = cfg['label_groups']
    m = build_model(cfg, len(g['arrhythmia']), len(g['mi']), 0, len(g['conduction']))
    p = sum(x.numel() for x in m.parameters())/1e6
    C, L = cfg['dataset']['num_leads'], cfg['dataset']['signal_length']
    fl = float('nan')
    if HAS:
        macs,_ = profile(m, inputs=(torch.randn(1,C,L),), verbose=False); fl = 2*macs/1e9
    m = m.cuda().eval(); torch.cuda.reset_peak_memory_stats()
    x = torch.randn(1,C,L).cuda()
    with torch.no_grad():
        for _ in range(10): m(x)
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(100): m(x)
        torch.cuda.synchronize(); t1=time.perf_counter()
    print(f\"{cfg['experiment']['id']:26}{p:>10.3f}{fl:>10.2f}{(t1-t0)/100*1000:>9.3f}{torch.cuda.max_memory_allocated()/1024**2:>10.1f}\")
    del m; torch.cuda.empty_cache()
print(f'gpu: {torch.cuda.get_device_name(0)}   torch: {torch.__version__}   cuda: {torch.version.cuda}')
" 2>&1 | tee /workspace/complexity_all.txt
```

---

# GIAI ĐOẠN 5 — Calibration (0 run, ~10 phút)

Chạy cho **mọi** run ablation và threshold — Bảng 7 cần số trước/sau cho từng bước.

```bash
cd /workspace/ECG_HRV && for d in $(ls -d checkpoints/run_*abl_arch* checkpoints/run_*gpb10_* 2>/dev/null); do
  echo "=== calibrate $d ==="
  python scripts/03_calibrate.py --dir $d
done 2>&1 | tee /workspace/calibration_all.txt
```

---

# GIAI ĐOẠN 6 — Kéo về

```bash
cd /workspace/ECG_HRV && cp /workspace/complexity_all.txt /workspace/threshold_counts.txt /workspace/calibration_all.txt /workspace/env.txt . 2>/dev/null; tar czf /workspace/final_metrics.tar.gz --exclude='*.pth' checkpoints/run_* configs/experiments/*.yaml *.txt 2>/dev/null; ls -lh /workspace/final_metrics.tar.gz
```

```bash
cd /workspace/ECG_HRV && tar cf /workspace/final_models.tar checkpoints/run_*/best_model.pth && sha256sum /workspace/final_models.tar | tee /workspace/final_models.sha && ls -lh /workspace/final_models.tar
```

[MÁY NHÀ] — tải, đối chiếu SHA256, giải nén, đếm đủ **rồi mới** destroy:

```powershell
(Get-FileHash "E:\KLTN\KL Project\ECG_HRV\final_models.tar" -Algorithm SHA256).Hash.ToLower(); Get-Content "E:\KLTN\KL Project\ECG_HRV\final_models.sha"
```

---

# Làm ở máy nhà — miễn phí

- [ ] **Zero-shot PTB + Georgia** — dữ liệu đã có sẵn; suy luận thuần nên chạy máy nào cũng được
- [ ] **Bootstrap CI** cho ILMI (38 mẫu test) và AMI (21 mẫu)
- [ ] **Vẽ gradient norm ratio** `grad_norm/mi` vs `grad_norm/backbone`, so ARCH-5 với ARCH-9
- [ ] **Đếm case XAI** theo TP/FP và theo nhãn
- [ ] **Đọc `src/data/label_builder.py`** — SCP khác ngoài 13 nhãn thì NORM = 0 hay 1 (review 315)
- [ ] **Đọc `src/training/trainer.py`** — có gradient clipping không, có AMP không (review 310, 320)
- [ ] **Sinh lại Bảng 3** từ `label_matrix.npy` — bảng hiện tại lệch 1–14 mẫu ở 7 nhãn
- [ ] **Loss curve** chứng minh hội tụ; thừa nhận `early_stopping_patience = 15` = `epochs` nên
      không kích hoạt, và `branche10_05` có best epoch = 15 (vẫn đang cải thiện khi dừng)

---

# Tổng

Ước tính ở 4 GPU song song, `nw=14`. Cơ sở: ~12 phút mỗi run, 4 run cùng lúc.

| Giai đoạn | Run | Thời gian |
|---|---|---|
| 0 Setup | 0 | 20 ph |
| 1 Recipe sweep | 33 | 1 h 40 |
| 2 Ablation kiến trúc | 24 | 1 h 12 |
| 3 Threshold | 6 | 18 ph |
| 4 Complexity | 0 | 5 ph |
| 5 Calibration | 0 | 10 ph |
| 6 Kéo về | 0 | 20 ph |
| **Tổng** | **63** | **~4 giờ** |

Ở $0,456/h ≈ **$1,8**.

## Sau chiến dịch — kiểm chứng miễn phí cho bài

`history.json` lưu mọi epoch, nên trả lời được phản biện *"sao không chọn checkpoint theo chỉ số
khác"* mà không train thêm:

```bash
cd /workspace/ECG_HRV && python scripts/compare_val_runs.py --glob "checkpoints/run_*abl_arch11*" --aggregate
```

Cột `S` và cột `IMI AUPRC` cho hai cách đọc; chênh lệch giữa chúng chính là cái giá của tiêu chí
đã chọn. Một dòng trong Mục 4 là đủ.
