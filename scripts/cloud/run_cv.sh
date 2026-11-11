#!/usr/bin/env bash
# run_cv.sh -- the whole protocol-v2 CV study (docs/PROTOCOL_CV_V2.md), unattended.
#
#   0. preflight  build the CV data if missing, check it against configs/cv/data_sha256.txt,
#                 record git commit + environment
#   1. phase 1    lr grid: every arm x lr in {1e-4, 3e-4, 1e-3} on fold 1 (train only)
#   2. select     scripts/29_select_lr.py: lowest validation loss per arm; the chosen
#                 fold-1 run is kept as the arm's fold-1 CV run, the rest are set aside
#   3. phase 2    calibrate + dump the chosen fold-1 runs; folds 2..10 at the chosen lr
#                 (train, calibrate, dump)
#   4. aggregate  scripts/28_cv_aggregate.py on all folds, and with fold 2 excluded
#
# One worker per GPU pulls jobs from a shared queue. Every job writes into a fixed
# folder (checkpoints/cv_v2/<arm>_f<fold>_lr<lr>), so the script can be started again
# after a crash or a reboot: finished work is skipped, unfinished work is redone.
#
# Usage (from anywhere):
#   bash scripts/cloud/run_cv.sh
# Environment:
#   GPUS="0 1 2 3"   GPUs to use (default: all that nvidia-smi lists)
#   PY=python        Python interpreter
#   DEBUG=1          smoke test: 2 arms, folds 1-2, 02_train --debug, separate folders
#   ALLOW_DIRTY=1    run although tracked code differs from the commit (not for the study)
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

# published backbones first: 30 epochs, longest jobs, best started early
ARMS_ALL="cv_xresnet1d101 cv_inception1d cv_ecgtransform cv_layout cv_decoupled cv_shared cv_partial_layout cv_contrast cv_pairsoft cv_hier"
LRS="1e-4 3e-4 1e-3"
if [ "${DEBUG:-0}" = 1 ]; then
  ARMS=${ARMS:-"cv_layout cv_xresnet1d101"}; FOLDS2="2"; TAG=cv_v2_debug
  TRAIN_EXTRA="--debug"; BOOT=50; AGG_EXTRA="--allow-missing"
else
  ARMS=$ARMS_ALL; FOLDS2="2 3 4 5 6 7 8 9 10"; TAG=cv_v2
  TRAIN_EXTRA=""; BOOT=2000; AGG_EXTRA=""
fi
CK=checkpoints/$TAG
GRID=checkpoints/${TAG}_lr_grid
LOG=logs/$TAG
RES=results/$TAG
mkdir -p "$CK" "$GRID" "$LOG" "$RES"
STATUS=$LOG/status.txt

note() { echo "$(date -u '+%F %T') $*" | tee -a "$STATUS"; }
die()  { note "ABORT $*"; exit 1; }
trap 'note "INTERRUPTED"; kill 0' INT TERM

# ----------------------------------------------------------------------------- 0
preflight() {
  note "=== preflight ($TAG) ==="
  local head; head=$(git rev-parse HEAD 2>/dev/null) || die "not a git checkout"
  if ! git diff --quiet HEAD -- src scripts configs; then
    if [ "${ALLOW_DIRTY:-0}" = 1 ] || [ "${DEBUG:-0}" = 1 ]; then
      note "WARNING tracked code differs from $head"
    else
      git status --short -- src scripts configs
      die "tracked code differs from commit $head (the registered protocol); set ALLOW_DIRTY=1 only for tests"
    fi
  fi
  [ -f data/raw/PTB-XL/ptbxl_database.csv ] || die "PTB-XL missing in data/raw/PTB-XL"
  if [ ! -f data/processed_mi_nolafb_500hz/label_matrix.npy ]; then
    note "building metadata (LBBB = CLBBB + ILBBB)"
    $PY scripts/01_build_metadata.py --config configs/cv/cv_layout.yaml --no-hrv > "$LOG/build_metadata.log" 2>&1 \
      || die "01_build_metadata failed, see $LOG/build_metadata.log"
  fi
  if [ ! -f data/splits_mi_nolafb_cvf10_500hz/test_indices.npy ]; then
    note "writing the ten CV splits"
    $PY scripts/26_make_cv_splits.py --config configs/cv/cv_layout.yaml > "$LOG/make_cv_splits.log" 2>&1 \
      || die "26_make_cv_splits failed, see $LOG/make_cv_splits.log"
  fi
  sha256sum -c configs/cv/data_sha256.txt --quiet > "$LOG/data_sha256_check.txt" 2>&1 \
    || die "CV data differ from configs/cv/data_sha256.txt, see $LOG/data_sha256_check.txt"
  note "data checksums OK"
  if [ -z "${GPUS:-}" ]; then GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' '); fi
  [ -n "${GPUS// /}" ] || die "no GPU found"
  local n_gpu; n_gpu=$($PY -c "import torch;print(torch.cuda.device_count())") || die "torch import failed"
  for g in $GPUS; do [ "$g" -lt "$n_gpu" ] || die "GPU $g not visible to torch ($n_gpu devices)"; done
  {
    echo "commit $head"; date -u
    nvidia-smi 2>&1; nproc 2>/dev/null
    $PY -c "import torch,numpy,scipy,sklearn;print('torch',torch.__version__,torch.version.cuda,'numpy',numpy.__version__,'scipy',scipy.__version__,'sklearn',sklearn.__version__)"
    $PY -m pip freeze 2>/dev/null
  } > "$LOG/env_$(date -u +%Y%m%d_%H%M%S).txt" 2>&1
  note "commit $head, GPUs: $GPUS, arms: $(echo $ARMS | wc -w)"
}

# ------------------------------------------------------------------- job queue
claim() {   # print the next line of queue $1, or nothing when it is empty
  local q=$1 n
  until mkdir "$q.lock" 2>/dev/null; do sleep 0.3; done
  n=$(cat "$q.next")
  echo $((n + 1)) > "$q.next"
  rmdir "$q.lock"
  sed -n "${n}p" "$q"
}

finished_training() { [ -f "$1/best_model.pth" ] && [ -f "$1/test_metrics.json" ]; }

run_job() {   # gpu mode arm fold seed lr ;  mode = train | post | full
  local g=$1 mode=$2 arm=$3 k=$4 s=$5 lr=$6
  local name=${arm}_f${k}_lr${lr}
  local d=$CK/$name lg=$LOG/$name
  export CUDA_VISIBLE_DEVICES=$g
  if [ "$mode" != post ]; then
    if finished_training "$d" || { [ "$mode" = train ] && finished_training "$GRID/$name"; }; then
      note "SKIP-TRAIN $name (finished earlier)"
    else
      rm -rf "$d"
      note "START $name mode=$mode gpu=$g seed=$s"
      $PY scripts/02_train.py --config "configs/cv/$arm.yaml" --cv-fold "$k" --seed "$s" --lr "$lr" \
        --run-name "$TAG/$name" $TRAIN_EXTRA > "$lg.train.log" 2>&1
      local rc=$?
      if [ $rc -ne 0 ] || ! finished_training "$d"; then
        note "FAIL train $name rc=$rc (see $lg.train.log)"; return 1
      fi
      note "TRAINED $name $(grep -m1 'Trainable parameters' "$lg.train.log" | tr -s ' ')"
    fi
  fi
  [ "$mode" = train ] && return 0
  if [ -f "$d/predictions.npz" ]; then note "SKIP-POST $name (done earlier)"; return 0; fi
  $PY scripts/03_calibrate.py --dir "$d" > "$lg.calibrate.log" 2>&1 \
    || { note "FAIL calibrate $name (see $lg.calibrate.log)"; return 1; }
  $PY scripts/27_dump_predictions.py --dir "$d" > "$lg.dump.log" 2>&1 && [ -f "$d/predictions.npz" ] \
    || { note "FAIL dump $name (see $lg.dump.log)"; return 1; }
  note "DONE $name"
}

worker() {   # gpu queue slot
  local g=$1 q=$2 line
  sleep $(( $3 * 20 ))          # stagger the first data loads
  while line=$(claim "$q"); [ -n "$line" ]; do
    # shellcheck disable=SC2086
    run_job "$g" $line || true
  done
  note "WORKER gpu$g finished $(basename "$q")"
}

run_phase() {   # queue file
  local q=$1 slot=0 pids=()
  rm -rf "$q.lock"; echo 1 > "$q.next"
  note "=== $(basename "$q"): $(grep -c . "$q") jobs on GPUs $GPUS ==="
  for g in $GPUS; do
    worker "$g" "$q" "$slot" >> "$LOG/worker_gpu$g.log" 2>&1 &
    pids+=($!); slot=$((slot + 1))
  done
  wait "${pids[@]}"
}

# ------------------------------------------------------------------------ main
STATUS_START=$( [ -f "$STATUS" ] && wc -l < "$STATUS" || echo 0 )
preflight

Q1=$LOG/jobs_phase1.txt
: > "$Q1"
for a in $ARMS; do for lr in $LRS; do echo "train $a 1 3001 $lr" >> "$Q1"; done; done
run_phase "$Q1"

note "=== lr selection ==="
# shellcheck disable=SC2086
$PY scripts/29_select_lr.py --root "$CK" --grid-dir "$GRID" --arms $ARMS --lrs $LRS --folds $FOLDS2 \
  --choice "$LOG/lr_choice.json" --jobs "$LOG/jobs_phase2.txt" 2>&1 | tee -a "$STATUS"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "29_select_lr failed"

run_phase "$LOG/jobs_phase2.txt"

n_done=$(ls "$CK"/*/predictions.npz 2>/dev/null | wc -l)
n_want=$(grep -c . "$LOG/jobs_phase2.txt")
note "=== runs with predictions: $n_done / $n_want ==="

note "=== aggregate ==="
# shellcheck disable=SC2086
$PY scripts/28_cv_aggregate.py --glob "$CK/*" --reference cv_layout --boot $BOOT $AGG_EXTRA \
  --out "$RES/cv_results.md" --csv "$RES/cv_results.csv" > "$LOG/aggregate.log" 2>&1 \
  && note "wrote $RES/cv_results.md" || note "FAIL aggregate (see $LOG/aggregate.log)"
# shellcheck disable=SC2086
$PY scripts/28_cv_aggregate.py --glob "$CK/*" --reference cv_layout --boot $BOOT $AGG_EXTRA --exclude-folds 2 \
  --out "$RES/cv_results_no_fold2.md" --csv "$RES/cv_results_no_fold2.csv" > "$LOG/aggregate_no_fold2.log" 2>&1 \
  && note "wrote $RES/cv_results_no_fold2.md" || note "FAIL aggregate without fold 2 (see $LOG/aggregate_no_fold2.log)"

n_fail=$(tail -n +$((STATUS_START + 1)) "$STATUS" | grep -c " FAIL " || true)
note "=== ALL FINISHED: $n_done/$n_want runs, FAIL lines in this attempt: $n_fail ==="
