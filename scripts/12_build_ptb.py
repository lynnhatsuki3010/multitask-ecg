"""
12_build_ptb.py
─────────────────────────────────────────────────────────────────
Builds the PTB Diagnostic ECG Database dataset for cross-dataset validation.

Key differences from Georgia:
  - Uses WFDB .dat/.hea format (NOT .mat)
  - Sampling rate is 1000 Hz → must downsample to 500 Hz
  - Recording length is ~38 seconds → crop to first 10 seconds (5000 samples)
  - Labels are parsed from free-text fields in .hea header (not SNOMED codes)
  - Has 15 channels (12 standard + 3 Frank leads VX/VY/VZ) → keep only 12

Label Mapping Strategy:
  - "Reason for admission: Healthy control"   → NORM = 1
  - "Reason for admission: Myocardial infarction" + "inferior"/"infero" in localization → IMI = 1
  - "Reason for admission: Myocardial infarction" + "anterior"/"antero"/"septal" in localization → ASMI = 1
  - NOTE: PTB has no fine-grained Arrhythmia labels, only "Dysrhythmia".
          Arrhythmia labels (AFIB, STACH, PVC, AFLT) will all be 0.
          This is OK because we will FREEZE the Arrhythmia Head during fine-tuning.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from tqdm import tqdm

# PTB standard lead order in the .hea file:
# i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6 → 12 standard leads
# vx, vy, vz                                          → 3 Frank leads (skip)
STANDARD_LEAD_NAMES = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

TARGET_SR    = 500   # Hz - target sampling rate
TARGET_LEN   = 5000  # samples = 10 seconds at 500 Hz


def parse_hea(hea_path):
    """Parse a PTB .hea file and return metadata dict."""
    meta = {
        "hea_path": hea_path,
        "n_leads": None,
        "src_sr": None,
        "n_samples": None,
        "leads": [],        # list of (dat_file, gain, baseline, lead_name)
        "reason": None,
        "acute_loc": None,
        "former_loc": None,
    }

    with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # First header line: "record_name n_leads sr n_samples"
    first = lines[0].strip().split()
    meta["n_leads"]  = int(first[1])
    meta["src_sr"]   = int(first[2])
    meta["n_samples"] = int(first[3])

    # Signal lines (next n_leads lines)
    for line in lines[1: meta["n_leads"] + 1]:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        dat_file  = parts[0]
        gain_str  = parts[2]
        baseline  = float(parts[4])
        lead_name = parts[8].lower()
        gain = float(gain_str.split('/')[0]) if '/' in gain_str else float(gain_str)
        meta["leads"].append((dat_file, gain, baseline, lead_name))

    # Comment lines (# ...)
    for line in lines[meta["n_leads"] + 1:]:
        line = line.strip()
        m = re.match(r"#\s*Reason for admission:\s*(.*)", line, re.IGNORECASE)
        if m:
            meta["reason"] = m.group(1).strip().lower()
        m = re.match(r"#\s*Acute infarction \(localization\):\s*(.*)", line, re.IGNORECASE)
        if m:
            meta["acute_loc"] = m.group(1).strip().lower()
        m = re.match(r"#\s*Former infarction \(localization\):\s*(.*)", line, re.IGNORECASE)
        if m:
            meta["former_loc"] = m.group(1).strip().lower()

    return meta


def map_labels(meta):
    """
    Return dict with NORM, AFIB, STACH, PVC, AFLT, IMI, ASMI (all 0/1).
    
    Arrhythmia labels (AFIB, STACH, PVC, AFLT) are always 0 because PTB
    only has generic "Dysrhythmia" label with no sub-classification.
    """
    labels = {"NORM": 0, "AFIB": 0, "STACH": 0, "PVC": 0, "AFLT": 0, "IMI": 0, "ASMI": 0}

    reason = meta.get("reason") or ""
    acute  = meta.get("acute_loc") or ""
    former = meta.get("former_loc") or ""

    # --- NORM ---
    if "healthy" in reason:
        labels["NORM"] = 1

    # --- MI: need "Myocardial infarction" as reason ---
    if "myocardial infarction" in reason:
        # Combine acute + former localization text
        loc_text = f"{acute} {former}".lower()

        # IMI: inferior / infero / infero-lateral / infero-postero / etc.
        if re.search(r'\binfer', loc_text):
            labels["IMI"] = 1

        # ASMI: anterior / antero / anteroseptal / antero-lateral / septal
        if re.search(r'\banter|septal', loc_text):
            labels["ASMI"] = 1

        # Edge case: localization is blank/no/n_a but reason is MI → mark as generic MI (set both? skip?)
        # Decision: skip records where localization is truly unknown (no/n/a).
        # They are not usable for our specific label matching.
        if not any(labels[k] for k in ["IMI", "ASMI"]):
            # no specific location identified → not useful, will be filtered below
            pass

    return labels


def load_wfdb_signal(hea_path, meta):
    """
    Load a PTB record using pure numpy (no wfdb dependency).
    Returns signal as (n_std_leads, TARGET_LEN) float32 in mV, or None on failure.
    """
    record_dir = os.path.dirname(hea_path)
    record_base = os.path.splitext(os.path.basename(hea_path))[0]

    # Find the .dat file (signal data)
    dat_path = os.path.join(record_dir, f"{record_base}.dat")
    if not os.path.exists(dat_path):
        return None

    src_sr   = meta["src_sr"]
    n_samp   = meta["n_samples"]
    n_leads  = meta["n_leads"]

    try:
        # PTB uses int16, multiplexed interleaved format
        raw = np.fromfile(dat_path, dtype=np.int16)
        # reshape to (n_samples, n_leads)
        if raw.size < n_samp * n_leads:
            n_samp = raw.size // n_leads
        raw = raw[:n_samp * n_leads].reshape(n_samp, n_leads)
    except Exception:
        return None

    # Convert to mV using gain and baseline from header
    signal_mv = np.zeros_like(raw, dtype=np.float32)
    for ch_idx, (_, gain, baseline, _) in enumerate(meta["leads"]):
        if gain != 0:
            signal_mv[:, ch_idx] = (raw[:, ch_idx].astype(np.float32) - baseline) / gain

    # Select only the 12 standard leads (skip vx, vy, vz)
    std_indices = []
    for lead_name in STANDARD_LEAD_NAMES:
        for ch_idx, (_, _, _, name) in enumerate(meta["leads"]):
            if name == lead_name:
                std_indices.append(ch_idx)
                break

    if len(std_indices) != 12:
        return None  # can't find all 12 standard leads

    signal_mv = signal_mv[:, std_indices]   # (n_samp, 12)

    # ── Crop to first 10 seconds before resampling (faster) ──
    crop_src = min(n_samp, src_sr * 10)     # 10 s worth of samples at source SR
    signal_mv = signal_mv[:crop_src, :]

    # ── Downsample 1000 Hz → 500 Hz ──
    if src_sr != TARGET_SR:
        up   = TARGET_SR
        down = src_sr
        # resample_poly works per column; transpose to (12, n) then back
        signal_mv = resample_poly(signal_mv, up, down, axis=0).astype(np.float32)

    # ── Crop / pad to exactly TARGET_LEN ──
    if signal_mv.shape[0] > TARGET_LEN:
        signal_mv = signal_mv[:TARGET_LEN, :]
    elif signal_mv.shape[0] < TARGET_LEN:
        pad = TARGET_LEN - signal_mv.shape[0]
        signal_mv = np.pad(signal_mv, ((0, pad), (0, 0)), mode='constant')

    return signal_mv   # (5000, 12)


def build_ptb(raw_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Find all .hea files (one per recording)
    hea_files = sorted(glob.glob(os.path.join(raw_dir, "patient*", "*.hea")))
    print(f"Found {len(hea_files)} .hea files under {raw_dir}")

    all_signals  = []
    metadata_rows = []
    skipped = {"no_label": 0, "load_fail": 0, "no_loc": 0}

    for hea_path in tqdm(hea_files, desc="Building PTB"):
        meta = parse_hea(hea_path)

        labels = map_labels(meta)

        # We only keep:
        #   - Records with NORM = 1, OR
        #   - Records with IMI or ASMI = 1
        has_useful = labels["NORM"] or labels["IMI"] or labels["ASMI"]
        if not has_useful:
            skipped["no_label"] += 1
            continue

        signal = load_wfdb_signal(hea_path, meta)
        if signal is None:
            skipped["load_fail"] += 1
            continue

        patient_id = os.path.basename(os.path.dirname(hea_path))  # e.g. "patient001"
        ecg_id = os.path.splitext(os.path.basename(hea_path))[0]  # e.g. "s0010_re"

        row = {"ecg_id": ecg_id, "patient_id": patient_id}
        row.update(labels)
        metadata_rows.append(row)
        all_signals.append(signal)

    print(f"\nSkipped — no matching label: {skipped['no_label']}")
    print(f"Skipped — signal load failure: {skipped['load_fail']}")
    print(f"Total records kept: {len(metadata_rows)}")

    if len(all_signals) == 0:
        print("ERROR: No records kept! Check raw_dir path.")
        return

    # Save
    print("Saving features.npy ...")
    X = np.stack(all_signals)   # (N, 5000, 12)
    np.save(os.path.join(output_dir, "features.npy"), X)

    print("Saving labels.csv ...")
    df = pd.DataFrame(metadata_rows)
    df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

    print("\nLabel distribution:")
    label_cols = ["NORM", "AFIB", "STACH", "PVC", "AFLT", "IMI", "ASMI"]
    for col in label_cols:
        count = int(df[col].sum())
        print(f"  {col:<6}: {count:>4}  ({count/len(df)*100:.1f}%)")

    print(f"\nDone! Output: {output_dir}")
    print(f"Feature shape: {X.shape}")


if __name__ == "__main__":
    raw_dir    = r"e:\KLTN\KL Project\ECG_HRV\data\raw\PTB"
    output_dir = r"e:\KLTN\KL Project\ECG_HRV\data\processed_ptb"
    build_ptb(raw_dir, output_dir)
