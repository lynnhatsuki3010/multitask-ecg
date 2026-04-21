"""
preprocessing.py
Load, filter, normalize ECG signals from PTB-XL, and provide
a PyTorch Dataset class for training.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from typing import List, Optional, Tuple, Dict
import wfdb


# ─── Signal Filtering ────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """Butterworth bandpass filter applied zero-phase (forward-backward)."""
    nyq = fs / 2
    low_norm = low / nyq
    high_norm = high / nyq
    low_norm = max(1e-4, min(low_norm, 0.999))
    high_norm = max(1e-4, min(high_norm, 0.999))
    sos = butter(4, [low_norm, high_norm], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal, axis=0)


def notch_filter(signal: np.ndarray, fs: float, freq: float = 50.0, Q: float = 30.0) -> np.ndarray:
    """Notch filter to remove powerline interference (50 or 60 Hz)."""
    nyq = fs / 2
    if freq >= nyq:
        return signal
    b, a = iirnotch(freq / nyq, Q)
    return filtfilt(b, a, signal, axis=0)


def normalize_signal(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize ECG signal per lead.
    signal shape: (T, 12) or (12, T).
    """
    if method == "zscore":
        mean = signal.mean(axis=-1, keepdims=True)
        std  = signal.std(axis=-1, keepdims=True)
        std  = np.where(std < 1e-8, 1.0, std)
        return (signal - mean) / std
    elif method == "minmax":
        vmin = signal.min(axis=-1, keepdims=True)
        vmax = signal.max(axis=-1, keepdims=True)
        denom = np.where((vmax - vmin) < 1e-8, 1.0, vmax - vmin)
        return (signal - vmin) / denom
    else:
        return signal


# ─── Signal Loading ───────────────────────────────────────────────────────────

def load_signal(
    filename: str,
    base_path: str,
    target_length: int = 1000,
    fs: float = 100.0,
    bandpass: Tuple[float, float] = (0.5, 40.0),
    notch: Optional[float] = 50.0,
    normalize: str = "zscore",
) -> np.ndarray:
    """
    Load a single ECG record, apply preprocessing, and return
    a float32 array of shape (12, T).

    Args:
        filename: Relative path to the record (e.g. 'records100/00000/00001_lr').
        base_path: Root directory of PTB-XL dataset.
        target_length: Number of samples to keep/pad.
        fs: Sampling frequency.
        bandpass: (low_hz, high_hz) for bandpass filter. None to skip.
        notch: Frequency for notch filter. None to skip.
        normalize: 'zscore', 'minmax', or 'none'.
    """
    full_path = os.path.join(base_path, filename)
    record = wfdb.rdsamp(full_path)
    signal = record[0].astype(np.float32)  # shape: (T, 12)

    # Replace NaN/Inf with 0
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # Bandpass filter
    if bandpass is not None:
        signal = bandpass_filter(signal, fs, low=bandpass[0], high=bandpass[1])

    # Notch filter
    if notch is not None:
        signal = notch_filter(signal, fs, freq=notch)

    # Clip to target length or pad
    T = signal.shape[0]
    if T >= target_length:
        signal = signal[:target_length, :]
    else:
        pad = np.zeros((target_length - T, signal.shape[1]), dtype=np.float32)
        signal = np.concatenate([signal, pad], axis=0)

    # Transpose: (T, 12) → (12, T)
    signal = signal.T

    # Normalize
    signal = normalize_signal(signal, method=normalize)

    return signal.astype(np.float32)


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────

class PTBXLDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL.

    Returns:
        signal: torch.Tensor of shape (12, T)
        labels: torch.Tensor of shape (num_labels,)  — binary multi-label
        hrv:    torch.Tensor of shape (3,)            — [rmssd, sdnn, mean_hr]
                (NaN values are zero-filled at batch time)
        ecg_id: int
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        label_matrix: np.ndarray,
        hrv_matrix: Optional[np.ndarray],
        base_path: str,
        sampling_rate: int = 100,
        target_length: int = 1000,
        bandpass: Tuple[float, float] = (0.5, 40.0),
        notch: Optional[float] = 50.0,
        normalize: str = "zscore",
        augment: bool = False,
    ):
        self.metadata = metadata.reset_index()
        self.label_matrix = label_matrix
        self.hrv_matrix = hrv_matrix  # shape (N, 3) or None
        self.base_path = base_path
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.bandpass = bandpass
        self.notch = notch
        self.normalize = normalize
        self.augment = augment

        # Choose filename column based on sampling rate
        self.filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]
        filename = row[self.filename_col]

        signal = load_signal(
            filename=filename,
            base_path=self.base_path,
            target_length=self.target_length,
            fs=float(self.sampling_rate),
            bandpass=self.bandpass,
            notch=self.notch,
            normalize=self.normalize,
        )

        time_warped = False
        if self.augment:
            signal, time_warped = self._augment(signal)

        labels = self.label_matrix[idx]  # (num_labels,)

        if self.hrv_matrix is not None:
            hrv = self.hrv_matrix[idx].copy()
            # time_warp changes R-R intervals → HRV labels no longer match warped signal
            hrv_valid = (not np.isnan(hrv).any()) and (not time_warped)
            hrv = np.nan_to_num(hrv, nan=0.0)
        else:
            hrv = np.zeros(3, dtype=np.float32)
            hrv_valid = False

        return {
            "signal":    torch.from_numpy(signal),
            "labels":    torch.from_numpy(labels),
            "hrv":       torch.from_numpy(hrv.astype(np.float32)),
            "hrv_valid": torch.tensor(hrv_valid, dtype=torch.bool),
            "ecg_id":    int(row.get("ecg_id", idx)),
        }

    def _augment(self, signal: np.ndarray):
        """
        Apply random augmentations. Returns (signal, time_warped).
        time_warped=True means HRV ground truth is no longer valid
        (time_warp changes R-R intervals making RMSSD/SDNN labels stale).
        signal shape: (12, T)

        Baseline (always on):
          - Gaussian noise + amplitude scaling

        Optional (controlled by self.aug_* flags):
          - aug_lead_dropout    : randomly zero out 1-2 leads
          - aug_random_crop     : crop 80% window then resize back
          - aug_baseline_wander : low-freq sinusoidal drift (HRV-safe)
          - aug_time_warp       : time-axis stretching (INVALIDATES HRV labels)
        """
        time_warped = False
        # ── Always-on baseline ────────────────────────────────────────────────
        scale = np.random.uniform(0.9, 1.1)
        signal = signal * scale
        signal = signal + np.random.normal(0, 0.01, signal.shape).astype(np.float32)

        # ── Lead Dropout (Aug #1) ─────────────────────────────────────────────
        # Randomly zero out 1–2 leads to simulate poor electrode contact.
        if getattr(self, "aug_lead_dropout", False):
            n_drop = np.random.randint(1, 3)          # drop 1 or 2 leads
            drop_leads = np.random.choice(12, n_drop, replace=False)
            signal[drop_leads, :] = 0.0

        # ── Random Crop (Aug #2) ──────────────────────────────────────────────
        # Take a random 80% window then resize back to original length,
        # simulating electrode placement at slightly different time offsets.
        if getattr(self, "aug_random_crop", False):
            T = signal.shape[1]
            crop_len = int(T * 0.8)
            start = np.random.randint(0, T - crop_len + 1)
            cropped = signal[:, start: start + crop_len]       # (12, crop_len)
            # Resize back with linear interpolation
            indices = np.linspace(0, crop_len - 1, T)
            xi = np.arange(crop_len)
            signal = np.stack([
                np.interp(indices, xi, cropped[c]) for c in range(12)
            ], axis=0).astype(np.float32)

        # ── Baseline Wander (Aug #3) ──────────────────────────────────────────
        # Add low-frequency sinusoidal drift (0.05–0.5 Hz) simulating
        # patient breathing or body movement artifact.
        if getattr(self, "aug_baseline_wander", False):
            T = signal.shape[1]
            t = np.arange(T) / self.sampling_rate
            freq = np.random.uniform(0.05, 0.5)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.05, 0.15)
            wander = (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
            signal = signal + wander[np.newaxis, :]   # broadcast to (12, T)

        # ── Time Warp (Aug #4) ────────────────────────────────────────────────
        # Mild non-uniform time stretching: split signal into segments,
        # stretch/compress each slightly, then resample back to T.
        if getattr(self, "aug_time_warp", False):
            time_warped = True   # HRV labels no longer valid after warp
            T = signal.shape[1]
            n_knots = 4
            orig_steps = np.linspace(0, T - 1, n_knots + 2)
            warped_steps = orig_steps.copy()
            for i in range(1, n_knots + 1):
                warped_steps[i] += np.random.uniform(-T * 0.05, T * 0.05)
            warped_steps = np.clip(warped_steps, 0, T - 1)
            warped_steps = np.sort(warped_steps)

            new_indices = np.interp(
                np.arange(T),
                np.linspace(0, T - 1, len(warped_steps)),
                warped_steps
            )
            signal = np.stack([
                np.interp(new_indices, np.arange(T), signal[c])
                for c in range(12)
            ], axis=0).astype(np.float32)

        return signal, time_warped


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate that stacks all fields."""
    return {
        "signal":    torch.stack([b["signal"]    for b in batch]),
        "labels":    torch.stack([b["labels"]    for b in batch]),
        "hrv":       torch.stack([b["hrv"]       for b in batch]),
        "hrv_valid": torch.stack([b["hrv_valid"] for b in batch]),
        "ecg_id":    torch.tensor([b["ecg_id"]   for b in batch]),
    }
