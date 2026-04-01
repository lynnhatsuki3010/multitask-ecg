"""
hrv_features.py
Compute HRV features (RMSSD, SDNN, mean HR) from ECG signals.
Uses neurokit2 for R-peak detection when available,
falls back to scipy.signal.find_peaks otherwise.
"""
import numpy as np
from typing import Dict, Optional, Tuple

try:
    import neurokit2 as nk
    _HAS_NK = True
except ImportError:
    _HAS_NK = False

from scipy.signal import find_peaks


def detect_rpeaks_scipy(signal_1d: np.ndarray, fs: int = 100) -> np.ndarray:
    """
    Fallback R-peak detector using scipy.
    Expects a single lead ECG (Lead II recommended).
    """
    # Estimate peak distance: assume HR between 30–200 bpm
    min_dist = int(fs * 60 / 200)  # 200 bpm max  → 0.3s
    # Threshold: 50% of signal range above baseline
    height = np.percentile(signal_1d, 75)
    peaks, _ = find_peaks(signal_1d, distance=min_dist, height=height)
    return peaks


def detect_rpeaks_nk(signal_1d: np.ndarray, fs: int = 100) -> np.ndarray:
    """
    R-peak detector using neurokit2 (Hamilton method, reliable).
    """
    try:
        signals, info = nk.ecg_peaks(signal_1d, sampling_rate=fs, method="hamilton2002")
        peaks = np.where(signals["ECG_R_Peaks"] == 1)[0]
        return peaks
    except Exception:
        # Fallback if nk fails on this sample
        return detect_rpeaks_scipy(signal_1d, fs)


def compute_rr_intervals(peaks: np.ndarray, fs: int = 100) -> np.ndarray:
    """Convert R-peak sample indices to RR intervals in milliseconds."""
    if len(peaks) < 2:
        return np.array([])
    rr_samples = np.diff(peaks)
    rr_ms = rr_samples / fs * 1000.0  # convert to ms
    # Filter physiologically plausible RR (300–2000 ms → 30–200 bpm)
    rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
    return rr_ms


def compute_hrv_features(
    signal_1d: np.ndarray,
    fs: int = 100,
    lead_name: str = "lead_II",
) -> Dict[str, float]:
    """
    Compute time-domain HRV features from a single-lead ECG signal.

    Args:
        signal_1d: 1D ECG signal array (samples,).
        fs: Sampling frequency in Hz.
        lead_name: Informational only.

    Returns:
        Dictionary with keys: rmssd, sdnn, mean_hr, num_rpeaks.
        Returns NaN values if detection fails.
    """
    nan_result = {"rmssd": np.nan, "sdnn": np.nan, "mean_hr": np.nan, "num_rpeaks": 0}

    try:
        # Detect R-peaks
        if _HAS_NK:
            peaks = detect_rpeaks_nk(signal_1d, fs)
        else:
            peaks = detect_rpeaks_scipy(signal_1d, fs)

        if len(peaks) < 3:
            return nan_result

        rr_ms = compute_rr_intervals(peaks, fs)
        if len(rr_ms) < 2:
            return nan_result

        rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
        sdnn = float(np.std(rr_ms, ddof=1))
        mean_hr = float(60000.0 / np.mean(rr_ms))  # bpm

        return {
            "rmssd":     rmssd,
            "sdnn":      sdnn,
            "mean_hr":   mean_hr,
            "num_rpeaks": len(peaks),
        }

    except Exception:
        return nan_result


def compute_hrv_batch(
    signals: np.ndarray,
    fs: int = 100,
    lead_idx: int = 1,  # Lead II (index 1 in standard 12-lead)
) -> Dict[str, np.ndarray]:
    """
    Compute HRV features for a batch of signals.

    Args:
        signals: np.ndarray of shape (N, 12, T) or (N, T).
        fs: Sampling frequency.
        lead_idx: Which lead to use for R-peak detection.

    Returns:
        Dict of arrays with keys: rmssd, sdnn, mean_hr, num_rpeaks.
        Each array has shape (N,).
    """
    if signals.ndim == 3:
        batch = signals[:, lead_idx, :]
    else:
        batch = signals

    n = len(batch)
    results = {k: np.full(n, np.nan) for k in ["rmssd", "sdnn", "mean_hr"]}
    results["num_rpeaks"] = np.zeros(n, dtype=int)

    for i, sig in enumerate(batch):
        feats = compute_hrv_features(sig, fs)
        for k in results:
            results[k][i] = feats[k]

    return results


if __name__ == "__main__":
    # Sanity check with synthetic signal
    fs = 100
    t = np.arange(0, 10, 1 / fs)
    # Synthetic ECG-like: spikes at ~1 Hz (60 bpm)
    sig = np.zeros(len(t))
    for beat in range(0, len(t), fs):  # every 100 samples = 1 beat
        if beat + 5 < len(t):
            sig[beat:beat+5] = [0.1, 0.5, 1.0, 0.3, 0.0]
    sig += np.random.normal(0, 0.02, len(t))

    feats = compute_hrv_features(sig, fs)
    print("HRV features from synthetic signal:")
    for k, v in feats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
