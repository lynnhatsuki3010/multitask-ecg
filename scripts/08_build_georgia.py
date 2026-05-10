import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

# Mapping from SNOMED-CT to our 7 classes
# Lưu ý: Georgia KHÔNG CÓ nhãn IMI/ASMI cụ thể. Nó chỉ có 7 mẫu Myocardial Infarction (164865005).
# Tuy nhiên, nó có nhãn Ischaemia (Thiếu máu cục bộ). Ta sẽ dùng Ischaemia làm Proxy (vật thế thân) cho Infarction.
SNOMED_MAPPING = {
    # --- Arrhythmia Head ---
    "426783006": "NORM",  # Sinus rhythm
    "164889003": "AFIB",  # Atrial fibrillation
    "427084000": "STACH", # Sinus tachycardia
    "164890007": "AFLT",  # Atrial flutter
    # 3 mã dưới đây đều chỉ chung một hiện tượng sinh lý là Ngoại tâm thu thất (PVC/VPB)
    "17338001":  "PVC",   # Ventricular premature beats
    "164884008": "PVC",   # Ventricular ectopics
    "427172004": "PVC",   # Premature ventricular contractions
    
    # --- MI Head (PROXY MAPPING) ---
    "425419005": "IMI",   # Inferior Ischaemia -> Proxy cho Inferior Myocardial Infarction
    "426434006": "ASMI",  # Anterior Ischaemia -> Proxy cho Anteroseptal Myocardial Infarction
}

def load_hea_labels(hea_path):
    labels = set()
    with open(hea_path, 'r') as f:
        for line in f:
            if line.startswith("#Dx:"):
                codes = line.strip().split("#Dx:")[1].strip().split(",")
                for code in codes:
                    code = code.strip()
                    if code in SNOMED_MAPPING:
                        labels.add(SNOMED_MAPPING[code])
    return list(labels)

def load_mat_signal(mat_path):
    mat_data = sio.loadmat(mat_path)
    # The key is usually 'val' for PhysioNet MAT files
    if 'val' in mat_data:
        signal = mat_data['val'] # shape (leads, time)
        # Convert to mV. Standard PhysioNet is usually int16, gain varies.
        # But for deep learning, we standardize it.
        # Let's read the gain from .hea file if needed, or assume standard scaling.
        # PTB-XL build script uses Wfdb but since we have .mat, let's use standard scaling.
        # Actually, it's safer to read the gain from the HEA file.
        pass
    
    # We should parse the HEA file properly for gain.
    gains = []
    baselines = []
    with open(mat_path.replace(".mat", ".hea"), 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            if line.startswith("#"): continue
            parts = line.strip().split()
            if len(parts) >= 8:
                gain = float(parts[2].split('/')[0]) if '/' in parts[2] else float(parts[2])
                baseline = float(parts[4])
                gains.append(gain)
                baselines.append(baseline)
                
    if 'val' in mat_data:
        signal = mat_data['val'].astype(np.float32)
        for i in range(len(gains)):
            if gains[i] > 0:
                signal[i] = (signal[i] - baselines[i]) / gains[i]
        
        # Transpose to (time, leads) to match PTB-XL (5000, 12)
        signal = signal.T
        
        # Truncate or pad to 5000
        if signal.shape[0] > 5000:
            signal = signal[:5000, :]
        elif signal.shape[0] < 5000:
            pad_len = 5000 - signal.shape[0]
            signal = np.pad(signal, ((0, pad_len), (0, 0)), mode='constant')
            
        return signal
    return None

def build_georgia(raw_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    hea_files = glob.glob(os.path.join(raw_dir, "*.hea"))
    
    all_signals = []
    metadata = []
    
    print("Processing Georgia dataset...")
    for hea_path in tqdm(hea_files):
        mat_path = hea_path.replace(".hea", ".mat")
        if not os.path.exists(mat_path):
            continue
            
        labels = load_hea_labels(hea_path)
        # Only keep records that have at least one of our target classes
        if not labels:
            continue
            
        signal = load_mat_signal(mat_path)
        if signal is not None:
            ecg_id = os.path.basename(hea_path).split(".")[0]
            all_signals.append(signal)
            
            # Create dummy multi-hot labels for Arrhythmia
            meta = {"ecg_id": ecg_id}
            meta["patient_id"] = ecg_id # Georgia doesn't specify patient ID, use ecg_id
            
            for cls in ["NORM", "AFIB", "STACH", "PVC", "AFLT", "IMI", "ASMI"]:
                meta[cls] = 1 if cls in labels else 0
                
            metadata.append(meta)
            
    # Save as numpy array
    print("Saving signals...")
    X = np.stack(all_signals)
    np.save(os.path.join(output_dir, "features.npy"), X)
    
    print("Saving metadata...")
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)
    
    print(f"Build complete. Total samples kept: {len(df)}")
    print("Label distribution in processed Georgia dataset:")
    print(df[["NORM", "AFIB", "STACH", "PVC", "AFLT", "IMI", "ASMI"]].sum())

if __name__ == "__main__":
    raw_dir = r"e:\KLTN\KL Project\ECG_HRV\data\raw\Georgia"
    output_dir = r"e:\KLTN\KL Project\ECG_HRV\data\processed_georgia"
    build_georgia(raw_dir, output_dir)
