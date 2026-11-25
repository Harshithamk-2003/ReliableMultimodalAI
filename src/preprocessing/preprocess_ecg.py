import pandas as pd
import numpy as np
import wfdb
import ast
import os
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt

print("="*60)
print("Preprocessing PTB-XL — 12 Lead ECG")
print("="*60)

base_path = "D:\\ReliableMultimodalAI\\data\\raw\\ptbxl\\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1\\"

df = pd.read_csv(base_path + "ptbxl_database.csv")

def get_primary_label(scp_codes_str):
    try:
        scp_codes = ast.literal_eval(scp_codes_str)
        best_code = max(scp_codes, key=scp_codes.get)
        return best_code
    except:
        return None

df['primary_label'] = df['scp_codes'].apply(get_primary_label)

# Check what classes exist and how many samples
print("\nAll available classes:")
print(df['primary_label'].value_counts().head(20))

# Keep only classes with at least 100 samples
class_counts = df['primary_label'].value_counts()
valid_classes = class_counts[class_counts >= 100].index.tolist()
print(f"\nClasses with 100+ samples: {valid_classes}")

df = df[df['primary_label'].isin(valid_classes)]

# Create class map from valid classes
class_map = {label: idx for idx, label in enumerate(sorted(valid_classes))}
print(f"\nFinal class map: {class_map}")
print(f"Total classes: {len(class_map)}")
print(f"Total records: {df.shape[0]}")

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=100.0, order=4):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        filtered[:, i] = filtfilt(b, a, signal[:, i])
    return filtered

def load_ecg_12lead(filename):
    try:
        path = base_path + filename
        record = wfdb.rdrecord(path)
        signal = record.p_signal
        signal = bandpass_filter(signal)
        if signal.shape[0] >= 1000:
            signal = signal[:1000, :]
        else:
            pad = np.zeros((1000 - signal.shape[0], 12))
            signal = np.vstack([signal, pad])
        return signal.T  # (12, 1000)
    except:
        return None

# Load up to 500 per class
print("\nLoading ECG signals...")
df_sample = df.groupby('primary_label').head(500)
print(f"Target records: {len(df_sample)}")

signals = []
labels = []
skipped = 0

for i, (_, row) in enumerate(df_sample.iterrows()):
    sig = load_ecg_12lead(row['filename_lr'])
    if sig is not None:
        signals.append(sig)
        labels.append(class_map[row['primary_label']])
    else:
        skipped += 1
    if (i+1) % 500 == 0:
        print(f"  Processed {i+1}/{len(df_sample)}, skipped {skipped}")

X = np.array(signals, dtype=np.float32)
y = np.array(labels)

print(f"\nFinal shape: {X.shape}")
print(f"Class counts: {np.bincount(y)}")
print(f"Number of classes: {len(np.unique(y))}")

# Normalize each lead
for lead in range(12):
    mean = X[:, lead, :].mean()
    std = X[:, lead, :].std() + 1e-8
    X[:, lead, :] = (X[:, lead, :] - mean) / std

# Split without stratify first, then check
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Save
save_dir = "D:\\ReliableMultimodalAI\\data\\splits"
os.makedirs(save_dir, exist_ok=True)
np.save(f"{save_dir}\\ecg_X_train.npy", X_train)
np.save(f"{save_dir}\\ecg_X_val.npy",   X_val)
np.save(f"{save_dir}\\ecg_X_test.npy",  X_test)
np.save(f"{save_dir}\\ecg_y_train.npy", y_train)
np.save(f"{save_dir}\\ecg_y_val.npy",   y_val)
np.save(f"{save_dir}\\ecg_y_test.npy",  y_test)

# Save class map for later use
import json
with open(f"{save_dir}\\ecg_class_map.json", 'w') as f:
    json.dump(class_map, f)

print(f"\nTrain: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")
print(f"\nClass map saved to splits/ecg_class_map.json")
print("ECG preprocessing complete!")