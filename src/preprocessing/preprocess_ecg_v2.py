import pandas as pd
import numpy as np
import wfdb
import ast
import os
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt

print("="*60)
print("Preprocessing PTB-XL — 5 Superclasses (Primary-Label Version)")
print("="*60)

base_path = "D:\\ReliableMultimodalAI\\data\\raw\\ptbxl\\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1\\"

df = pd.read_csv(base_path + "ptbxl_database.csv")
agg_df = pd.read_csv(base_path + "scp_statements.csv", index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

class_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}

def get_primary_superclass(scp_codes_str):
    try:
        scp_codes = ast.literal_eval(scp_codes_str)
    except:
        return None
    # scp_codes = {code: likelihood}. Map each code to its superclass,
    # then pick the superclass whose code has the HIGHEST likelihood.
    best_class, best_score = None, -1
    for code, score in scp_codes.items():
        if code in agg_df.index:
            sc = agg_df.loc[code].diagnostic_class
            if sc in class_map and score > best_score:
                best_class, best_score = sc, score
    return best_class

df['label'] = df['scp_codes'].apply(get_primary_superclass)
df = df[df['label'].notna()]

print("\nClass distribution (primary-label method):")
print(df['label'].value_counts())
print(f"\nTotal usable records: {df.shape[0]}")

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=100.0, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    filtered = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        filtered[:, i] = filtfilt(b, a, signal[:, i])
    return filtered

def load_ecg_12lead(filename):
    try:
        record = wfdb.rdrecord(base_path + filename)
        signal = bandpass_filter(record.p_signal)
        if signal.shape[0] >= 1000:
            signal = signal[:1000, :]
        else:
            signal = np.vstack([signal, np.zeros((1000 - signal.shape[0], 12))])
        return signal.T
    except:
        return None

# Cap per class at 2500 (HYP will likely have fewer than this — that's fine, we take all of it)
print("\nLoading ECG signals — using FULL dataset, no cap...")
df_sample = df.reset_index(drop=True)
print(f"Target records: {len(df_sample)}")
print(f"Per-class counts going in: {df_sample['label'].value_counts().to_dict()}")

signals, labels = [], []
for i, (_, row) in enumerate(df_sample.iterrows()):
    sig = load_ecg_12lead(row['filename_lr'])
    if sig is not None:
        signals.append(sig)
        labels.append(class_map[row['label']])
    if (i+1) % 1000 == 0:
        print(f"  Processed {i+1}/{len(df_sample)}")

X = np.array(signals, dtype=np.float32)
y = np.array(labels)
print(f"\nFinal shape: {X.shape}")
print(f"Class counts: {np.bincount(y)}")

for lead in range(12):
    mean = X[:, lead, :].mean()
    std = X[:, lead, :].std() + 1e-8
    X[:, lead, :] = (X[:, lead, :] - mean) / std

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

save_dir = "D:\\ReliableMultimodalAI\\data\\splits"
os.makedirs(save_dir, exist_ok=True)
np.save(f"{save_dir}\\ecg5_X_train.npy", X_train)
np.save(f"{save_dir}\\ecg5_X_val.npy",   X_val)
np.save(f"{save_dir}\\ecg5_X_test.npy",  X_test)
np.save(f"{save_dir}\\ecg5_y_train.npy", y_train)
np.save(f"{save_dir}\\ecg5_y_val.npy",   y_val)
np.save(f"{save_dir}\\ecg5_y_test.npy",  y_test)

import json
with open(f"{save_dir}\\ecg5_class_map.json", 'w') as f:
    json.dump(class_map, f)

print(f"\nTrain: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print("Done!")