import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys, os

sys.path.append("D:\\ReliableMultimodalAI\\src\\fusion")
from multimodal_fusion import (
    glucose_model, glucose_scaler, bp_model, bp_scaler,
    ecg_models, ecg_class_map, device, fuse_modalities
)

MODELS_DIR = "D:\\ReliableMultimodalAI\\models_saved\\"
DATA_DIR   = "D:\\ReliableMultimodalAI\\data\\"
np.random.seed(42)

print("="*60)
print("Loading test data for all 3 modalities")
print("="*60)

# ── BP test set — recreate exact same split used in training ──
df_bp = pd.read_csv(DATA_DIR + "raw\\uci_heart.csv")
df_bp['ca'] = df_bp['ca'].fillna(df_bp['ca'].median())
df_bp['thal'] = df_bp['thal'].fillna(df_bp['thal'].median())
df_bp['target'] = (df_bp['target'] > 0).astype(int)
feature_cols = [c for c in df_bp.columns if c != 'target']
X_bp_all = df_bp[feature_cols].values
y_bp_all = df_bp['target'].values
X_bp_scaled_all = bp_scaler.transform(X_bp_all)
_, X_bp_test, _, y_bp_test = train_test_split(
    X_bp_scaled_all, y_bp_all, test_size=0.2, random_state=42, stratify=y_bp_all)
print(f"BP test set: {X_bp_test.shape}")

# ── Glucose test set — already preprocessed and scaled ──
X_glucose_test = np.load(DATA_DIR + "splits\\glucose_X_test.npy")
y_glucose_test = np.load(DATA_DIR + "splits\\glucose_y_test.npy")
print(f"Glucose test set: {X_glucose_test.shape}")

# ── ECG test set — already preprocessed ──
X_ecg_test = np.load(DATA_DIR + "splits\\ecg5_X_test.npy")
y_ecg_test = np.load(DATA_DIR + "splits\\ecg5_y_test.npy")
print(f"ECG test set: {X_ecg_test.shape}")

print("\n" + "="*60)
print("Running batch inference on all test sets")
print("="*60)

# ── Glucose predictions (batch) ──
glucose_probs = glucose_model.predict_proba(X_glucose_test)[:, 1]
glucose_conf  = np.maximum(glucose_probs, 1 - glucose_probs)
print("Glucose predictions done.")

# ── BP predictions (batch) ──
bp_probs = bp_model.predict_proba(X_bp_test)[:, 1]
bp_conf  = np.maximum(bp_probs, 1 - bp_probs)
print("BP predictions done.")

# ── ECG predictions (batch, in chunks to save memory) ──
ecg_risk_probs = []
ecg_confidences = []
batch_size = 128
with torch.no_grad():
    for i in range(0, len(X_ecg_test), batch_size):
        batch = torch.FloatTensor(X_ecg_test[i:i+batch_size]).to(device)
        probs_sum = torch.zeros(batch.shape[0], 5).to(device)
        for m in ecg_models:
            probs_sum += torch.softmax(m(batch), dim=1)
        probs = (probs_sum / len(ecg_models)).cpu().numpy()
        risk = 1 - probs[:, ecg_class_map['NORM']]
        conf = probs.max(axis=1)
        ecg_risk_probs.extend(risk)
        ecg_confidences.extend(conf)
        if (i // batch_size) % 10 == 0:
            print(f"  ECG batch {i}/{len(X_ecg_test)}")

ecg_risk_probs = np.array(ecg_risk_probs)
ecg_confidences = np.array(ecg_confidences)
ecg_ground_truth = (y_ecg_test != ecg_class_map['NORM']).astype(int)  # 1 = abnormal
print("ECG predictions done.\n")

# ──────────────────────────────────────────────────────────
# Build N synthetic multimodal patients — CLASS-CONSISTENT pairing
# (all 3 readings match the same underlying true condition)
# ──────────────────────────────────────────────────────────

N = 500  # number of synthetic multimodal test cases, balanced 50/50

ecg_pos_idx = np.where(ecg_ground_truth == 1)[0]
ecg_neg_idx = np.where(ecg_ground_truth == 0)[0]
bp_pos_idx  = np.where(y_bp_test == 1)[0]
bp_neg_idx  = np.where(y_bp_test == 0)[0]
glu_pos_idx = np.where(y_glucose_test == 1)[0]
glu_neg_idx = np.where(y_glucose_test == 0)[0]

combined_truth = np.zeros(N, dtype=int)
idx_ecg = np.zeros(N, dtype=int)
idx_bp = np.zeros(N, dtype=int)
idx_glucose = np.zeros(N, dtype=int)

for i in range(N):
    is_diseased = np.random.rand() < 0.5
    combined_truth[i] = int(is_diseased)

    if is_diseased:
        idx_ecg[i]     = np.random.choice(ecg_pos_idx)
        idx_bp[i]      = np.random.choice(bp_pos_idx)
        idx_glucose[i] = np.random.choice(glu_pos_idx)
    else:
        idx_ecg[i]     = np.random.choice(ecg_neg_idx)
        idx_bp[i]      = np.random.choice(bp_neg_idx)
        idx_glucose[i] = np.random.choice(glu_neg_idx)

print(f"Built {N} synthetic multimodal test cases (class-consistent pairing).")
print(f"Combined risk-positive cases: {combined_truth.sum()} / {N}\n")

# ──────────────────────────────────────────────────────────
# Run all 4 configurations
# ──────────────────────────────────────────────────────────

def evaluate_configuration(name, use_ecg, use_bp, use_glucose):
    fused_scores = []
    for i in range(N):
        mods = []
        if use_ecg:
            mods.append({'name': 'ECG', 'risk_prob': ecg_risk_probs[idx_ecg[i]],
                         'confidence': ecg_confidences[idx_ecg[i]], 'data_quality': 1.0})
        if use_bp:
            mods.append({'name': 'BP', 'risk_prob': bp_probs[idx_bp[i]],
                         'confidence': bp_conf[idx_bp[i]], 'data_quality': 1.0})
        if use_glucose:
            mods.append({'name': 'Glucose', 'risk_prob': glucose_probs[idx_glucose[i]],
                         'confidence': glucose_conf[idx_glucose[i]], 'data_quality': 1.0})
        result = fuse_modalities(mods)
        fused_scores.append(result['fused_risk_score'])

    fused_scores = np.array(fused_scores)
    preds = (fused_scores >= 0.5).astype(int)

    acc = accuracy_score(combined_truth, preds)
    f1  = f1_score(combined_truth, preds)
    try:
        auc = roc_auc_score(combined_truth, fused_scores)
    except:
        auc = float('nan')

    print(f"{name:25s} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    return {'config': name, 'accuracy': acc, 'f1': f1, 'auc': auc}

print("="*60)
print("4-CONFIGURATION FUSION COMPARISON")
print("="*60)

results = []
results.append(evaluate_configuration("ECG alone",           True,  False, False))
results.append(evaluate_configuration("ECG + BP",             True,  True,  False))
results.append(evaluate_configuration("ECG + Glucose",        True,  False, True))
results.append(evaluate_configuration("ECG + BP + Glucose",   True,  True,  True))

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

os.makedirs("D:\\ReliableMultimodalAI\\reports", exist_ok=True)
df_results.to_csv("D:\\ReliableMultimodalAI\\reports\\fusion_comparison.csv", index=False)
print("\nSaved to reports/fusion_comparison.csv")