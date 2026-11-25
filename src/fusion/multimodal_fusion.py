import numpy as np
import torch
import torch.nn as nn
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────
# PART 1 — Load all trained models
# ──────────────────────────────────────────────────────────

MODELS_DIR = "D:\\ReliableMultimodalAI\\models_saved\\"
DATA_DIR   = "D:\\ReliableMultimodalAI\\data\\"

print("Loading trained models...")

# --- Glucose (LightGBM) ---
glucose_model  = joblib.load(MODELS_DIR + "glucose_model.pkl")
glucose_scaler = joblib.load(MODELS_DIR + "glucose_scaler.pkl")

# --- BP (LightGBM) ---
bp_model  = joblib.load(MODELS_DIR + "bp_model.pkl")
bp_scaler = joblib.load(MODELS_DIR + "bp_scaler.pkl") if __import__('os').path.exists(MODELS_DIR + "bp_scaler.pkl") else None

# --- ECG (TCN ensemble) ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid())
    def forward(self, x):
        b, c, _ = x.size()
        s = self.squeeze(x).view(b, c)
        e = self.excitation(s).view(b, c, 1)
        return x * e

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, drop=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.se = SEBlock(out_ch)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        res = self.residual(x)
        out = self.drop(self.relu(self.bn1(self.conv1(x))))
        out = self.drop(self.relu(self.bn2(self.conv2(out))))
        out = self.se(out)
        out = out[:, :, :res.shape[2]]
        return self.relu(out + res)

class ECGModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.tcn1 = TCNBlock(12, 64, 7, 1, drop=0.15)
        self.tcn2 = TCNBlock(64, 128, 7, 2, drop=0.2)
        self.tcn3 = TCNBlock(128, 256, 7, 4, drop=0.2)
        self.tcn4 = TCNBlock(256, 384, 7, 8, drop=0.2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.tcn4(self.tcn3(self.tcn2(self.tcn1(x))))
        avg, mx = self.gap(x).squeeze(-1), self.gmp(x).squeeze(-1)
        return self.classifier(torch.cat([avg, mx], dim=1))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ecg_class_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}

ecg_models = []
for seed in [42, 123, 777]:
    m = ECGModel(num_classes=5).to(device)
    m.load_state_dict(torch.load(MODELS_DIR + f"ecg_model_seed{seed}.pth", map_location=device))
    m.eval()
    ecg_models.append(m)

print("All models loaded successfully!\n")

# ──────────────────────────────────────────────────────────
# PART 2 — Per-modality prediction functions
# ──────────────────────────────────────────────────────────

def predict_glucose(features_raw):
    """features_raw: 1D array of raw (unscaled) glucose features (21 BRFSS columns)"""
    X = glucose_scaler.transform(features_raw.reshape(1, -1))
    prob = glucose_model.predict_proba(X)[0, 1]  # P(diabetic)
    confidence = max(prob, 1 - prob)
    return {'risk_prob': float(prob), 'confidence': float(confidence)}

def predict_bp(features_raw):
    """features_raw: 1D array of raw UCI heart features (13 columns)"""
    if bp_scaler is not None:
        X = bp_scaler.transform(features_raw.reshape(1, -1))
    else:
        X = features_raw.reshape(1, -1)
    prob = bp_model.predict_proba(X)[0, 1]  # P(heart disease)
    confidence = max(prob, 1 - prob)
    return {'risk_prob': float(prob), 'confidence': float(confidence)}

def predict_ecg(ecg_signal_12x1000):
    """ecg_signal_12x1000: numpy array shape (12, 1000), already bandpass-filtered + normalized"""
    x = torch.FloatTensor(ecg_signal_12x1000).unsqueeze(0).to(device)  # (1, 12, 1000)
    probs_sum = torch.zeros(1, 5).to(device)
    with torch.no_grad():
        for m in ecg_models:
            probs_sum += torch.softmax(m(x), dim=1)
    probs = (probs_sum / len(ecg_models)).cpu().numpy()[0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    # Convert to binary "abnormal risk" = 1 - P(NORM)
    risk_prob = float(1 - probs[ecg_class_map['NORM']])
    return {
        'risk_prob': risk_prob,
        'confidence': confidence,
        'predicted_class': list(ecg_class_map.keys())[pred_class],
        'class_probs': {k: float(probs[v]) for k, v in ecg_class_map.items()}
    }

# ──────────────────────────────────────────────────────────
# PART 3 — Dempster-Shafer Fusion
# ──────────────────────────────────────────────────────────

def compute_reliability(confidence, data_quality=1.0):
    return confidence * data_quality

def make_mass_function(risk_prob, reliability):
    m_high  = risk_prob * reliability
    m_low   = (1 - risk_prob) * reliability
    m_theta = 1 - reliability
    return {'high_risk': m_high, 'low_risk': m_low, 'uncertain': m_theta}

def combine_two(m1, m2):
    K = m1['high_risk'] * m2['low_risk'] + m1['low_risk'] * m2['high_risk']
    if K >= 0.99:
        return {'high_risk': 0.5, 'low_risk': 0.5, 'uncertain': 1.0, 'conflict': K}
    high = (m1['high_risk']*m2['high_risk'] + m1['high_risk']*m2['uncertain'] +
            m1['uncertain']*m2['high_risk']) / (1 - K)
    low = (m1['low_risk']*m2['low_risk'] + m1['low_risk']*m2['uncertain'] +
           m1['uncertain']*m2['low_risk']) / (1 - K)
    theta = (m1['uncertain']*m2['uncertain']) / (1 - K)
    return {'high_risk': high, 'low_risk': low, 'uncertain': theta, 'conflict': K}

def fuse_modalities(modality_results):
    """modality_results: list of {'name', 'risk_prob', 'confidence', 'data_quality'}"""
    if len(modality_results) == 0:
        return None

    mass_functions = []
    for mod in modality_results:
        reliability = compute_reliability(mod['confidence'], mod.get('data_quality', 1.0))
        m = make_mass_function(mod['risk_prob'], reliability)
        mass_functions.append(m)
        mod['reliability'] = reliability

    if len(mass_functions) == 1:
        combined = mass_functions[0]
        combined['conflict'] = 0.0
    else:
        combined = mass_functions[0]
        for m in mass_functions[1:]:
            combined = combine_two(combined, m)

    total_conflict = combined.get('conflict', 0.0)
    fused_risk = combined['high_risk'] / (combined['high_risk'] + combined['low_risk'] + 1e-8)

    return {
        'fused_risk_score': fused_risk,
        'uncertainty_mass': combined['uncertain'],
        'conflict': total_conflict,
        'is_reliable': total_conflict < 0.3,
        'modality_details': modality_results
    }

print("Fusion module ready.")