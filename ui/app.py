import streamlit as st
import numpy as np
import pandas as pd
import torch
import sys, os
import datetime
import uuid
import io
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src", "fusion"))
sys.path.append(os.path.join(BASE_DIR, "src", "agent"))

from multimodal_fusion import ecg_class_map, bp_scaler, glucose_scaler, fuse_modalities, ecg_models, device
from coordinator_agent import CoordinatorAgent

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

st.set_page_config(page_title="CardioMetabolic AI", layout="wide", page_icon="🫀")

# ───────────────────────── Global style ─────────────────────────
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #0b1526 0%, #0f1f38 100%); }
section[data-testid="stSidebar"] { background: #0a1222; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp p, .stApp li, .stApp label,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stTable"], [data-testid="stTable"] *,
[data-testid="stText"], [data-testid="stText"] *,
.stApp pre, .stApp pre *, .stApp code,
.stApp .stRadio label, .stApp .stCheckbox label, .stApp .stSelectbox label,
section[data-testid="stSidebar"] * { color: #eaf6ff !important; }
.stApp pre { background: rgba(255,255,255,0.03) !important; border-radius: 10px; padding: 12px; }
.hero-banner {
    background: linear-gradient(120deg, #0d2140 0%, #123a5e 55%, #0f2e52 100%);
    border: 1px solid rgba(0,229,255,0.25); border-radius: 18px;
    padding: 22px 28px; margin-bottom: 18px; box-shadow: 0 0 40px rgba(0,180,255,0.08);
}
.hero-title { font-size: 1.9rem; font-weight: 800; color: #eaf6ff; margin-bottom: 2px; }
.hero-sub { font-size: 0.95rem; color: #7fd8ff; font-weight: 500; }
.status-pill {
    display:inline-block; padding:4px 12px; border-radius:999px;
    background: rgba(40,220,140,0.15); color:#38e08a; font-weight:600; font-size:0.8rem;
    border: 1px solid rgba(40,220,140,0.35);
}
.card {
    background: rgba(255,255,255,0.035); border: 1px solid rgba(120,180,255,0.18);
    border-radius: 14px; padding: 16px 20px; margin-bottom: 12px;
}
.metric-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(120,180,255,0.2);
    border-radius: 12px; padding: 14px 18px; margin-bottom: 8px;
}
.risk-pill { display:inline-block; padding:4px 14px; border-radius:999px; font-weight:700; font-size:0.85rem; letter-spacing: 0.4px; }
.pill-low  { background: rgba(56,224,138,0.15); color:#38e08a; border:1px solid rgba(56,224,138,0.4); }
.pill-mod  { background: rgba(255,196,0,0.15); color:#ffc400; border:1px solid rgba(255,196,0,0.4); }
.pill-high { background: rgba(255,82,82,0.15); color:#ff5252; border:1px solid rgba(255,82,82,0.4); }
.patient-chip {
    display:inline-block; background: rgba(120,180,255,0.1); color:#bcdfff;
    padding: 3px 12px; border-radius: 8px; margin-right:8px; font-size:0.85rem;
    border: 1px solid rgba(120,180,255,0.25);
}
.feature-note { font-size: 0.78rem; color: #9fd8ff; margin: -6px 0 10px 2px; }
.agent-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(120,180,255,0.25);
    border-radius: 14px; padding: 16px; text-align:center; min-height: 170px;
}
.agent-icon { font-size: 2rem; }
.agent-name { font-weight:700; font-size:1rem; margin-top:4px; }
.agent-desc { font-size:0.8rem; color:#9fb4cc; margin-top:4px; }
.agent-status { margin-top:8px; font-size:0.78rem; font-weight:600; }
.sensor-card {
    background: rgba(255,255,255,0.04); border-radius: 14px; padding: 16px; text-align:center;
}
.badge-avail   { color:#38e08a; border:1px solid rgba(56,224,138,0.4); border-radius:999px; padding:3px 12px; font-size:0.8rem; }
.badge-unavail { color:#ff5252; border:1px solid rgba(255,82,82,0.4); border-radius:999px; padding:3px 12px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

now = datetime.datetime.now().strftime("%b %d, %Y • %H:%M")

st.markdown(f"""
<div class="hero-banner">
  <div class="hero-title">🫀 CardioMetabolic AI</div>
  <div class="hero-sub">Multimodal Cardiometabolic Risk Assessment · ECG + Blood Pressure + Glucose</div>
  <div style="margin-top:8px;">
    <span class="status-pill">● System Ready</span>
    &nbsp;&nbsp;<span style="color:#7fa8c9; font-size:0.85rem;">{now}</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.caption("Research Prototype — Not for Clinical Use. Scores shown are model estimates, not clinical diagnoses.")

DATA_DIR = os.path.join(BASE_DIR, "data") + os.sep
EXPECTED_LEADS = 12
EXPECTED_SAMPLES = 1000
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

IMG_SIZE = 900
GRID_COLS = 3
GRID_ROWS = 4
CELL_W = IMG_SIZE // GRID_COLS
CELL_H = IMG_SIZE // GRID_ROWS

try:
    LABEL_FONT = ImageFont.truetype("arial.ttf", 24)
except Exception:
    LABEL_FONT = ImageFont.load_default()

# ───────────────────────── Glossaries (real meanings, not clinical claims) ─────────────────────────

ECG_CLASS_NAMES = {
    "NORM": "Normal ECG pattern",
    "MI": "Myocardial Infarction pattern (heart-attack-related ECG changes)",
    "STTC": "ST/T-wave Changes (changes in the heart's electrical recovery pattern)",
    "CD": "Conduction Disturbance (abnormal electrical signal conduction)",
    "HYP": "Hypertrophy pattern (thickening of heart muscle)",
}

# Standard UCI Heart Disease (Cleveland) feature encoding — matches raw uci_heart.csv columns
# used as-is (no remapping) in train_bp.py, so this is the exact encoding the model was trained on.
BP_GLOSSARY = {
    "age": "Age (years)",
    "sex": "Sex (0 = Female, 1 = Male)",
    "cp": "Chest Pain Type (0=Typical angina, 1=Atypical angina, 2=Non-anginal pain, 3=Asymptomatic)",
    "trestbps": "Resting Blood Pressure (mmHg, measured at rest)",
    "chol": "Cholesterol (serum cholesterol, mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl (0=No, 1=Yes)",
    "restecg": "Resting ECG (0=Normal, 1=ST-T abnormality, 2=Probable/definite left ventricular hypertrophy)",
    "thalach": "Max Heart Rate Achieved (highest heart rate reached during exercise)",
    "exang": "Exercise-Induced Angina (0=No, 1=Yes — chest discomfort triggered by activity)",
    "oldpeak": "ST Depression (ECG ST-segment drop during exercise vs. rest; higher value, not a diagnosis)",
    "slope": "Slope of Peak Exercise ST Segment (0=Upsloping, 1=Flat, 2=Downsloping)",
    "ca": "Major Vessels (0-3; number of major coronary vessels shown narrowed on imaging)",
    "thal": "Thalassemia Test Result (3=Normal, 6=Fixed defect, 7=Reversible defect)",
}


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def risk_pill_html(prob):
    if prob < 0.33:
        return '<span class="risk-pill pill-low">LOW</span>'
    elif prob < 0.66:
        return '<span class="risk-pill pill-mod">MODERATE</span>'
    else:
        return '<span class="risk-pill pill-high">HIGH</span>'


def risk_label(prob):
    if prob < 0.33:
        return "Low"
    elif prob < 0.66:
        return "Moderate"
    else:
        return "High"


# ───────────────────────── ECG helpers (model logic unchanged) ─────────────────────────

def render_ecg_grid_image(signal_12x1000):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=255)
    draw = ImageDraw.Draw(img)
    for lead_idx in range(12):
        row = lead_idx // GRID_COLS
        col = lead_idx % GRID_COLS
        x0 = col * CELL_W
        y0 = row * CELL_H
        x1 = x0 + CELL_W
        y1 = y0 + CELL_H
        for gx in range(x0, x1, 30):
            draw.line([(gx, y0), (gx, y1)], fill=235, width=1)
        for gy in range(y0, y1, 30):
            draw.line([(x0, gy), (x1, gy)], fill=235, width=1)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=200, width=2)

        lead_signal = signal_12x1000[lead_idx]
        sig_min, sig_max = lead_signal.min(), lead_signal.max()
        sig_range = sig_max - sig_min
        if sig_range < 1e-6:
            sig_range = 1.0
        n_points = len(lead_signal)
        margin_y = 16
        usable_h = CELL_H - 2 * margin_y
        points = []
        for i, val in enumerate(lead_signal):
            px = x0 + int((i / (n_points - 1)) * (CELL_W - 1))
            norm = (val - sig_min) / sig_range
            py = y1 - margin_y - int(norm * usable_h)
            points.append((px, py))
        draw.line(points, fill=0, width=2)
        draw.text((x0 + 8, y0 + 4), LEAD_NAMES[lead_idx], fill=0, font=LABEL_FONT)
    return np.array(img, dtype=np.uint8)


def show_12_lead_grid(signal):
    lead_cols = st.columns(3)
    for li in range(12):
        with lead_cols[li % 3]:
            st.caption(LEAD_NAMES[li])
            st.line_chart(signal[li], height=160)
    zoom_lead = st.selectbox("Zoom into one lead:", LEAD_NAMES, index=1)
    zoom_idx = LEAD_NAMES.index(zoom_lead)
    st.line_chart(signal[zoom_idx], height=320, use_container_width=True)


def compute_lead_attribution(ecg_signal, predicted_class_idx):
    x = torch.FloatTensor(ecg_signal).unsqueeze(0).to(device)
    x.requires_grad_(True)
    total_logit = 0.0
    for m in ecg_models:
        m.eval()
        out = m(x)
        total_logit = total_logit + out[0, predicted_class_idx]
    total_logit.backward()
    grad = x.grad.detach().cpu().numpy()[0]
    input_vals = x.detach().cpu().numpy()[0]
    importance = np.sum(np.abs(grad * input_vals), axis=1)
    total = importance.sum()
    if total < 1e-10:
        importance = np.ones(12) / 12.0
    else:
        importance = importance / total
    return importance


def render_highlighted_ecg_image(signal_12x1000, importance, predicted_class, top_k=3, threshold=0.12):
    base = render_ecg_grid_image(signal_12x1000)
    rgb = np.stack([base] * 3, axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)
    highlighted_leads = []
    if predicted_class != "NORM":
        ranked = np.argsort(importance)[::-1]
        for lead_idx in ranked[:top_k]:
            if importance[lead_idx] < threshold:
                continue
            highlighted_leads.append(lead_idx)
            row = lead_idx // GRID_COLS
            col = lead_idx % GRID_COLS
            x0 = col * CELL_W
            y0 = row * CELL_H
            x1 = x0 + CELL_W
            y1 = y0 + CELL_H
            draw.rectangle([x0 + 1, y0 + 1, x1 - 2, y1 - 2], outline=(220, 20, 20), width=8)
    return img, highlighted_leads


@st.cache_resource
def load_coordinator():
    return CoordinatorAgent()


coordinator = load_coordinator()

# ───────────────────────── Patient Information ─────────────────────────

if "patient_id" not in st.session_state:
    st.session_state["patient_id"] = "PT-" + str(uuid.uuid4())[:6].upper()

st.markdown('<div class="card"><b>🧾 Patient Information</b></div>', unsafe_allow_html=True)
p1, p2, p3, p4, p5 = st.columns([2, 1, 1, 1.4, 1.4])
with p1:
    patient_name = st.text_input("Patient Name", value=st.session_state.get("patient_name", ""), placeholder="e.g. John Doe")
with p2:
    patient_age_display = st.number_input("Age", 0, 120, value=st.session_state.get("patient_age_display", 45))
with p3:
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"],
                                   index=["Male", "Female", "Other"].index(st.session_state.get("patient_gender", "Male")))
with p4:
    patient_phone = st.text_input("Phone Number", value=st.session_state.get("patient_phone", ""), placeholder="e.g. +91 98765 43210")
with p5:
    st.text_input("Patient ID", value=st.session_state["patient_id"], disabled=True)

st.session_state["patient_name"] = patient_name
st.session_state["patient_age_display"] = patient_age_display
st.session_state["patient_gender"] = patient_gender
st.session_state["patient_phone"] = patient_phone

st.divider()

# ───────────────────────── Sidebar ─────────────────────────
st.sidebar.header("Modalities")
use_ecg = st.sidebar.checkbox("ECG", value=True)
use_bp = st.sidebar.checkbox("Blood Pressure", value=True)
use_glucose = st.sidebar.checkbox("Glucose / Diabetes Risk", value=True)

st.sidebar.divider()
st.sidebar.header("Simulate Sensor Failure")
st.sidebar.caption("Sensor Failure: when one input becomes unavailable to the system.")
sim_drop_ecg = st.sidebar.checkbox("ECG Failure (ECG signal unavailable)", value=False)
sim_drop_bp = st.sidebar.checkbox("BP/Cardiac Failure (clinical info unavailable)", value=False)
sim_drop_glucose = st.sidebar.checkbox("Glucose Failure (diabetes-related info unavailable)", value=False)

bp_features = None
glucose_raw = None
ecg_signal = None
patient_bmi = None

tab_ecg, tab_bp, tab_glucose = st.tabs(["🫀 ECG", "🩺 Blood Pressure", "🩸 Glucose / Diabetes Risk"])

# ───────────────────────── ECG Tab ─────────────────────────
with tab_ecg:
    if use_ecg:
        ecg_mode = st.radio("ECG input source", ["Use a sample from test set", "Upload my own 12-lead ECG (CSV)"], index=0)

        if ecg_mode == "Use a sample from test set":
            X_ecg_test = np.load(DATA_DIR + "splits/ecg5_X_demo.npy")
            y_ecg_test = np.load(DATA_DIR + "splits/ecg5_y_demo.npy")
            inv_class_map = {v: k for k, v in ecg_class_map.items()}

            sample_idx = st.slider("ECG Sample Index", 0, len(X_ecg_test) - 1, 0)
            true_label = inv_class_map[y_ecg_test[sample_idx]]
            st.caption(f"Ground Truth (actual correct label): {true_label} — {ECG_CLASS_NAMES.get(true_label, '')}")

            ecg_signal = X_ecg_test[sample_idx]
            show_12_lead_grid(ecg_signal)
        else:
            st.caption(f"CSV with {EXPECTED_LEADS} leads × {EXPECTED_SAMPLES} samples (or transposed), no header row.")
            uploaded_file = st.file_uploader("Upload 12-lead ECG CSV", type=["csv"])
            if uploaded_file is not None:
                try:
                    raw_df = pd.read_csv(uploaded_file, header=None)
                    raw_arr = raw_df.values.astype(np.float32)
                    if raw_arr.shape == (EXPECTED_LEADS, EXPECTED_SAMPLES):
                        ecg_raw = raw_arr
                    elif raw_arr.shape == (EXPECTED_SAMPLES, EXPECTED_LEADS):
                        ecg_raw = raw_arr.T
                    else:
                        ecg_raw = None
                        st.error(f"Unexpected shape {raw_arr.shape}.")
                    if ecg_raw is not None:
                        mean = ecg_raw.mean(axis=1, keepdims=True)
                        std = ecg_raw.std(axis=1, keepdims=True)
                        std[std == 0] = 1.0
                        ecg_signal = (ecg_raw - mean) / std
                        st.success(f"ECG loaded. Shape: {ecg_signal.shape}")
                        show_12_lead_grid(ecg_signal)
                except Exception as e:
                    st.error(f"Could not read this file: {e}")
            else:
                st.info("Upload a CSV, or switch back to the sample option.")
        if sim_drop_ecg:
            st.warning("ECG failure simulation is ON — ECG will be excluded from this assessment.")
    else:
        st.info("ECG not included in this assessment.")

# ───────────────────────── Blood Pressure Tab ─────────────────────────
with tab_bp:
    if use_bp:
        input_widgets = {}
        c1, c2 = st.columns(2)
        bp_order_c1 = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"]
        bp_order_c2 = ["thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        with c1:
            input_widgets["age"] = st.number_input("Age", 20, 90, 55)
            st.markdown(f"<div class='feature-note'>{BP_GLOSSARY['age']}</div>", unsafe_allow_html=True)
            input_widgets["sex"] = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            st.markdown(f"<div class='feature-note'>{BP_GLOSSARY['sex']}</div>", unsafe_allow_html=True)
            input_widgets["cp"] = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
            st.markdown(f"<div class='feature-note'>{BP_GLOSSARY['cp']}</div>", unsafe_allow_html=True)
            input_widgets["trestbps"] = st.number_input("Resting BP (mmHg)", 80, 220, 130)
            st.markdown(f"<div class='feature-note'>{BP_GLOSSARY['trestbps']}</div>", unsafe_allow_html=True)
            input_widgets["chol"] = st.number_input("Cholesterol (mg/dl)", 100, 600, 230)
            st.markdown(f"<div class='feature-note'>{BP_GLOSSARY['chol']}</div>", unsafe_allow_html=True)
            input_widgets["fbs"] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
            st.markdown(f"<div class='feature-note'>{BP_GLOSSARY['fbs']}</div>", unsafe_allow_html=True)
            input_widgets["restecg"] = st.slider("Resting ECG (0-2)", 0, 2, 1)
            st.markdown(f"<div class='feature-note'>{BP_GLOSSARY['restecg']}</div>", unsafe_allow_html=True)
        with c2:
            input_widgets["thalach"] = st.number_input("Max Heart Rate", 60, 220, 150)
            st.markdown(f"<div class='feature-note'>{BP_GLOSSARY['thalach']}</div>", unsafe_allow_html=True)
            input_widgets["exang"] = st.selectbox("Exercise Induced Angina", [0, 1])
            st.markdown(f"<div class='feature-note'>{BP_GLOSSARY['exang']}</div>", unsafe_allow_html=True)
            input_widgets["oldpeak"] = st.number_input("ST Depression", 0.0, 6.0, 1.0)
            st.markdown(f"<div class='feature-note'>{BP_GLOSSARY['oldpeak']}</div>", unsafe_allow_html=True)
            input_widgets["slope"] = st.slider("Slope (0-2)", 0, 2, 1)
            st.markdown(f"<div class='feature-note'>{BP_GLOSSARY['slope']}</div>", unsafe_allow_html=True)
            input_widgets["ca"] = st.slider("Major Vessels (0-3)", 0, 3, 0)
            st.markdown(f"<div class='feature-note'>{BP_GLOSSARY['ca']}</div>", unsafe_allow_html=True)
            input_widgets["thal"] = st.slider("Thal (3,6,7)", 3, 7, 3)
            st.markdown(f"<div class='feature-note'>{BP_GLOSSARY['thal']}</div>", unsafe_allow_html=True)

        bp_features = np.array([
            input_widgets["age"], input_widgets["sex"], input_widgets["cp"], input_widgets["trestbps"],
            input_widgets["chol"], input_widgets["fbs"], input_widgets["restecg"], input_widgets["thalach"],
            input_widgets["exang"], input_widgets["oldpeak"], input_widgets["slope"], input_widgets["ca"],
            input_widgets["thal"],
        ])
        if sim_drop_bp:
            st.warning("BP failure simulation is ON — BP will be excluded from this assessment.")
    else:
        st.info("BP not included in this assessment.")

# ───────────────────────── Glucose Tab ─────────────────────────
with tab_glucose:
    if use_glucose:
        g1, g2 = st.columns(2)
        with g1:
            high_bp_flag = st.selectbox("High Blood Pressure History", [0, 1])
            high_chol = st.selectbox("High Cholesterol", [0, 1])
            bmi = st.number_input("BMI", 10.0, 60.0, 27.0)
            st.markdown(
                f"<div class='feature-note'>BMI (Body Mass Index — based on height and weight). "
                f"Value {bmi:.1f} falls in the <b>{bmi_category(bmi)}</b> range. "
                f"BMI alone does not determine diabetes risk.</div>", unsafe_allow_html=True)
            patient_bmi = bmi
            smoker = st.selectbox("Smoker", [0, 1])
        with g2:
            phys_activity = st.selectbox("Physically Active", [0, 1])
            gen_health = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
            diff_walk = st.selectbox("Difficulty Walking", [0, 1])
            age_glucose = st.slider("Age Category (1-13)", 1, 13, 7, help="1=18-24 ... 13=80+.")

        glucose_raw = np.zeros(21)
        glucose_raw[0] = high_bp_flag
        glucose_raw[1] = high_chol
        glucose_raw[2] = 1
        glucose_raw[3] = bmi
        glucose_raw[4] = smoker
        glucose_raw[8] = phys_activity
        glucose_raw[13] = gen_health
        glucose_raw[16] = diff_walk
        glucose_raw[18] = age_glucose

        if sim_drop_glucose:
            st.warning("Glucose failure simulation is ON — Glucose will be excluded from this assessment.")
    else:
        st.info("Glucose not included in this assessment.")

st.divider()

# ───────────────────────── Run Multi-Agent Assessment ─────────────────────────
if st.button("🔍 Run Assessment", type="primary", use_container_width=True):
    ecg_input = ecg_signal if (use_ecg and not sim_drop_ecg) else None
    bp_input = bp_features if (use_bp and not sim_drop_bp) else None
    glucose_input = glucose_raw if (use_glucose and not sim_drop_glucose) else None

    if ecg_input is None and bp_input is None and glucose_input is None:
        st.error("Please select at least one available modality.")
    else:
        # Matches the real CoordinatorAgent.run() signature — no invented parameters.
        fusion_result, report, modality_results = coordinator.run(
            ecg_signal=ecg_input,
            bp_features=bp_input,
            glucose_features=glucose_input,
        )

        if fusion_result is None:
            st.error(report)
        else:
            by_name = {m['name']: m for m in modality_results}
            fused_score = fusion_result['fused_risk_score']
            uncertainty = fusion_result['uncertainty_mass']
            conflict = fusion_result['conflict']
            is_reliable = fusion_result['is_reliable']

            # ═══════════════ 1-2. PATIENT SUMMARY + OVERALL RISK ═══════════════
            assess_date = datetime.datetime.now().strftime("%b %d, %Y")
            overall_label = risk_label(fused_score)
            name_display = patient_name if patient_name else "Not provided"

            st.markdown(f"""
<div class="hero-banner" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
  <div>
    <div style="font-size:1.1rem; font-weight:700; color:#eaf6ff;">👤 {name_display}</div>
    <div style="margin-top:6px;">
      <span class="patient-chip">ID: {st.session_state['patient_id']}</span>
      <span class="patient-chip">Age: {patient_age_display}</span>
      <span class="patient-chip">Gender: {patient_gender}</span>
      <span class="patient-chip">Date: {assess_date}</span>
    </div>
  </div>
  <div style="text-align:right;">
    <div style="color:#9fb4cc; font-size:0.85rem;">Overall Multimodal Risk Assessment</div>
    <div style="font-size:1.5rem; font-weight:800; color:#eaf6ff;">{overall_label} Risk</div>
    {risk_pill_html(fused_score)}
    <div style="margin-top:4px; color:#7fd8ff; font-size:0.9rem;">Fused Risk Score: {fused_score*100:.1f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

            if not is_reliable:
                st.error("⚠ CLINICAL REVIEW RECOMMENDED — the available modalities disagree enough "
                         "(conflict ≥ 30%) that the fused result should be checked by a clinician.")

            # ═══════════════ 3. ECG FINDINGS ═══════════════
            ecg_result = by_name.get('ECG')
            if ecg_result is not None and ecg_input is not None:
                st.markdown("## 🫀 ECG Findings")
                pred_class = ecg_result.get('predicted_class', 'NORM')
                ecg_risk = ecg_result.get('risk_prob', 0.0)

                st.markdown(
                    f"**Predicted Class (class predicted by the AI model):** "
                    f"{pred_class} — {ECG_CLASS_NAMES.get(pred_class, pred_class)}",
                )
                st.markdown(
                    f"ECG Risk Category: {risk_pill_html(ecg_risk)}  "
                    f"<span class='feature-note'>(based on the model's probability that the ECG is NOT Normal — "
                    f"1 − P(Normal) = {ecg_risk*100:.1f}%. This is separate from the predicted class above.)</span>",
                    unsafe_allow_html=True)

                predicted_class_idx = ecg_class_map[pred_class]
                importance = compute_lead_attribution(ecg_input, predicted_class_idx)
                highlighted_img, highlighted_leads = render_highlighted_ecg_image(ecg_input, importance, pred_class)
                st.image(highlighted_img, caption="12-lead ECG", use_container_width=True)

                if pred_class == "NORM":
                    st.success("No abnormality detected.")
                elif highlighted_leads:
                    lead_txt = ", ".join(f"{LEAD_NAMES[i]} ({importance[i]*100:.0f}%)" for i in highlighted_leads)
                    st.caption(
                        f"Model-highlighted leads (the leads the AI model relied on most for this prediction, "
                        f"based on gradient-based attribution — not confirmed abnormal leads): {lead_txt}")

            # ═══════════════ 4. CARDIAC RISK ASSESSMENT ═══════════════
            bp_result = by_name.get('BP')
            if bp_result is not None and bp_input is not None:
                st.markdown("## 🩺 Cardiac Risk Assessment")
                bp_risk = bp_result.get('risk_prob', 0.0)
                st.markdown(f"**Cardiac Risk Score:** {bp_risk*100:.1f}% {risk_pill_html(bp_risk)}", unsafe_allow_html=True)
                st.caption(
                    f"The model estimates a relatively {risk_label(bp_risk).lower()} cardiac risk based on the "
                    f"provided clinical features (model probability of heart disease = {bp_risk*100:.1f}%). "
                    f"This is a model estimate, not a diagnosis.")

            # ═══════════════ 5. DIABETES RISK ASSESSMENT ═══════════════
            glucose_result = by_name.get('Glucose')
            if glucose_result is not None and glucose_input is not None:
                st.markdown("## 🩸 Diabetes Risk Assessment")
                gl_risk = glucose_result.get('risk_prob', 0.0)
                st.markdown(f"**Diabetes Risk Score:** {gl_risk*100:.1f}% {risk_pill_html(gl_risk)}", unsafe_allow_html=True)
                st.caption(
                    "This model does not measure actual blood glucose concentration. It estimates diabetes risk "
                    "from health and lifestyle indicators (including BMI). The score is a model-estimated risk, "
                    "not a diabetes diagnosis.")

            # ═══════════════ 6. RELIABILITY & FUSION ANALYSIS ═══════════════
            st.markdown("## 🔗 Reliability & Fusion Analysis")
            reliab_txt = ", ".join(f"{m['name']}: {m['reliability']*100:.0f}%" for m in modality_results)

            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.markdown(f"<div class='metric-card'><b>Fused Risk Score ⓘ</b>"
                            f"<div class='feature-note'>Combined risk estimate from all available modalities, "
                            f"using Dempster-Shafer evidence fusion weighted by each modality's reliability.</div>"
                            f"<span style='font-size:1.5rem'>{fused_score*100:.1f}%</span> "
                            f"{risk_pill_html(fused_score)}</div>", unsafe_allow_html=True)
            with fc2:
                st.markdown(f"<div class='metric-card'><b>Uncertainty ⓘ</b>"
                            f"<div class='feature-note'>How unsure the system is about this assessment. "
                            f"Higher value = more uncertain. Driven by each modality's confidence × data quality.</div>"
                            f"<span style='font-size:1.5rem'>{uncertainty*100:.1f}%</span></div>",
                            unsafe_allow_html=True)
            with fc3:
                st.markdown(f"<div class='metric-card'><b>Conflict ⓘ</b>"
                            f"<div class='feature-note'>How much the modalities disagree with each other "
                            f"(Dempster's conflict term). Below 30% is considered reliable agreement.</div>"
                            f"<span style='font-size:1.5rem'>{conflict*100:.1f}%</span></div>",
                            unsafe_allow_html=True)

            st.caption(f"Per-modality reliability used in fusion (confidence × data quality): {reliab_txt}")

            if conflict > 0.3:
                st.warning("Modalities disagree significantly — interpret the fused result with caution.")
            elif is_reliable:
                st.success("Modalities agree — consistent evidence across available inputs.")

            # ═══════════════ 7. SINGLE vs MULTIMODAL ACCURACY COMPARISON ═══════════════
            st.markdown("## 📊 Single vs. Multimodal Accuracy Comparison")
            st.caption("Real benchmark results — Accuracy, F1-score and AUC, measured on the same held-out "
                       "synthetic multimodal test set for every configuration (Precision/Recall were not "
                       "separately evaluated in this benchmark).")

            # Locked benchmark results from run_comparison_full.py (500 synthetic patients, seed 42)
            benchmark_results = [
                {"Configuration": "ECG alone",          "Accuracy": 82.2, "F1": 0.837, "AUC": 0.956},
                {"Configuration": "BP alone",            "Accuracy": 86.0, "F1": 0.863, "AUC": 0.953},
                {"Configuration": "Glucose alone",       "Accuracy": 79.6, "F1": 0.798, "AUC": 0.874},
                {"Configuration": "ECG + BP",             "Accuracy": 89.0, "F1": 0.892, "AUC": 0.973},
                {"Configuration": "ECG + Glucose",        "Accuracy": 89.0, "F1": 0.894, "AUC": 0.977},
                {"Configuration": "BP + Glucose",         "Accuracy": 90.2, "F1": 0.903, "AUC": 0.970},
                {"Configuration": "ECG + BP + Glucose",   "Accuracy": 91.4, "F1": 0.915, "AUC": 0.981},
            ]
            bench_df = pd.DataFrame(benchmark_results)

            def highlight_all3(row):
                if row["Configuration"] == "ECG + BP + Glucose":
                    return ['background-color: rgba(56,224,138,0.18)'] * len(row)
                return [''] * len(row)

            st.dataframe(bench_df.style.apply(highlight_all3, axis=1).format({"Accuracy": "{:.1f}%", "F1": "{:.3f}", "AUC": "{:.3f}"}),
                         use_container_width=True, hide_index=True)
            st.bar_chart(bench_df.set_index("Configuration")["Accuracy"])

            single_best = max(benchmark_results[0:3], key=lambda r: r["Accuracy"])
            all3 = benchmark_results[-1]
            improvement = all3["Accuracy"] - single_best["Accuracy"]
            st.markdown(
                f"<div class='metric-card'><b>Accuracy Improvement</b><br>"
                f"All-three-modality accuracy ({all3['Accuracy']:.1f}%) vs. best single modality "
                f"'{single_best['Configuration']}' ({single_best['Accuracy']:.1f}%) = "
                f"<span style='font-size:1.3rem; color:#38e08a;'>+{improvement:.1f} percentage points</span></div>",
                unsafe_allow_html=True)
            st.caption("Combining all three modalities improved fused-risk accuracy over the best single modality "
                      "on this benchmark.")

            # ═══════════════ 8. AGENTIC AI REASONING ═══════════════
            st.markdown("## 🤖 Agentic AI Reasoning")
            st.caption("Each specialized agent analyzes its modality independently; the Fusion Agent then "
                      "combines the available, reliability-weighted results.")

            agent_defs = [
                ("🫀", "ECG Agent", "Analyzes the 12-lead ECG using the TCN ensemble model.", 'ECG'),
                ("🩺", "Cardiac Agent", "Evaluates cardiac clinical risk indicators (BP model).", 'BP'),
                ("🩸", "Diabetes Agent", "Evaluates diabetes-related risk indicators (Glucose model).", 'Glucose'),
                ("🔗", "Fusion Agent", "Combines available, reliability-weighted results into one assessment.", None),
            ]
            acols = st.columns(4)
            for (icon, aname, desc, key) in agent_defs:
                with acols[agent_defs.index((icon, aname, desc, key))]:
                    if key is None:
                        status = "Fused result produced"
                        color = "#38e08a"
                    elif key in by_name:
                        status = "Analyzed"
                        color = "#38e08a"
                    else:
                        status = "Not included"
                        color = "#9fb4cc"
                    st.markdown(f"""
<div class="agent-card">
  <div class="agent-icon">{icon}</div>
  <div class="agent-name">{aname}</div>
  <div class="agent-desc">{desc}</div>
  <div class="agent-status" style="color:{color};">● {status}</div>
</div>
""", unsafe_allow_html=True)

            st.markdown(
                "<div style='text-align:center; margin:10px 0; color:#7fd8ff; font-size:1.1rem;'>"
                "ECG Agent + Cardiac Agent + Diabetes Agent  →  Fusion Agent  →  Final Assessment</div>",
                unsafe_allow_html=True)

            with st.expander("Detailed agent output (Data Quality + Explanation Agent text)"):
                st.text(report)

            # ═══════════════ 9. SENSOR AVAILABILITY / FAILURE ANALYSIS ═══════════════
            st.markdown("## 📡 Sensor Availability")
            scols = st.columns(3)
            sensor_defs = [
                ("❤️", "ECG", use_ecg and not sim_drop_ecg, "ECG signal unavailable"),
                ("🩺", "BP / Cardiac", use_bp and not sim_drop_bp, "Cardiac clinical information unavailable"),
                ("🩸", "Glucose / Diabetes", use_glucose and not sim_drop_glucose, "Diabetes-related health information unavailable"),
            ]
            for col, (icon, label, available, unavail_text) in zip(scols, sensor_defs):
                with col:
                    badge = '<span class="badge-avail">Available</span>' if available else '<span class="badge-unavail">Unavailable</span>'
                    note = "" if available else f"<div class='feature-note'>{unavail_text}</div>"
                    st.markdown(f"""
<div class="sensor-card">
  <div style="font-size:1.8rem;">{icon}</div>
  <div style="font-weight:700;">{label}</div>
  {badge}
  {note}
</div>
""", unsafe_allow_html=True)
            if not all(a for _, _, a, _ in sensor_defs):
                st.caption("The system assesses the remaining available modalities — missing inputs are not "
                          "treated as a normal result, they are simply excluded from fusion (shown above as "
                          "'Not included').")

            # ═══════════════ 10. PDF REPORT ═══════════════
            st.markdown("## 📄 Report")
            if PDF_AVAILABLE:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 16)
                pdf.cell(0, 10, "CardioMetabolic AI - Assessment Report", ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
                pdf.ln(4)

                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "Patient Information", ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 7, f"Name: {name_display}   ID: {st.session_state['patient_id']}", ln=True)
                pdf.cell(0, 7, f"Age: {patient_age_display}   Gender: {patient_gender}   Phone: {patient_phone or 'Not provided'}", ln=True)
                pdf.ln(3)

                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "Overall Assessment", ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 7, f"Overall Risk: {overall_label}   Fused Risk Score: {fused_score*100:.1f}%", ln=True)
                pdf.cell(0, 7, f"Uncertainty: {uncertainty*100:.1f}%   Conflict: {conflict*100:.1f}%", ln=True)
                pdf.ln(3)

                if ecg_result is not None and ecg_input is not None:
                    pdf.set_font("Helvetica", "B", 12)
                    pdf.cell(0, 8, "ECG Summary", ln=True)
                    pdf.set_font("Helvetica", "", 10)
                    pdf.multi_cell(0, 6, f"Predicted class: {ecg_result.get('predicted_class')} - "
                                        f"{ECG_CLASS_NAMES.get(ecg_result.get('predicted_class'), '')}. "
                                        f"ECG risk score: {ecg_result.get('risk_prob', 0.0)*100:.1f}%")
                    pdf.ln(2)

                if bp_result is not None and bp_input is not None:
                    pdf.set_font("Helvetica", "B", 12)
                    pdf.cell(0, 8, "Cardiac Risk Summary", ln=True)
                    pdf.set_font("Helvetica", "", 10)
                    pdf.multi_cell(0, 6, f"Cardiac risk score: {bp_result.get('risk_prob', 0.0)*100:.1f}% (model estimate, not a diagnosis)")
                    pdf.ln(2)

                if glucose_result is not None and glucose_input is not None:
                    pdf.set_font("Helvetica", "B", 12)
                    pdf.cell(0, 8, "Diabetes Risk Summary", ln=True)
                    pdf.set_font("Helvetica", "", 10)
                    pdf.multi_cell(0, 6, f"Diabetes risk score: {glucose_result.get('risk_prob', 0.0)*100:.1f}% (estimated risk, not a measured glucose level or diagnosis)")
                    pdf.ln(2)

                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "Agentic AI Summary", ln=True)
                pdf.set_font("Helvetica", "", 9)
                safe_report = report.encode("latin-1", "replace").decode("latin-1")
                pdf.multi_cell(0, 5, safe_report[:3000])

                pdf.set_font("Helvetica", "I", 9)
                pdf.ln(4)
                pdf.multi_cell(0, 6, "Final clinician-oriented note: This report presents AI model estimates "
                                     "from a research prototype. It is not a clinical diagnosis and does not "
                                     "replace professional medical assessment.")

                pdf_bytes = bytes(pdf.output(dest="S"))
                st.download_button("⬇ Download PDF Report", data=pdf_bytes,
                                   file_name=f"cardiometabolic_report_{st.session_state['patient_id']}.pdf",
                                   mime="application/pdf", use_container_width=True)
            else:
                st.info("PDF export needs the 'fpdf2' package. Add `fpdf2` to requirements.txt and "
                       "`pip install fpdf2 --break-system-packages` locally, then rerun.")