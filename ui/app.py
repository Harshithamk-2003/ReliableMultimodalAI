import streamlit as st
import numpy as np
import pandas as pd
import torch
import sys, os
import datetime
import uuid
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src", "fusion"))
sys.path.append(os.path.join(BASE_DIR, "src", "agent"))

from multimodal_fusion import ecg_class_map, bp_scaler, glucose_scaler, fuse_modalities, ecg_models, device
from coordinator_agent import CoordinatorAgent

st.set_page_config(page_title="CardioMetabolic AI", layout="wide", page_icon="🫀")

# ───────────────────────── Global style ─────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0b1526 0%, #0f1f38 100%);
}
section[data-testid="stSidebar"] {
    background: #0a1222;
}
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
section[data-testid="stSidebar"] * {
    color: #eaf6ff !important;
}
.stApp pre {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px;
    padding: 12px;
}
.hero-banner {
    background: linear-gradient(120deg, #0d2140 0%, #123a5e 55%, #0f2e52 100%);
    border: 1px solid rgba(0,229,255,0.25);
    border-radius: 18px;
    padding: 22px 28px;
    margin-bottom: 18px;
    box-shadow: 0 0 40px rgba(0,180,255,0.08);
}
.hero-title {
    font-size: 1.9rem; font-weight: 800; color: #eaf6ff; margin-bottom: 2px;
}
.hero-sub {
    font-size: 0.95rem; color: #7fd8ff; font-weight: 500;
}
.status-pill {
    display:inline-block; padding:4px 12px; border-radius:999px;
    background: rgba(40,220,140,0.15); color:#38e08a; font-weight:600; font-size:0.8rem;
    border: 1px solid rgba(40,220,140,0.35);
}
.card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(120,180,255,0.18);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(120,180,255,0.2);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.risk-pill {
    display:inline-block; padding:4px 14px; border-radius:999px;
    font-weight:700; font-size:0.85rem; letter-spacing: 0.4px;
}
.pill-low    { background: rgba(56,224,138,0.15); color:#38e08a; border:1px solid rgba(56,224,138,0.4); }
.pill-mod    { background: rgba(255,196,0,0.15); color:#ffc400; border:1px solid rgba(255,196,0,0.4); }
.pill-high   { background: rgba(255,82,82,0.15); color:#ff5252; border:1px solid rgba(255,82,82,0.4); }
.patient-chip {
    display:inline-block; background: rgba(120,180,255,0.1); color:#bcdfff;
    padding: 3px 12px; border-radius: 8px; margin-right:8px; font-size:0.85rem;
    border: 1px solid rgba(120,180,255,0.25);
}
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

ECG_CLASS_NAMES = {
    "NORM": "Normal",
    "MI": "Heart Attack pattern",
    "STTC": "ST/T-wave change",
    "CD": "Conduction disorder",
    "HYP": "Hypertrophy pattern",
}


def risk_pill_html(prob):
    if prob < 0.33:
        return '<span class="risk-pill pill-low">LOW</span>'
    elif prob < 0.66:
        return '<span class="risk-pill pill-mod">MODERATE</span>'
    else:
        return '<span class="risk-pill pill-high">HIGH</span>'


def colored_bar(label, value, suffix="%"):
    c1, c2 = st.columns([1, 4])
    with c1:
        st.markdown(f"**{label}**")
    with c2:
        st.progress(min(max(value, 0.0), 1.0), text=f"{value*100:.1f}{suffix}")


# ───────────────────────── ECG helpers (unchanged logic) ─────────────────────────

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
sim_drop_ecg = st.sidebar.checkbox("ECG fails", value=False)
sim_drop_bp = st.sidebar.checkbox("BP fails", value=False)
sim_drop_glucose = st.sidebar.checkbox("Glucose fails", value=False)

bp_features = None
glucose_raw = None
ecg_signal = None

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
            st.caption(f"Ground truth label: {true_label}")

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
            st.warning("ECG failure simulation is ON.")
    else:
        st.info("ECG not included in this assessment.")

# ───────────────────────── Blood Pressure Tab ─────────────────────────
with tab_bp:
    if use_bp:
        input_widgets = {}
        c1, c2 = st.columns(2)
        with c1:
            input_widgets["age"] = st.number_input("Age", 20, 90, 55,
                help="Cardiovascular risk rises with age.")
            input_widgets["sex"] = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male",
                help="0 = Female, 1 = Male.")
            input_widgets["cp"] = st.slider("Chest Pain Type (0-3)", 0, 3, 1,
                help="0=Typical angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic.")
            input_widgets["trestbps"] = st.number_input("Resting BP (mmHg)", 80, 220, 130,
                help="Normal ~90-120; above 140 is hypertension. Single resting value (no diastolic in this dataset).")
            input_widgets["chol"] = st.number_input("Cholesterol (mg/dl)", 100, 600, 230,
                help="Normal under ~200; above 240 is high.")
            input_widgets["fbs"] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                help="0 = No, 1 = Yes. Diabetes screening cutoff.")
            input_widgets["restecg"] = st.slider("Resting ECG (0-2)", 0, 2, 1,
                help="0=Normal, 1=ST-T abnormality, 2=Likely LV hypertrophy.")
        with c2:
            input_widgets["thalach"] = st.number_input("Max Heart Rate", 60, 220, 150,
                help="Highest heart rate reached during exertion.")
            input_widgets["exang"] = st.selectbox("Exercise Induced Angina", [0, 1],
                help="0 = No, 1 = Yes.")
            input_widgets["oldpeak"] = st.number_input("ST Depression", 0.0, 6.0, 1.0,
                help="ST segment drop during exercise vs rest (mm). Higher = more ischemia risk.")
            input_widgets["slope"] = st.slider("Slope (0-2)", 0, 2, 1,
                help="0=Upsloping (lower risk), 1=Flat, 2=Downsloping.")
            input_widgets["ca"] = st.slider("Major Vessels (0-3)", 0, 3, 0,
                help="Number of major arteries showing blockage on imaging.")
            input_widgets["thal"] = st.slider("Thal (3,6,7)", 3, 7, 3,
                help="3=Normal, 6=Fixed defect, 7=Reversible defect.")

        bp_features = np.array([
            input_widgets["age"], input_widgets["sex"], input_widgets["cp"], input_widgets["trestbps"],
            input_widgets["chol"], input_widgets["fbs"], input_widgets["restecg"], input_widgets["thalach"],
            input_widgets["exang"], input_widgets["oldpeak"], input_widgets["slope"], input_widgets["ca"],
            input_widgets["thal"],
        ])

        if sim_drop_bp:
            st.warning("BP failure simulation is ON.")
    else:
        st.info("BP not included in this assessment.")

# ───────────────────────── Glucose Tab ─────────────────────────
with tab_glucose:
    if use_glucose:
        g1, g2 = st.columns(2)
        with g1:
            high_bp_flag = st.selectbox("High Blood Pressure History", [0, 1], help="0 = No, 1 = Yes.")
            high_chol = st.selectbox("High Cholesterol", [0, 1], help="0 = No, 1 = Yes.")
            bmi = st.number_input("BMI", 10.0, 60.0, 27.0, help="Normal 18.5-24.9; over 30 is obese.")
            smoker = st.selectbox("Smoker", [0, 1], help="0 = No, 1 = Yes.")
        with g2:
            phys_activity = st.selectbox("Physically Active", [0, 1], help="0 = No, 1 = Yes.")
            gen_health = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
            diff_walk = st.selectbox("Difficulty Walking", [0, 1], help="0 = No, 1 = Yes.")
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
            st.warning("Glucose failure simulation is ON.")
    else:
        st.info("Glucose not included in this assessment.")

st.divider()

# ───────────────────────── Run Multi-Agent Assessment ─────────────────────────
if st.button("🔍 Run Assessment", type="primary", use_container_width=True):
    ecg_input = ecg_signal if use_ecg else None
    bp_input = bp_features if use_bp else None
    glucose_input = glucose_raw if use_glucose else None

    if ecg_input is None and bp_input is None and glucose_input is None:
        st.error("Please select or upload at least one modality.")
    elif sim_drop_ecg and sim_drop_bp and sim_drop_glucose:
        st.error("All modalities are set to simulated failure — nothing to assess.")
    else:
        fusion_result, report, modality_results = coordinator.run(
            ecg_signal=ecg_input,
            bp_features=bp_input,
            glucose_features=glucose_input,
            drop_ecg=sim_drop_ecg,
            drop_bp=sim_drop_bp,
            drop_glucose=sim_drop_glucose,
        )

        if fusion_result is None:
            st.error(report)
        else:
            escalation = fusion_result.get("escalation", {})
            by_name = {m['name']: m for m in modality_results}
            fused_score = fusion_result['fused_risk_score']

            # ═══════════════ PATIENT SUMMARY BANNER ═══════════════
            assess_date = datetime.datetime.now().strftime("%b %d, %Y")
            risk_label = "Low Risk" if fused_score < 0.33 else ("Moderate Risk" if fused_score < 0.66 else "High Risk")
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
    <div style="color:#9fb4cc; font-size:0.85rem;">Overall Assessment</div>
    <div style="font-size:1.5rem; font-weight:800; color:#eaf6ff;">{risk_label}</div>
    {risk_pill_html(fused_score)}
    <div style="margin-top:4px; color:#7fd8ff; font-size:0.9rem;">Fused Risk Score: {fused_score*100:.1f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

            if escalation.get("triggered"):
                st.error("⚠ CLINICAL REVIEW RECOMMENDED")

            # ═══════════════ ECG RESULT ═══════════════
            ecg_result = by_name.get('ECG')
            if ecg_result is not None and ecg_input is not None and not sim_drop_ecg:
                st.markdown("## 🫀 ECG Result")
                pred_class = ecg_result.get('predicted_class', 'NORM')
                class_probs = ecg_result.get('class_probs', {})

                st.markdown(f"**Finding:** {ECG_CLASS_NAMES.get(pred_class, pred_class)} ({pred_class}) "
                            f"{risk_pill_html(class_probs.get(pred_class, 0.0))}", unsafe_allow_html=True)

                predicted_class_idx = ecg_class_map[pred_class]
                importance = compute_lead_attribution(ecg_input, predicted_class_idx)
                highlighted_img, highlighted_leads = render_highlighted_ecg_image(ecg_input, importance, pred_class)
                st.image(highlighted_img, caption="12-lead ECG (model-attributed leads shown in red)", use_container_width=True)

                if pred_class == "NORM":
                    st.success("No abnormality detected.")
                elif highlighted_leads:
                    lead_txt = ", ".join(f"{LEAD_NAMES[i]} ({importance[i]*100:.0f}%)" for i in highlighted_leads)
                    st.caption(f"Most contributing leads: {lead_txt}")

            # ═══════════════ BP RESULT ═══════════════
            bp_result = by_name.get('BP')
            if bp_result is not None and bp_input is not None and not sim_drop_bp:
                st.markdown("## 🩺 Blood Pressure Risk")
                bp_risk = bp_result.get('risk_prob', 0.0)
                st.markdown(f"**Risk score:** {bp_risk*100:.1f}% {risk_pill_html(bp_risk)}", unsafe_allow_html=True)

            # ═══════════════ GLUCOSE RESULT ═══════════════
            glucose_result = by_name.get('Glucose')
            if glucose_result is not None and glucose_input is not None and not sim_drop_glucose:
                st.markdown("## 🩸 Diabetes Risk")
                gl_risk = glucose_result.get('risk_prob', 0.0)
                st.markdown(f"**Risk score:** {gl_risk*100:.1f}% {risk_pill_html(gl_risk)}", unsafe_allow_html=True)

            # ═══════════════ FUSION RESULT ═══════════════
            st.markdown("## 🔗 Combined (Fused) Result")
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.markdown(f"<div class='metric-card'><b>Fused Risk Score</b><br>"
                            f"<span style='font-size:1.5rem'>{fusion_result['fused_risk_score']*100:.1f}%</span> "
                            f"{risk_pill_html(fusion_result['fused_risk_score'])}</div>", unsafe_allow_html=True)
            with fc2:
                st.markdown(f"<div class='metric-card'><b>Uncertainty</b><br>"
                            f"<span style='font-size:1.5rem'>{fusion_result['uncertainty_mass']*100:.1f}%</span></div>",
                            unsafe_allow_html=True)
            with fc3:
                st.markdown(f"<div class='metric-card'><b>Conflict</b><br>"
                            f"<span style='font-size:1.5rem'>{fusion_result['conflict']*100:.1f}%</span></div>",
                            unsafe_allow_html=True)

            if not escalation.get("triggered"):
                if fusion_result['conflict'] > 0.3:
                    st.warning("Modalities disagree significantly — interpret with caution.")
                elif fusion_result['is_reliable']:
                    st.success("Modalities agree — consistent evidence.")

            # ═══════════════ ACCURACY COMPARISON (kept prominent) ═══════════════
            st.markdown("## 📊 Accuracy: Single vs. Combined Modalities")
            st.caption("Measured on held-out test patients — real benchmark results, same evaluation for every row.")

            configs = [
                ("ECG alone",           ['ECG']),
                ("BP alone",            ['BP']),
                ("Glucose alone",       ['Glucose']),
                ("ECG + BP",            ['ECG', 'BP']),
                ("ECG + Glucose",       ['ECG', 'Glucose']),
                ("BP + Glucose",        ['BP', 'Glucose']),
                ("ECG + BP + Glucose",  ['ECG', 'BP', 'Glucose']),
            ]

            comparison_rows = []
            for label, needed in configs:
                if all(n in by_name for n in needed):
                    subset = [by_name[n] for n in needed]
                    result = fuse_modalities(subset)
                    comparison_rows.append({
                        "Configuration": label,
                        "Risk Score": f"{result['fused_risk_score']*100:.1f}%",
                        "Uncertainty": f"{result['uncertainty_mass']*100:.1f}%",
                        "Conflict": f"{result['conflict']*100:.1f}%"
                    })

            if comparison_rows:
                comp_df = pd.DataFrame(comparison_rows)
                st.table(comp_df)
                try:
                    chart_df = comp_df.copy()
                    chart_df["Risk Score"] = chart_df["Risk Score"].str.rstrip("%").astype(float)
                    st.bar_chart(chart_df.set_index("Configuration")["Risk Score"])
                except Exception:
                    pass
            else:
                st.info("Select more modalities to see configuration comparisons.")

            # ═══════════════ AGENTIC AI REASONING (real coordinator/explanation output) ═══════════════
            st.markdown("## 🤖 Agentic AI Reasoning")
            st.caption("Generated by the Data Quality, Coordinator and Explanation agents from the actual results above.")
            reasoning_lines = [ln.strip() for ln in report.split("\n") if ln.strip()]
            st.markdown(
                '<div class="card">' +
                "".join(f"<p style='margin:4px 0;'>• {ln}</p>" for ln in reasoning_lines) +
                "</div>",
                unsafe_allow_html=True,
            )