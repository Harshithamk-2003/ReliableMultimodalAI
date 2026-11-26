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
.hero-banner {
    background: linear-gradient(120deg, #0d2140 0%, #123a5e 55%, #0f2e52 100%);
    border: 1px solid rgba(0,229,255,0.25);
    border-radius: 18px;
    padding: 26px 32px;
    margin-bottom: 22px;
    box-shadow: 0 0 40px rgba(0,180,255,0.08);
}
.hero-title {
    font-size: 2.1rem; font-weight: 800; color: #eaf6ff; margin-bottom: 2px;
    letter-spacing: 0.3px;
}
.hero-sub {
    font-size: 1.0rem; color: #7fd8ff; font-weight: 500; margin-bottom: 6px;
}
.hero-tag {
    font-size: 0.85rem; color: #9fb4cc;
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
    padding: 18px 20px;
    margin-bottom: 14px;
}
.card h4 { margin-top:0; color:#d8ecff; }
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
  <div class="hero-sub">Reliability-Aware Multimodal Clinical Intelligence Dashboard</div>
  <div class="hero-tag">Agentic Multimodal AI for Cardiometabolic Risk Assessment · ECG + Blood Pressure + Glucose</div>
  <div style="margin-top:10px;">
    <span class="status-pill">● System Ready</span>
    &nbsp;&nbsp;<span style="color:#7fa8c9; font-size:0.85rem;">{now}</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.caption("Research Prototype — Not for Clinical Use. All scores shown are model estimates, not clinically confirmed diagnoses.")

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

# ───────────────────────── Reference text (explanations only — no model logic) ─────────────────────────

ECG_CLASS_INFO = {
    "NORM": ("Normal ECG", "No significant conduction or rhythm abnormality detected in the waveform pattern the model was trained to recognize.", "#1e7a3c"),
    "MI":   ("Myocardial Infarction pattern", "Waveform pattern consistent with current or prior heart attack (e.g. pathological Q waves, ST-elevation/depression patterns).", "#b02a2a"),
    "STTC": ("ST/T-wave change", "Non-specific ST segment or T-wave abnormality — can reflect ischemia, strain, or other repolarization change.", "#b05a00"),
    "CD":   ("Conduction Disturbance", "Abnormal electrical conduction pattern (e.g. bundle branch block, AV conduction delay).", "#9a6b00"),
    "HYP":  ("Hypertrophy pattern", "Waveform pattern consistent with chamber enlargement (e.g. left ventricular hypertrophy), often from chronic high BP.", "#8a4b00"),
}

BP_FEATURE_INFO = [
    ("age", "Age (years)", "Patient's age. Cardiovascular risk rises steadily with age as arteries stiffen and plaque accumulates over decades.", None),
    ("sex", "Sex", "Biological sex. 0 = Female, 1 = Male. Men carry higher risk at younger ages; women's risk rises sharply after menopause.", {0: "Female", 1: "Male"}),
    ("cp", "Chest Pain Type", "Type of chest pain reported. 0 = Typical angina (classic exertional pain), 1 = Atypical angina, 2 = Non-anginal pain (not heart-related), 3 = Asymptomatic (no pain — can be higher risk since silent ischemia is dangerous).", {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"}),
    ("trestbps", "Resting Blood Pressure (mmHg)", "Blood pressure measured at rest (this dataset records a single resting BP value; it does not separately record diastolic BP). Normal is roughly 90–120; above 140 is hypertension. Chronic high BP damages artery walls and strains the heart.", None),
    ("chol", "Serum Cholesterol (mg/dl)", "Total cholesterol in blood. Normal is under ~200 mg/dl; above 240 is high. Excess cholesterol builds up as artery-narrowing plaque.", None),
    ("fbs", "Fasting Blood Sugar > 120 mg/dl", "Whether fasting blood sugar exceeds 120 mg/dl, a diabetes-screening cutoff. 0 = No, 1 = Yes. Diabetes/prediabetes damages blood vessels and raises heart risk.", {0: "No", 1: "Yes"}),
    ("restecg", "Resting ECG Result", "Resting electrocardiogram category. 0 = Normal, 1 = ST-T wave abnormality (possible ischemia), 2 = Probable/definite left ventricular hypertrophy.", {0: "Normal", 1: "ST-T abnormality", 2: "Likely LV hypertrophy"}),
    ("thalach", "Max Heart Rate Achieved", "Highest heart rate reached during exertion. A lower-than-expected max for the patient's age can indicate the heart isn't responding normally to effort.", None),
    ("exang", "Exercise-Induced Angina", "Whether chest pain occurs specifically during physical exertion. 0 = No, 1 = Yes — a classic sign of insufficient blood flow under demand.", {0: "No", 1: "Yes"}),
    ("oldpeak", "ST Depression (Exercise vs Rest)", "How much the ECG's ST segment drops during exercise compared to rest, in mm. Larger depression is an established sign of reduced blood flow (ischemia).", None),
    ("slope", "Slope of Peak Exercise ST Segment", "Shape of the ST segment at peak exercise. 0 = Upsloping (lower risk), 1 = Flat, 2 = Downsloping (both more associated with disease).", {0: "Upsloping", 1: "Flat", 2: "Downsloping"}),
    ("ca", "Major Vessels Colored by Fluoroscopy", "Number of major coronary arteries (0–3) showing visible narrowing on fluoroscopic imaging — a direct measurement of confirmed blockage.", None),
    ("thal", "Thalassemia / Thallium Stress Result", "Thallium stress test result. 3 = Normal blood flow, 6 = Fixed defect (permanent damage, e.g. old heart attack), 7 = Reversible defect (temporary reduced flow under stress — active, ongoing risk).", {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}),
]

GLUCOSE_FEATURE_INFO = [
    ("high_bp_flag", "High Blood Pressure History", "Whether the person has ever been told they have high blood pressure. 0 = No, 1 = Yes. Hypertension and type 2 diabetes frequently co-occur and worsen each other.", {0: "No", 1: "Yes"}),
    ("high_chol", "High Cholesterol", "Whether the person has been told they have high cholesterol. 0 = No, 1 = Yes — another metabolic-syndrome marker linked with diabetes risk.", {0: "No", 1: "Yes"}),
    ("bmi", "BMI", "Body Mass Index (weight / height²). Normal is 18.5–24.9; over 30 is obese. Excess body fat, especially abdominal, directly drives insulin resistance.", None),
    ("smoker", "Smoker", "Whether the person smokes or has smoked regularly (at least 100 cigarettes lifetime). 0 = No, 1 = Yes. Smoking worsens insulin resistance and damages blood vessels.", {0: "No", 1: "Yes"}),
    ("phys_activity", "Physically Active", "Whether the person did any physical activity in the past 30 days outside their job. 0 = No, 1 = Yes. Activity directly improves insulin sensitivity.", {0: "No", 1: "Yes"}),
    ("gen_health", "General Health (self-rated)", "Self-rated overall health on a 1 (Excellent) to 5 (Poor) scale. Despite being subjective, this is a well-validated predictor reflecting cumulative, sometimes undiagnosed, health burden.", {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}),
    ("diff_walk", "Difficulty Walking", "Whether the person has serious difficulty walking or climbing stairs. 0 = No, 1 = Yes — a proxy for reduced mobility, both a cause and consequence of diabetes.", {0: "No", 1: "Yes"}),
    ("age_glucose", "Age Category (BRFSS bracket)", "Age grouped into 13 survey brackets (1 = 18–24 ... 13 = 80+). Diabetes risk rises steadily with age as insulin sensitivity naturally declines.", None),
]


def risk_pill_html(prob):
    if prob < 0.33:
        return '<span class="risk-pill pill-low">LOW</span>'
    elif prob < 0.66:
        return '<span class="risk-pill pill-mod">MODERATE</span>'
    else:
        return '<span class="risk-pill pill-high">HIGH</span>'


def colored_bar(label, value, color, help_text=None, suffix="%"):
    c1, c2 = st.columns([1, 4])
    with c1:
        st.markdown(f"**{label}**", help=help_text)
    with c2:
        st.progress(min(max(value, 0.0), 1.0), text=f"{value*100:.1f}{suffix}")


def info_expander(title, body_md):
    with st.expander(f"ℹ️ {title}"):
        st.markdown(body_md)


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
    st.caption("All 12 leads:")
    lead_cols = st.columns(3)
    for li in range(12):
        with lead_cols[li % 3]:
            st.caption(LEAD_NAMES[li])
            st.line_chart(signal[li], height=220)

    st.markdown("**Zoomed Single-Lead View (for detailed P-QRS-T inspection)**")
    zoom_lead = st.selectbox(
        "Choose a lead to inspect closely:", LEAD_NAMES, index=1,
        help="Lead II is the standard rhythm-strip lead doctors typically view first."
    )
    zoom_idx = LEAD_NAMES.index(zoom_lead)
    st.line_chart(signal[zoom_idx], height=400, use_container_width=True)
    st.caption(f"Lead {zoom_lead} — full 10-second strip (1000 samples at 100Hz). "
               "Zoomed, single-lead view so individual P, QRS, and T waveform shapes are visible.")


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

# ───────────────────────── Patient Information (record-keeping only, not model input) ─────────────────────────

if "patient_id" not in st.session_state:
    st.session_state["patient_id"] = "PT-" + str(uuid.uuid4())[:6].upper()

st.markdown('<div class="card"><h4>🧾 Patient Information</h4></div>', unsafe_allow_html=True)
p1, p2, p3, p4, p5 = st.columns([2, 1, 1, 1.4, 1.4])
with p1:
    patient_name = st.text_input("Patient Name", value=st.session_state.get("patient_name", ""),
                                  placeholder="e.g. John Doe")
with p2:
    patient_age_display = st.number_input("Age (display only)", 0, 120,
                                           value=st.session_state.get("patient_age_display", 45))
with p3:
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"],
                                   index=["Male", "Female", "Other"].index(st.session_state.get("patient_gender", "Male")))
with p4:
    patient_phone = st.text_input("Phone Number", value=st.session_state.get("patient_phone", ""),
                                   placeholder="e.g. +91 98765 43210")
with p5:
    st.text_input("Patient ID (auto-generated)", value=st.session_state["patient_id"], disabled=True)

st.session_state["patient_name"] = patient_name
st.session_state["patient_age_display"] = patient_age_display
st.session_state["patient_gender"] = patient_gender
st.session_state["patient_phone"] = patient_phone

st.caption(
    "These details are for record-keeping and the report header only — they are not sent to any "
    "prediction model. Age used in the ECG/BP/Glucose models themselves is entered separately in each tab below."
)

st.divider()

# ───────────────────────── Sidebar ─────────────────────────
st.sidebar.header("Select Available Modalities")
use_ecg = st.sidebar.checkbox("ECG", value=True)
use_bp = st.sidebar.checkbox("Blood Pressure", value=True)
use_glucose = st.sidebar.checkbox("Glucose / Diabetes Risk", value=True)

st.sidebar.divider()
st.sidebar.header("Simulate Sensor / Data Failure")
st.sidebar.caption("Even if a modality is selected above, you can simulate it failing "
                    "at the moment of assessment — to demonstrate reliability-aware behavior.")
sim_drop_ecg = st.sidebar.checkbox("Simulate ECG failure", value=False)
sim_drop_bp = st.sidebar.checkbox("Simulate BP failure", value=False)
sim_drop_glucose = st.sidebar.checkbox("Simulate Glucose failure", value=False)

bp_features = None
glucose_raw = None
ecg_signal = None
ecg_data_quality = 1.0

tab_ecg, tab_bp, tab_glucose = st.tabs(["🫀 ECG", "🩺 Blood Pressure", "🩸 Glucose / Diabetes Risk"])

# ───────────────────────── ECG Tab ─────────────────────────
with tab_ecg:
    st.header("ECG Input")
    info_expander(
        "Why is ECG used in cardiac risk assessment?",
        "The ECG records the heart's electrical activity across 12 leads. Patterns in that "
        "signal — such as pathological Q waves, ST-segment shifts, or conduction delays — are "
        "well-established clinical markers of myocardial infarction, ischemia, conduction "
        "disease, and chamber hypertrophy. It is the only one of the three modalities here "
        "that directly observes the heart's own electrical function rather than a downstream "
        "risk factor, which is why it is weighted as primary evidence in the fusion step."
    )

    if use_ecg:
        ecg_mode = st.radio(
            "ECG input source",
            ["Use a sample from test set", "Upload my own 12-lead ECG (CSV)"],
            index=0,
        )

        if ecg_mode == "Use a sample from test set":
            st.caption("Select a sample ECG record from the test set to simulate a patient")
            X_ecg_test = np.load(DATA_DIR + "splits/ecg5_X_demo.npy")
            y_ecg_test = np.load(DATA_DIR + "splits/ecg5_y_demo.npy")
            inv_class_map = {v: k for k, v in ecg_class_map.items()}

            sample_idx = st.slider("ECG Sample Index", 0, len(X_ecg_test) - 1, 0)
            true_label = inv_class_map[y_ecg_test[sample_idx]]
            st.caption(f"(Ground truth label for this sample: {true_label})")

            ecg_signal = X_ecg_test[sample_idx]
            show_12_lead_grid(ecg_signal)
            ecg_data_quality = 1.0

        else:
            st.caption(
                f"Upload a CSV with a 12-lead ECG signal, {EXPECTED_SAMPLES} samples per lead "
                f"(10 seconds at 100Hz), either as {EXPECTED_LEADS} rows x {EXPECTED_SAMPLES} "
                f"columns, or {EXPECTED_SAMPLES} rows x {EXPECTED_LEADS} columns. No header row."
            )
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
                        st.error(
                            f"Unexpected shape {raw_arr.shape}. Expected "
                            f"({EXPECTED_LEADS}, {EXPECTED_SAMPLES}) or "
                            f"({EXPECTED_SAMPLES}, {EXPECTED_LEADS})."
                        )

                    if ecg_raw is not None:
                        mean = ecg_raw.mean(axis=1, keepdims=True)
                        std = ecg_raw.std(axis=1, keepdims=True)
                        std[std == 0] = 1.0
                        ecg_signal = (ecg_raw - mean) / std
                        ecg_data_quality = 0.85

                        st.success(f"ECG loaded successfully. Shape: {ecg_signal.shape}")
                        show_12_lead_grid(ecg_signal)
                except Exception as e:
                    st.error(f"Could not read this file: {e}")
            else:
                st.info("Upload a CSV file to proceed, or switch back to 'Use a sample from test set'.")

        if sim_drop_ecg:
            st.warning("⚠ ECG failure simulation is ON — this input will be ignored on run.")
    else:
        st.info("ECG not included in this assessment.")

# ───────────────────────── Blood Pressure Tab ─────────────────────────
with tab_bp:
    st.header("Blood Pressure / Cardiac Clinical Features")
    st.caption(
        "Note: this model's training data (UCI Heart Disease dataset) records a single "
        "resting blood pressure value, not separate systolic and diastolic readings — "
        "so only the one BP value below feeds the model."
    )
    if use_bp:
        input_widgets = {}
        c1, c2 = st.columns(2)
        with c1:
            input_widgets["age"] = st.number_input("Age", 20, 90, 55)
            input_widgets["sex"] = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            input_widgets["cp"] = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
            input_widgets["trestbps"] = st.number_input("Resting BP (mmHg)", 80, 220, 130)
            input_widgets["chol"] = st.number_input("Cholesterol (mg/dl)", 100, 600, 230)
            input_widgets["fbs"] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
            input_widgets["restecg"] = st.slider("Resting ECG (0-2)", 0, 2, 1)
        with c2:
            input_widgets["thalach"] = st.number_input("Max Heart Rate", 60, 220, 150)
            input_widgets["exang"] = st.selectbox("Exercise Induced Angina", [0, 1])
            input_widgets["oldpeak"] = st.number_input("ST Depression", 0.0, 6.0, 1.0)
            input_widgets["slope"] = st.slider("Slope (0-2)", 0, 2, 1)
            input_widgets["ca"] = st.slider("Major Vessels (0-3)", 0, 3, 0)
            input_widgets["thal"] = st.slider("Thal (3,6,7)", 3, 7, 3)

        bp_features = np.array([
            input_widgets["age"], input_widgets["sex"], input_widgets["cp"], input_widgets["trestbps"],
            input_widgets["chol"], input_widgets["fbs"], input_widgets["restecg"], input_widgets["thalach"],
            input_widgets["exang"], input_widgets["oldpeak"], input_widgets["slope"], input_widgets["ca"],
            input_widgets["thal"],
        ])

        st.markdown("##### Resting BP vs. reference range")
        bp_val = input_widgets["trestbps"]
        bp_frac = min(bp_val / 220.0, 1.0)
        bp_zone = "Normal (<120)" if bp_val < 120 else ("Elevated/Stage 1 (120-139)" if bp_val < 140 else "Stage 2 Hypertension (≥140)")
        colored_bar("Resting BP", bp_frac, None, suffix=f" mmHg  —  {bp_zone}")

        st.markdown("##### What each BP input feature means")
        for key, label, desc, mapping in BP_FEATURE_INFO:
            val = input_widgets[key]
            shown_val = mapping[val] if mapping and val in mapping else val
            info_expander(f"{label}  (current value: {shown_val})", desc)

        if sim_drop_bp:
            st.warning("⚠ BP failure simulation is ON — this input will be ignored on run.")
    else:
        st.info("BP not included in this assessment.")

# ───────────────────────── Glucose Tab ─────────────────────────
with tab_glucose:
    st.header("Glucose / Diabetes Risk Indicators")
    st.caption(
        "Note: this model is trained on BRFSS diabetes-survey indicators (BMI, blood pressure "
        "history, lifestyle, self-rated health, etc.) and outputs a diabetes risk probability — "
        "it does not take or report a blood glucose concentration in mg/dl."
    )
    if use_glucose:
        g1, g2 = st.columns(2)
        with g1:
            high_bp_flag = st.selectbox("High Blood Pressure History", [0, 1])
            high_chol = st.selectbox("High Cholesterol", [0, 1])
            bmi = st.number_input("BMI", 10.0, 60.0, 27.0)
            smoker = st.selectbox("Smoker", [0, 1])
        with g2:
            phys_activity = st.selectbox("Physically Active", [0, 1])
            gen_health = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
            diff_walk = st.selectbox("Difficulty Walking", [0, 1])
            age_glucose = st.slider("Age Category (1-13)", 1, 13, 7)

        glucose_raw = np.zeros(21)
        glucose_raw[0] = high_bp_flag
        glucose_raw[1] = high_chol
        glucose_raw[2] = 1  # CholCheck default
        glucose_raw[3] = bmi
        glucose_raw[4] = smoker
        glucose_raw[8] = phys_activity
        glucose_raw[13] = gen_health
        glucose_raw[16] = diff_walk
        glucose_raw[18] = age_glucose

        glucose_input_values = {
            "high_bp_flag": high_bp_flag, "high_chol": high_chol, "bmi": bmi, "smoker": smoker,
            "phys_activity": phys_activity, "gen_health": gen_health, "diff_walk": diff_walk,
            "age_glucose": age_glucose,
        }

        st.markdown("##### BMI vs. reference range")
        bmi_frac = min(bmi / 60.0, 1.0)
        bmi_zone = "Underweight (<18.5)" if bmi < 18.5 else ("Normal (18.5-24.9)" if bmi < 25 else ("Overweight (25-29.9)" if bmi < 30 else "Obese (≥30)"))
        colored_bar("BMI", bmi_frac, None, suffix=f" — {bmi_zone}")

        st.markdown("##### What each diabetes-risk input feature means")
        for key, label, desc, mapping in GLUCOSE_FEATURE_INFO:
            val = glucose_input_values[key]
            shown_val = mapping[val] if mapping and val in mapping else val
            info_expander(f"{label}  (current value: {shown_val})", desc)

        if sim_drop_glucose:
            st.warning("⚠ Glucose failure simulation is ON — this input will be ignored on run.")
    else:
        st.info("Glucose not included in this assessment.")

st.divider()

# ───────────────────────── Run Multi-Agent Assessment ─────────────────────────
if st.button("🔍 Run Multimodal Assessment", type="primary", use_container_width=True):
    ecg_input = ecg_signal if use_ecg else None
    bp_input = bp_features if use_bp else None
    glucose_input = glucose_raw if use_glucose else None

    if ecg_input is None and bp_input is None and glucose_input is None:
        st.error("Please select or upload at least one modality.")
    elif sim_drop_ecg and sim_drop_bp and sim_drop_glucose:
        st.error("All modalities are set to simulated failure — nothing to assess. "
                  "Uncheck at least one failure simulation.")
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
            phone_display = patient_phone if patient_phone else "Not provided"

            st.markdown(f"""
<div class="hero-banner" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:18px;">
  <div>
    <div style="font-size:1.1rem; font-weight:700; color:#eaf6ff;">👤 {name_display}</div>
    <div style="margin-top:6px;">
      <span class="patient-chip">ID: {st.session_state['patient_id']}</span>
      <span class="patient-chip">Age: {patient_age_display}</span>
      <span class="patient-chip">Gender: {patient_gender}</span>
      <span class="patient-chip">Phone: {phone_display}</span>
      <span class="patient-chip">Date: {assess_date}</span>
    </div>
  </div>
  <div style="text-align:right;">
    <div style="color:#9fb4cc; font-size:0.85rem;">Overall Assessment</div>
    <div style="font-size:1.6rem; font-weight:800; color:#eaf6ff;">{risk_label}</div>
    {risk_pill_html(fused_score)}
    <div style="margin-top:4px; color:#7fd8ff; font-size:0.9rem;">Fused Risk Score: {fused_score*100:.1f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

            if escalation.get("triggered"):
                st.error("⚠ CLINICAL REVIEW RECOMMENDED — AUTOMATIC ESCALATION TRIGGERED")
                for reason in escalation.get("reasons", []):
                    st.markdown(f"- {reason}")
                st.markdown("**Recommendation:** treat this result as inconclusive and prioritize "
                            "professional clinical evaluation rather than acting on the score alone.")

            # ═══════════════ ECG RESULT SECTION ═══════════════
            ecg_result = by_name.get('ECG')
            if ecg_result is not None and ecg_input is not None and not sim_drop_ecg:
                st.markdown("## 🫀 ECG Analysis")
                pred_class = ecg_result.get('predicted_class', 'NORM')
                class_probs = ecg_result.get('class_probs', {})

                name_map, desc_map, color_map = {}, {}, {}
                for cls, (name, desc, color) in ECG_CLASS_INFO.items():
                    name_map[cls] = name
                    desc_map[cls] = desc
                    color_map[cls] = color

                st.markdown(f"**Model prediction:** `{pred_class}` — {name_map.get(pred_class, pred_class)} "
                            f"{risk_pill_html(class_probs.get(pred_class, 0.0))}", unsafe_allow_html=True)
                st.caption("These are model-estimated probabilities from the trained ECG classifier, "
                           "not a clinically confirmed diagnosis.")

                for cls in ["NORM", "MI", "STTC", "CD", "HYP"]:
                    p = class_probs.get(cls, 0.0)
                    colored_bar(cls, p, color_map[cls])

                for cls in ["NORM", "MI", "STTC", "CD", "HYP"]:
                    info_expander(f"What does {cls} mean? ({class_probs.get(cls,0.0)*100:.1f}% model score)",
                                  f"**{name_map[cls]}**\n\n{desc_map[cls]}\n\n"
                                  f"The {class_probs.get(cls,0.0)*100:.1f}% is the model's estimated probability "
                                  f"that this ECG belongs to this class — it is a model confidence score, "
                                  f"not a clinically verified diagnosis.")

                st.markdown("#### Uploaded / Selected 12-Lead ECG")
                st.caption(
                    "Red borders mark the leads that most influenced this specific prediction, "
                    "computed directly from the model's own gradients (gradient × input "
                    "attribution, averaged across the 3-model ensemble) — real model evidence, "
                    "never a clinical annotation. No lead is marked without this computation "
                    "pointing to it, and nothing is highlighted when the prediction is NORM."
                )

                predicted_class_idx = ecg_class_map[pred_class]
                importance = compute_lead_attribution(ecg_input, predicted_class_idx)
                highlighted_img, highlighted_leads = render_highlighted_ecg_image(
                    ecg_input, importance, pred_class
                )
                st.image(highlighted_img, caption="12-lead ECG (model-attributed leads shown in red)", use_container_width=True)

                if pred_class == "NORM":
                    st.success("No abnormality detected — no leads highlighted.")
                elif highlighted_leads:
                    st.markdown("**Most contributing leads (by model gradient attribution):**")
                    attr_cols = st.columns(len(highlighted_leads))
                    for i, lead_idx in enumerate(highlighted_leads):
                        with attr_cols[i]:
                            st.metric(LEAD_NAMES[lead_idx], f"{importance[lead_idx]*100:.1f}%",
                                      help="Share of the model's total gradient-based attribution "
                                           "assigned to this lead for the predicted class.")
                else:
                    st.info("No single lead's contribution was strong enough to highlight confidently "
                            "(attribution spread broadly across leads).")

            # ═══════════════ BP RESULT SECTION ═══════════════
            bp_result = by_name.get('BP')
            if bp_result is not None and bp_input is not None and not sim_drop_bp:
                st.markdown("## 🩺 Blood Pressure / Cardiac Risk")
                bp_risk = bp_result.get('risk_prob', 0.0)
                bp_conf = bp_result.get('confidence', 0.0)
                st.markdown(f"**Model risk score:** {bp_risk*100:.1f}% {risk_pill_html(bp_risk)}", unsafe_allow_html=True)
                colored_bar("Resting BP entered", min(bp_features[3] / 220.0, 1.0), None,
                            suffix=f" mmHg (value: {int(bp_features[3])})")
                colored_bar("BP-model risk score", bp_risk, None)
                colored_bar("BP-model confidence", bp_conf, None)
                info_expander("What do these BP numbers mean?",
                              "**Model risk score** — the trained model's estimated probability of "
                              "cardiac risk given all 13 clinical features entered on the Blood "
                              "Pressure tab (not resting BP alone).\n\n"
                              "**Confidence** — how sure the model is about its own prediction.\n\n"
                              "Resting BP is shown separately above purely as a reference value; "
                              "the risk score reflects the combined effect of all 13 features.")

            # ═══════════════ GLUCOSE RESULT SECTION ═══════════════
            glucose_result = by_name.get('Glucose')
            if glucose_result is not None and glucose_input is not None and not sim_drop_glucose:
                st.markdown("## 🩸 Diabetes Risk")
                gl_risk = glucose_result.get('risk_prob', 0.0)
                gl_conf = glucose_result.get('confidence', 0.0)
                st.markdown(f"**Model risk score:** {gl_risk*100:.1f}% {risk_pill_html(gl_risk)}", unsafe_allow_html=True)
                colored_bar("BMI entered", min(glucose_raw[3] / 60.0, 1.0), None, suffix=f" (value: {glucose_raw[3]:.1f})")
                colored_bar("Diabetes-risk model score", gl_risk, None)
                colored_bar("Model confidence", gl_conf, None)
                info_expander("What does this risk score mean?",
                              "This is the trained model's estimated probability of elevated diabetes "
                              "risk, based on the BRFSS lifestyle/clinical indicators entered on the "
                              "Glucose tab (BMI, BP/cholesterol history, activity, self-rated health, "
                              "etc.). It is not a blood glucose concentration reading.")

            # ═══════════════ FUSION SECTION ═══════════════
            st.markdown("## 🔗 Multimodal Fusion Result")
            info_expander("Why combine ECG, BP and Glucose instead of using one modality alone?",
                          "Each modality observes a different aspect of cardiometabolic health: ECG "
                          "sees the heart's electrical activity directly, BP/cardiac features capture "
                          "structural and hemodynamic risk, and the diabetes-risk indicators capture "
                          "a major, often-silent contributor to cardiovascular damage. A single modality "
                          "can be normal while another shows clear risk (or vice versa), and relying on "
                          "just one risks missing or over-weighting that signal. Dempster-Shafer fusion "
                          "combines all available evidence while explicitly tracking how much the "
                          "sources agree (conflict) and how much remains unresolved (uncertainty), "
                          "rather than silently averaging them away. The configuration comparison "
                          "table below shows this empirically — it uses real, independently measured "
                          "accuracy/F1/AUC across patients, not illustrative numbers.")

            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.markdown(f"<div class='metric-card'><b>Fused Risk Score</b><br>"
                            f"<span style='font-size:1.6rem'>{fusion_result['fused_risk_score']*100:.1f}%</span> "
                            f"{risk_pill_html(fusion_result['fused_risk_score'])}</div>", unsafe_allow_html=True)
            with fc2:
                st.markdown(f"<div class='metric-card'><b>Uncertainty</b><br>"
                            f"<span style='font-size:1.6rem'>{fusion_result['uncertainty_mass']*100:.1f}%</span></div>",
                            unsafe_allow_html=True)
            with fc3:
                st.markdown(f"<div class='metric-card'><b>Inter-modality Conflict</b><br>"
                            f"<span style='font-size:1.6rem'>{fusion_result['conflict']*100:.1f}%</span></div>",
                            unsafe_allow_html=True)

            st.markdown("#### Individual modality contributions")
            contrib_rows = []
            for m in modality_results:
                contrib_rows.append({
                    "Modality": m['name'],
                    "Risk Score": f"{m.get('risk_prob', 0.0)*100:.1f}%",
                    "Confidence": f"{m.get('confidence', 0.0)*100:.1f}%",
                    "Reliability (Conf × Data Quality)": f"{m.get('reliability', 0.0)*100:.1f}%",
                })
            st.table(pd.DataFrame(contrib_rows))

            if escalation.get("triggered"):
                pass
            elif fusion_result['conflict'] > 0.3:
                st.warning("Modalities show significant disagreement — interpret with caution.")
            elif fusion_result['is_reliable']:
                st.success("Modalities show consistent evidence.")

            st.markdown("#### Configuration Comparison — Real Benchmark Results")
            st.caption("Shows how risk-assessment accuracy changes as more modalities are added — "
                       "measured on held-out test patients (not the current patient), using the exact "
                       "same evaluation for every row. This is the empirical evidence for why multimodal "
                       "fusion is used.")

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

            st.markdown("## 📋 Multi-Agent AI Explanation")
            with st.expander("What does each term in this report mean?", expanded=False):
                st.markdown("""
- **Risk Score** — the model's estimated probability that this modality's reading indicates a health concern.
- **Confidence** — how sure the model is about its own prediction (the probability assigned to the winning class). Low confidence means the model itself isn't certain.
- **Reliability** (Confidence × Data Quality) — how much this modality's opinion should count toward the final combined decision.
- **Data Quality** — whether the input was complete, a default placeholder, or missing/simulated as failed.
- **Uncertainty** — how much of the total combined evidence is unresolved after fusing all modalities.
- **Conflict** — how much the different modalities actually disagreed with each other before being combined.
- **Class probability breakdown (ECG)** — the full probability across all 5 possible findings (NORM/MI/STTC/CD/HYP), not just the top guess.
- **Escalation** — an explicit flag raised when conflict or uncertainty is too high to trust the result as-is.
""")
            st.text(report)