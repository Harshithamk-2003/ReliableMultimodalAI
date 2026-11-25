import numpy as np
import torch
import sys

sys.path.append("D:\\ReliableMultimodalAI\\src\\fusion")
from multimodal_fusion import (
    glucose_model, glucose_scaler, bp_model, bp_scaler,
    ecg_models, ecg_class_map, device, fuse_modalities
)

# ──────────────────────────────────────────────────────────
# AGENT 1 — Data Quality Agent
# Job: inspect raw inputs before any model sees them, flag issues
# ──────────────────────────────────────────────────────────
class DataQualityAgent:
    name = "Data Quality Agent"

    def check(self, modality_name, raw_value=None, is_missing=False, is_default=False):
        """Returns a data_quality multiplier (0 to 1) and a list of flags."""
        flags = []
        quality = 1.0

        if is_missing:
            flags.append(f"{modality_name}: input not provided")
            quality = 0.0
        elif is_default:
            flags.append(f"{modality_name}: using default/placeholder values — reduced trust")
            quality = 0.7

        return quality, flags


# ──────────────────────────────────────────────────────────
# AGENT 2 — ECG Analysis Agent
# Job: run the ECG ensemble model and report risk + confidence
# ──────────────────────────────────────────────────────────
class ECGAgent:
    name = "ECG Analysis Agent"

    def analyze(self, ecg_signal_12x1000, data_quality=1.0):
        x = torch.FloatTensor(ecg_signal_12x1000).unsqueeze(0).to(device)
        probs_sum = torch.zeros(1, 5).to(device)
        with torch.no_grad():
            for m in ecg_models:
                probs_sum += torch.softmax(m(x), dim=1)
        probs = (probs_sum / len(ecg_models)).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        risk_prob = float(1 - probs[ecg_class_map['NORM']])
        inv_map = {v: k for k, v in ecg_class_map.items()}

        return {
            'name': 'ECG',
            'risk_prob': risk_prob,
            'confidence': confidence,
            'data_quality': data_quality,
            'predicted_class': inv_map[pred_class],
            'class_probs': {k: float(probs[v]) for k, v in ecg_class_map.items()},
            'agent': self.name
        }


# ──────────────────────────────────────────────────────────
# AGENT 3 — Blood Pressure Agent
# Job: run the BP model and report risk + confidence
# ──────────────────────────────────────────────────────────
class BPAgent:
    name = "Blood Pressure Agent"

    def analyze(self, bp_features_raw, data_quality=1.0):
        X = bp_scaler.transform(bp_features_raw.reshape(1, -1))
        prob = float(bp_model.predict_proba(X)[0, 1])
        confidence = max(prob, 1 - prob)

        return {
            'name': 'BP',
            'risk_prob': prob,
            'confidence': confidence,
            'data_quality': data_quality,
            'agent': self.name
        }


# ──────────────────────────────────────────────────────────
# AGENT 4 — Glucose Agent
# Job: run the glucose model and report risk + confidence
# ──────────────────────────────────────────────────────────
class GlucoseAgent:
    name = "Glucose Agent"

    def analyze(self, glucose_features_raw, data_quality=1.0):
        X = glucose_scaler.transform(glucose_features_raw.reshape(1, -1))
        prob = float(glucose_model.predict_proba(X)[0, 1])
        confidence = max(prob, 1 - prob)

        return {
            'name': 'Glucose',
            'risk_prob': prob,
            'confidence': confidence,
            'data_quality': data_quality,
            'agent': self.name
        }


# ──────────────────────────────────────────────────────────
# AGENT 5 — Fusion Agent
# Job: combine all modality agent outputs using Dempster-Shafer
# ──────────────────────────────────────────────────────────
class FusionAgent:
    name = "Fusion Agent"

    def fuse(self, modality_results):
        result = fuse_modalities(modality_results)
        result['agent'] = self.name
        return result


# ──────────────────────────────────────────────────────────
# AGENT 6 — Explanation Agent
# Job: turn numbers into a plain-language, evidence-grounded report
# ──────────────────────────────────────────────────────────
class ExplanationAgent:
    name = "Explanation Agent"

    def explain(self, modality_results, fusion_result, quality_flags):
        lines = []
        lines.append("="*60)
        lines.append("AGENTIC AI ANALYSIS REPORT")
        lines.append("="*60)

        lines.append(f"\nAgents consulted: " +
                     ", ".join(set(m.get('agent', m['name']) for m in modality_results)) +
                     f", {FusionAgent.name}, {self.name}")

        available = [m['name'] for m in modality_results]
        lines.append(f"Modalities analyzed: {', '.join(available)}")
        missing = set(['ECG', 'BP', 'Glucose']) - set(available)
        if missing:
            lines.append(f"Note: {', '.join(missing)} data not provided — assessment based on available modalities only.")

        if quality_flags:
            lines.append("\n--- Data Quality Agent Flags ---")
            for f in quality_flags:
                lines.append(f"  - {f}")

        lines.append("\n--- Per-Modality Findings ---")
        for m in modality_results:
            risk_level = "HIGH" if m['risk_prob'] >= 0.6 else ("MODERATE" if m['risk_prob'] >= 0.4 else "LOW")
            reliability = m['confidence'] * m.get('data_quality', 1.0)
            lines.append(f"[{m.get('agent', m['name'])}] {m['name']:10s}: Risk={risk_level} (score={m['risk_prob']:.2f}), "
                         f"Confidence={m['confidence']:.2f}, Reliability={reliability:.2f}")
            if m['confidence'] < 0.6:
                lines.append(f"  WARNING: {m['name']} prediction has low confidence — interpret cautiously.")

        lines.append(f"\n--- {FusionAgent.name} Output ---")
        fused_risk = fusion_result['fused_risk_score']
        uncertainty = fusion_result['uncertainty_mass']
        conflict = fusion_result['conflict']
        risk_label = "HIGH RISK" if fused_risk >= 0.6 else ("MODERATE RISK" if fused_risk >= 0.4 else "LOW RISK")
        lines.append(f"Combined Risk Assessment: {risk_label} (fused score = {fused_risk:.2f})")
        lines.append(f"Uncertainty: {uncertainty:.2f}  |  Inter-modality conflict: {conflict:.2f}")

        if conflict > 0.3:
            lines.append("\nCAUTION: Modalities show significant disagreement — may indicate data")
            lines.append("quality issues or a genuinely complex clinical picture.")
        elif not fusion_result['is_reliable']:
            lines.append("\nNOTE: Fusion confidence reduced due to moderate inter-modality conflict.")
        else:
            lines.append("\nModalities show consistent evidence, supporting the fused assessment.")

        lines.append("\n" + "="*60)
        lines.append("IMPORTANT: This is an AI-generated research prototype output.")
        lines.append("It is NOT a medical diagnosis and must NOT replace professional")
        lines.append("clinical evaluation. Please consult a qualified healthcare provider.")
        lines.append("="*60)

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# AGENT 0 — Coordinator Agent
# Job: orchestrate all agents in the correct order
# ──────────────────────────────────────────────────────────
class CoordinatorAgent:
    name = "Coordinator Agent"

    def __init__(self):
        self.quality_agent = DataQualityAgent()
        self.ecg_agent = ECGAgent()
        self.bp_agent = BPAgent()
        self.glucose_agent = GlucoseAgent()
        self.fusion_agent = FusionAgent()
        self.explanation_agent = ExplanationAgent()

    def run(self, ecg_signal=None, bp_features=None, glucose_features=None):
        modality_results = []
        quality_flags = []

        if ecg_signal is not None:
            q, flags = self.quality_agent.check("ECG", is_missing=False)
            quality_flags.extend(flags)
            modality_results.append(self.ecg_agent.analyze(ecg_signal, data_quality=q))

        if bp_features is not None:
            q, flags = self.quality_agent.check("BP", is_missing=False)
            quality_flags.extend(flags)
            modality_results.append(self.bp_agent.analyze(bp_features, data_quality=q))

        if glucose_features is not None:
            q, flags = self.quality_agent.check("Glucose", is_missing=False)
            quality_flags.extend(flags)
            modality_results.append(self.glucose_agent.analyze(glucose_features, data_quality=q))

        if len(modality_results) == 0:
            return None, "No modalities provided — cannot run assessment."

        for m in modality_results:
            m['reliability'] = m['confidence'] * m.get('data_quality', 1.0)

        fusion_result = self.fusion_agent.fuse(modality_results)
        report = self.explanation_agent.explain(modality_results, fusion_result, quality_flags)

        return fusion_result, report


if __name__ == "__main__":
    print("Multi-agent system loaded. Testing with dummy inputs...\n")
    coordinator = CoordinatorAgent()

    bp_test = np.array([65, 1, 0, 170, 290, 1, 1, 110, 1, 3.0, 2, 2, 7])
    glucose_test = np.zeros(21)
    glucose_test[0] = 1; glucose_test[1] = 1; glucose_test[3] = 34
    glucose_test[4] = 1; glucose_test[13] = 4; glucose_test[16] = 1; glucose_test[18] = 10

    fusion_result, report = coordinator.run(bp_features=bp_test, glucose_features=glucose_test)
    print(report)