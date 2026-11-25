import sys
import os
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_BASE_DIR, "src", "fusion"))
from multimodal_fusion import (
    glucose_model, glucose_scaler, bp_model, bp_scaler,
    ecg_models, ecg_class_map, device, fuse_modalities
)

from data_quality_agent import DataQualityAgent
from ecg_agent import ECGAgent
from bp_agent import BPAgent
from glucose_agent import GlucoseAgent
from fusion_agent import FusionAgent
from explanation_agent import ExplanationAgent

CONFLICT_ESCALATION_THRESHOLD = 0.30
UNCERTAINTY_ESCALATION_THRESHOLD = 0.35


class CoordinatorAgent:
    name = "Coordinator Agent"

    def __init__(self):
        self.quality_agent = DataQualityAgent()
        self.ecg_agent = ECGAgent(ecg_models, ecg_class_map, device)
        self.bp_agent = BPAgent(bp_model, bp_scaler)
        self.glucose_agent = GlucoseAgent(glucose_model, glucose_scaler)
        self.fusion_agent = FusionAgent(fuse_modalities)
        self.explanation_agent = ExplanationAgent()

    def run(self, ecg_signal=None, bp_features=None, glucose_features=None,
            drop_ecg=False, drop_bp=False, drop_glucose=False):
        modality_results = []
        quality_flags = []

        if ecg_signal is not None and not drop_ecg:
            q, flags = self.quality_agent.check("ECG")
            quality_flags.extend(flags)
            modality_results.append(self.ecg_agent.analyze(ecg_signal, data_quality=q))
        elif ecg_signal is not None and drop_ecg:
            quality_flags.append("ECG: simulated sensor failure — modality excluded from assessment")

        if bp_features is not None and not drop_bp:
            q, flags = self.quality_agent.check("BP")
            quality_flags.extend(flags)
            modality_results.append(self.bp_agent.analyze(bp_features, data_quality=q))
        elif bp_features is not None and drop_bp:
            quality_flags.append("BP: simulated sensor failure — modality excluded from assessment")

        if glucose_features is not None and not drop_glucose:
            q, flags = self.quality_agent.check("Glucose")
            quality_flags.extend(flags)
            modality_results.append(self.glucose_agent.analyze(glucose_features, data_quality=q))
        elif glucose_features is not None and drop_glucose:
            quality_flags.append("Glucose: simulated sensor failure — modality excluded from assessment")

        if len(modality_results) == 0:
            return None, "No modalities provided (all dropped or missing).", []

        for m in modality_results:
            m["reliability"] = m["confidence"] * m.get("data_quality", 1.0)

        fusion_result = self.fusion_agent.fuse(modality_results)

        escalation = self._check_escalation(fusion_result, modality_results)
        fusion_result["escalation"] = escalation

        report = self.explanation_agent.explain(
            modality_results, fusion_result, quality_flags, escalation=escalation
        )

        return fusion_result, report, modality_results

    def _check_escalation(self, fusion_result, modality_results):
        conflict = fusion_result.get("conflict", 0.0)
        uncertainty = fusion_result.get("uncertainty_mass", 0.0)

        triggered = False
        reasons = []

        if conflict >= CONFLICT_ESCALATION_THRESHOLD:
            triggered = True
            risk_summary = ", ".join(
                f"{m['name']}={'HIGH' if m['risk_prob'] >= 0.5 else 'LOW'} ({m['risk_prob']*100:.1f}%)"
                for m in modality_results
            )
            reasons.append(
                f"The modalities disagreed significantly ({conflict*100:.1f}% conflict). "
                f"Individual readings: {risk_summary}."
            )

        if uncertainty >= UNCERTAINTY_ESCALATION_THRESHOLD:
            triggered = True
            reasons.append(
                f"Overall uncertainty is high ({uncertainty*100:.1f}%), meaning the system "
                f"does not have enough consistent evidence to be confident in this result."
            )

        if len(modality_results) == 1:
            reasons.append(
                "Only one modality was available for this assessment — results based on a "
                "single test are inherently less reliable than a multimodal combination."
            )

        return {
            "triggered": triggered,
            "reasons": reasons,
        }


if __name__ == "__main__":
    print("Testing coordinator with escalation logic...")
    coordinator = CoordinatorAgent()

    import numpy as np
    bp_test = np.array([65, 1, 0, 170, 290, 1, 1, 110, 1, 3.0, 2, 2, 7])
    glucose_test = np.zeros(21)
    glucose_test[0] = 1
    glucose_test[1] = 1
    glucose_test[3] = 34
    glucose_test[4] = 1
    glucose_test[13] = 4
    glucose_test[16] = 1
    glucose_test[18] = 10

    fusion_result, report, modality_results = coordinator.run(bp_features=bp_test, glucose_features=glucose_test)
    print(report)

    print("\n\n=== TEST: simulated BP sensor failure ===")
    fusion_result2, report2, modality_results2 = coordinator.run(
        bp_features=bp_test, glucose_features=glucose_test, drop_bp=True
    )
    print(report2)
