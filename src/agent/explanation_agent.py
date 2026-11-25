ECG_DESCRIPTIONS = {
    "NORM": "a normal ECG pattern with no significant abnormality detected",
    "MI":   "signs consistent with a possible myocardial infarction (heart attack) pattern",
    "STTC": "ST/T wave changes, which can indicate ischemia or other cardiac strain",
    "CD":   "a conduction disturbance, affecting how electrical signals travel through the heart",
    "HYP":  "signs consistent with cardiac hypertrophy (enlarged heart muscle)",
}

AGENT_DISPLAY_NAMES = {
    "ECG": "ECG Agent",
    "BP": "Blood Pressure Agent",
    "Glucose": "Glucose Agent",
}

PLAIN_TERMS = {
    "BP": {
        "HIGH": "your blood pressure and related indicators point to a high risk of cardiovascular disease.",
        "MODERATE": "your blood pressure and related indicators suggest a moderate risk of cardiovascular disease.",
        "LOW": "your blood pressure and related indicators suggest a low risk of cardiovascular disease.",
    },
    "Glucose": {
        "HIGH": "your lifestyle and health indicators suggest a high risk of diabetes or blood sugar issues.",
        "MODERATE": "your lifestyle and health indicators suggest a moderate risk of diabetes or blood sugar issues.",
        "LOW": "your lifestyle and health indicators suggest a low risk of diabetes or blood sugar issues.",
    },
}


def risk_word(prob):
    if prob >= 0.70:
        return "HIGH", "a significant concern that needs attention"
    elif prob >= 0.30:
        return "MODERATE", "a moderate concern worth keeping an eye on"
    else:
        return "LOW", "a low level of concern based on current data"


def confidence_word(conf):
    if conf >= 0.90:
        return "very confident"
    elif conf >= 0.70:
        return "fairly confident"
    elif conf >= 0.50:
        return "somewhat confident"
    else:
        return "not very confident"


def make_bar(value, width=20, fill_char="\u2588", empty_char="\u2591"):
    filled = int(round(value * width))
    filled = max(0, min(width, filled))
    return fill_char * filled + empty_char * (width - filled)


class ExplanationAgent:
    name = "Explanation Agent"

    def explain(self, modality_results, fusion_result, quality_flags, escalation=None):
        lines = []
        lines.append("=" * 72)
        lines.append("  MULTIMODAL AI HEALTH ASSESSMENT REPORT")
        lines.append("=" * 72)
        lines.append("")

        overall_risk = fusion_result.get("fused_risk_score", 0.0)
        uncertainty = fusion_result.get("uncertainty_mass", 0.0)
        conflict = fusion_result.get("conflict", 0.0)
        level, meaning = risk_word(overall_risk)

        lines.append(f" OVERALL COMBINED RISK: {level}")
        lines.append(f"   Score:       {overall_risk*100:5.1f}%   [{make_bar(overall_risk)}]")
        lines.append(f"   Meaning:    {meaning}")
        lines.append(f"   Uncertainty:   {uncertainty*100:4.1f}%   (how much the system is still unsure)")
        lines.append(f"   Conflict:     {conflict*100:4.1f}%   (how much the tests disagreed with each other)")
        lines.append("")

        used_names = [m["name"] for m in modality_results]
        all_names = ["ECG", "BP", "Glucose"]
        not_provided = [n for n in all_names if n not in used_names]
        used_str = ", ".join(used_names) if used_names else "none"
        not_provided_str = ", ".join(not_provided) if not_provided else "none"
        lines.append(f"   Modalities used: {used_str}  |  Not provided: {not_provided_str}")
        lines.append("")

        if quality_flags:
            lines.append(" DATA QUALITY NOTES")
            for flag in quality_flags:
                lines.append(f"   - {flag}")
            lines.append("")

        if escalation and escalation.get("triggered"):
            lines.append("=" * 72)
            lines.append(" \u26a0 CLINICAL REVIEW RECOMMENDED \u2014 AUTOMATIC ESCALATION TRIGGERED")
            lines.append("=" * 72)
            lines.append("   The system detected a condition that warrants extra caution before")
            lines.append("   relying on the combined result above:")
            lines.append("")
            for reason in escalation.get("reasons", []):
                lines.append(f"   - {reason}")
            lines.append("")
            lines.append("   Recommendation: treat this result as inconclusive and prioritize")
            lines.append("   professional clinical evaluation rather than acting on the score alone.")
            lines.append("")

        lines.append("-" * 72)
        lines.append(" DETAILED BREAKDOWN BY TEST")
        lines.append("-" * 72)
        lines.append("")

        for m in modality_results:
            name = m["name"]
            agent_display = AGENT_DISPLAY_NAMES.get(name, f"{name} Agent")
            risk_prob = m.get("risk_prob", 0.0)
            confidence = m.get("confidence", 0.0)
            reliability = m.get("reliability", confidence)
            m_level, _ = risk_word(risk_prob)
            conf_desc = confidence_word(confidence)

            lines.append(f" [{name}]  (handled by: {agent_display})")
            lines.append(f"   Risk Score:    {risk_prob*100:5.1f}%   [{make_bar(risk_prob)}]   -> {m_level}")
            lines.append(f"   Confidence:    {confidence*100:5.1f}%   ({conf_desc})")
            lines.append(f"   Reliability:   {reliability*100:5.1f}%   (confidence x data quality)")

            if name == "ECG":
                predicted_class = m.get("predicted_class", "NORM")
                desc = ECG_DESCRIPTIONS.get(predicted_class, "an ECG finding")
                lines.append(f"   In plain terms: the ECG shows {desc}.")
                class_probs = m.get("class_probs", {})
                if class_probs:
                    lines.append("   Class probability breakdown:")
                    for cls, p in sorted(class_probs.items(), key=lambda x: -x[1]):
                        lines.append(f"      {cls:6s} {p*100:5.1f}%   [{make_bar(p, width=15)}]")
            else:
                plain = PLAIN_TERMS.get(name, {}).get(m_level, "")
                if plain:
                    lines.append(f"   In plain terms: {plain}")

            lines.append("")

        lines.append("-" * 72)
        lines.append(" HOW THE FINAL RESULT WAS CALCULATED")
        lines.append("-" * 72)
        lines.append("")
        lines.append("   The Fusion Agent combines all available test results using a method called")
        lines.append("   Dempster-Shafer evidence theory. Instead of simply averaging the scores, it")
        lines.append("   gives MORE weight to tests the system trusts more (high confidence + good data")
        lines.append("   quality) and LESS weight to uncertain or lower-quality readings.")
        lines.append("")
        if conflict >= 0.30:
            lines.append(f"   In this case, the tests DISAGREED noticeably ({conflict*100:.1f}% conflict),")
            lines.append("   so the combined result above should be interpreted with extra caution.")
        else:
            lines.append(f"   In this case, the tests AGREED with each other (only {conflict*100:.1f}% conflict),")
            lines.append("   so the system is reasonably confident in the combined result above.")
        lines.append("")

        lines.append("-" * 72)
        lines.append(" RECOMMENDED NEXT STEP")
        lines.append("-" * 72)
        lines.append("")
        if escalation and escalation.get("triggered"):
            lines.append("   Because of the flagged concerns above, please discuss these results with")
            lines.append("   a doctor rather than relying on this score alone.")
        elif level == "HIGH":
            lines.append("   Your combined risk is on the higher side. Consider discussing these results")
            lines.append("   with a doctor soon, even if you feel fine, so they can decide if further")
            lines.append("   tests are needed.")
        elif level == "MODERATE":
            lines.append("   Your combined risk is moderate. Consider a routine check-up and healthy")
            lines.append("   lifestyle habits, and monitor for any new symptoms.")
        else:
            lines.append("   Your combined risk is currently low. Continue routine health monitoring")
            lines.append("   and healthy lifestyle habits.")
        lines.append("")

        lines.append("=" * 72)
        lines.append(" IMPORTANT \u2014 PLEASE READ")
        lines.append("=" * 72)
        lines.append(" This is an AI-generated result from a research prototype, not a real medical")
        lines.append(" diagnosis. It should never replace advice from a qualified doctor. If you have")
        lines.append(" any symptoms or concerns, please consult a healthcare professional directly.")
        lines.append("=" * 72)

        return "\n".join(lines)
