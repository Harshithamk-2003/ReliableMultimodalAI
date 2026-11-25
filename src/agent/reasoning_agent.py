import numpy as np

def assess_data_quality(modality_name, confidence, raw_features=None):
    """Simple data quality heuristic per modality."""
    issues = []
    quality = 1.0

    if confidence < 0.55:
        issues.append(f"{modality_name} prediction confidence is low ({confidence:.2f})")
        quality *= 0.7

    return quality, issues

def generate_explanation(modality_results, fusion_result):
    """
    modality_results: list of dicts with 'name', 'risk_prob', 'confidence', 'reliability'
    fusion_result: output from fuse_modalities()
    Returns a structured, evidence-grounded explanation.
    """
    lines = []
    lines.append("="*60)
    lines.append("AGENTIC AI ANALYSIS REPORT")
    lines.append("="*60)

    # 1. Data availability check
    available = [m['name'] for m in modality_results]
    lines.append(f"\nModalities analyzed: {', '.join(available)}")
    if len(available) < 3:
        missing = set(['ECG', 'BP', 'Glucose']) - set(available)
        lines.append(f"Note: {', '.join(missing)} data not provided — assessment is based on available modalities only.")

    # 2. Per-modality findings with reliability
    lines.append("\n--- Per-Modality Findings ---")
    for m in modality_results:
        risk_level = "HIGH" if m['risk_prob'] >= 0.6 else ("MODERATE" if m['risk_prob'] >= 0.4 else "LOW")
        lines.append(f"{m['name']:10s}: Risk={risk_level} (score={m['risk_prob']:.2f}), "
                     f"Confidence={m['confidence']:.2f}, Reliability={m['reliability']:.2f}")
        if m['confidence'] < 0.6:
            lines.append(f"  WARNING: {m['name']} prediction has low confidence — interpret cautiously.")

    # 3. Fusion outcome
    lines.append("\n--- Fused Assessment ---")
    fused_risk = fusion_result['fused_risk_score']
    uncertainty = fusion_result['uncertainty_mass']
    conflict = fusion_result['conflict']

    risk_label = "HIGH RISK" if fused_risk >= 0.6 else ("MODERATE RISK" if fused_risk >= 0.4 else "LOW RISK")
    lines.append(f"Combined Risk Assessment: {risk_label} (fused score = {fused_risk:.2f})")
    lines.append(f"Uncertainty in assessment: {uncertainty:.2f}")
    lines.append(f"Inter-modality conflict: {conflict:.2f}")

    # 4. Reliability/conflict flagging — core novelty
    if conflict > 0.3:
        lines.append("\nCAUTION: Modalities show significant disagreement.")
        lines.append("This may indicate data quality issues, measurement error, or a genuinely")
        lines.append("complex clinical picture requiring professional evaluation.")
    elif not fusion_result['is_reliable']:
        lines.append("\nNOTE: Fusion confidence is reduced due to moderate inter-modality conflict.")
    else:
        lines.append("\nModalities show consistent evidence, supporting the fused assessment.")

    # 5. Mandatory disclaimer
    lines.append("\n" + "="*60)
    lines.append("IMPORTANT: This is an AI-generated research prototype output.")
    lines.append("It is NOT a medical diagnosis and must NOT be used as a substitute")
    lines.append("for professional clinical evaluation. Please consult a qualified")
    lines.append("healthcare provider for any health concerns.")
    lines.append("="*60)

    return "\n".join(lines)


if __name__ == "__main__":
    # Example test
    example_modalities = [
        {'name': 'ECG',     'risk_prob': 0.72, 'confidence': 0.81, 'reliability': 0.79},
        {'name': 'BP',      'risk_prob': 0.65, 'confidence': 0.90, 'reliability': 0.88},
        {'name': 'Glucose', 'risk_prob': 0.40, 'confidence': 0.75, 'reliability': 0.70},
    ]
    example_fusion = {
        'fused_risk_score': 0.68,
        'uncertainty_mass': 0.12,
        'conflict': 0.18,
        'is_reliable': True
    }
    report = generate_explanation(example_modalities, example_fusion)
    print(report)