class BPAgent:
    """Runs the blood pressure risk model."""
    name = "Blood Pressure Agent"

    def __init__(self, bp_model, bp_scaler):
        self.bp_model = bp_model
        self.bp_scaler = bp_scaler

    def analyze(self, bp_features_raw, data_quality=1.0):
        X = self.bp_scaler.transform(bp_features_raw.reshape(1, -1))
        prob = float(self.bp_model.predict_proba(X)[0, 1])
        confidence = max(prob, 1 - prob)

        return {
            'name': 'BP',
            'risk_prob': prob,
            'confidence': confidence,
            'data_quality': data_quality,
            'agent': self.name
        }