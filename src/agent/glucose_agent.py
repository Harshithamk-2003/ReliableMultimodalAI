class GlucoseAgent:
    """Runs the glucose/lifestyle risk model."""
    name = "Glucose Agent"

    def __init__(self, glucose_model, glucose_scaler):
        self.glucose_model = glucose_model
        self.glucose_scaler = glucose_scaler

    def analyze(self, glucose_features_raw, data_quality=1.0):
        X = self.glucose_scaler.transform(glucose_features_raw.reshape(1, -1))
        prob = float(self.glucose_model.predict_proba(X)[0, 1])
        confidence = max(prob, 1 - prob)

        return {
            'name': 'Glucose',
            'risk_prob': prob,
            'confidence': confidence,
            'data_quality': data_quality,
            'agent': self.name
        }