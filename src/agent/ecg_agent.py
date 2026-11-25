import numpy as np
import torch

class ECGAgent:
    """Runs the 12-lead ECG ensemble model and reports risk + confidence."""
    name = "ECG Analysis Agent"

    def __init__(self, ecg_models, ecg_class_map, device):
        self.ecg_models = ecg_models
        self.ecg_class_map = ecg_class_map
        self.device = device
        self.inv_map = {v: k for k, v in ecg_class_map.items()}

    def analyze(self, ecg_signal_12x1000, data_quality=1.0):
        x = torch.FloatTensor(ecg_signal_12x1000).unsqueeze(0).to(self.device)
        probs_sum = torch.zeros(1, 5).to(self.device)
        with torch.no_grad():
            for m in self.ecg_models:
                probs_sum += torch.softmax(m(x), dim=1)
        probs = (probs_sum / len(self.ecg_models)).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        risk_prob = float(1 - probs[self.ecg_class_map['NORM']])

        return {
            'name': 'ECG',
            'risk_prob': risk_prob,
            'confidence': confidence,
            'data_quality': data_quality,
            'predicted_class': self.inv_map[pred_class],
            'class_probs': {k: float(probs[v]) for k, v in self.ecg_class_map.items()},
            'agent': self.name
        }