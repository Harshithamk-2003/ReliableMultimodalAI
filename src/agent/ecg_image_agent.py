import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

IMG_SIZE = 224
IMG_MEAN = 0.5
IMG_STD = 0.5
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']  # must match ecg5_class_map order (index 0..4)


class ECGImageAgent:
    name = "ECG Image Agent (comparison model)"

    def __init__(self, model_path="D:\\ReliableMultimodalAI\\models_saved\\ecg_image_model.pth", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def analyze(self, ecg_image_224x224, data_quality=1.0):
        """
        ecg_image_224x224: numpy array, shape (224, 224), dtype uint8 or float, values 0-255 or 0-1.
        Returns a dict in the same shape as ECGAgent.analyze() for easy side-by-side comparison.
        """
        img = ecg_image_224x224.astype(np.float32)
        if img.max() > 1.5:  # assume 0-255 range
            img = img / 255.0
        img = (img - IMG_MEAN) / IMG_STD
        img_3ch = np.stack([img, img, img], axis=0)  # (3, 224, 224)
        tensor = torch.from_numpy(img_3ch).float().unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        norm_idx = CLASS_NAMES.index('NORM')
        risk_prob = float(1 - probs[norm_idx])

        class_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        return {
            'name': 'ECG (Image)',
            'agent': self.name,
            'risk_prob': risk_prob,
            'confidence': confidence,
            'predicted_class': predicted_class,
            'class_probs': class_probs,
            'data_quality': data_quality,
        }