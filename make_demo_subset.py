import numpy as np
import json

DATA_DIR = "D:\\ReliableMultimodalAI\\data\\splits\\"
SAMPLES_PER_CLASS = 40  # 40 x 5 classes = 200 real samples, ~9MB instead of 196MB

X_test = np.load(DATA_DIR + "ecg5_X_test.npy")
y_test = np.load(DATA_DIR + "ecg5_y_test.npy")

with open(DATA_DIR + "ecg5_class_map.json") as f:
    class_map = json.load(f)

selected_idx = []
for class_idx in sorted(set(y_test)):
    idx = np.where(y_test == class_idx)[0][:SAMPLES_PER_CLASS]
    selected_idx.extend(idx)

selected_idx = np.array(selected_idx)
X_demo = X_test[selected_idx]
y_demo = y_test[selected_idx]

np.save(DATA_DIR + "ecg5_X_demo.npy", X_demo)
np.save(DATA_DIR + "ecg5_y_demo.npy", y_demo)

print(f"Saved demo subset: {X_demo.shape[0]} samples -> ecg5_X_demo.npy / ecg5_y_demo.npy")
print(f"File size: {X_demo.nbytes / 1e6:.1f} MB")