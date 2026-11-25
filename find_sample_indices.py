import numpy as np
import json

DATA_DIR = "D:\\ReliableMultimodalAI\\data\\"
y_test = np.load(DATA_DIR + "splits\\ecg5_y_test.npy")

with open(DATA_DIR + "splits\\ecg5_class_map.json") as f:
    class_map = json.load(f)

inv_class_map = {v: k for k, v in class_map.items()}

print("Example test-set sample indices for each class (first 5 found):\n")
for class_idx in sorted(inv_class_map.keys()):
    class_name = inv_class_map[class_idx]
    indices = np.where(y_test == class_idx)[0][:5]
    print(f"{class_name:6s} (label {class_idx}): indices {list(indices)}")