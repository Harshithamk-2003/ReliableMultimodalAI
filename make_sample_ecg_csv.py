import numpy as np

X_ecg_test = np.load("D:\\ReliableMultimodalAI\\data\\splits\\ecg5_X_test.npy")
y_ecg_test = np.load("D:\\ReliableMultimodalAI\\data\\splits\\ecg5_y_test.npy")

sample_idx = 0  # change this to try different patients (0 to len(X_ecg_test)-1)
sample = X_ecg_test[sample_idx]  # shape (12, 1000)

output_path = "D:\\ReliableMultimodalAI\\sample_ecg_for_upload.csv"
np.savetxt(output_path, sample, delimiter=",")

print(f"Saved: {output_path}")
print(f"Shape: {sample.shape}  (12 leads x 1000 samples)")
print(f"Ground truth label index: {y_ecg_test[sample_idx]}")