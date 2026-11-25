import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

print("="*50)
print("Preprocessing UCI Heart Disease Dataset")
print("="*50)

# Load
df = pd.read_csv("D:\\ReliableMultimodalAI\\data\\raw\\uci_heart.csv")
print(f"Original shape: {df.shape}")

# Fix missing values in ca and thal
df['ca'] = df['ca'].fillna(df['ca'].median())
df['thal'] = df['thal'].fillna(df['thal'].median())
print(f"Fixed missing values in ca and thal")

# Convert target to binary (0 = no disease, 1 = disease)
df['target'] = (df['target'] > 0).astype(int)
print(f"Target distribution: {df['target'].value_counts().to_dict()}")

# Features and labels
X = df.drop('target', axis=1).values
y = df['target'].values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Save
np.save("D:\\ReliableMultimodalAI\\data\\splits\\bp_X_train.npy", X_train)
np.save("D:\\ReliableMultimodalAI\\data\\splits\\bp_X_val.npy", X_val)
np.save("D:\\ReliableMultimodalAI\\data\\splits\\bp_X_test.npy", X_test)
np.save("D:\\ReliableMultimodalAI\\data\\splits\\bp_y_train.npy", y_train)
np.save("D:\\ReliableMultimodalAI\\data\\splits\\bp_y_val.npy", y_val)
np.save("D:\\ReliableMultimodalAI\\data\\splits\\bp_y_test.npy", y_test)

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Class balance in train - 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")
print("BP preprocessing complete!")