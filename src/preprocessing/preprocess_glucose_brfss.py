import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os, subprocess, sys

print("="*50)
print("Preprocessing BRFSS — Balanced Version")
print("="*50)

# Use the balanced 50/50 split file — already balanced
df = pd.read_csv("D:\\ReliableMultimodalAI\\data\\raw\\brfss\\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
print(f"Shape: {df.shape}")
print(f"Class balance:\n{df['Diabetes_binary'].value_counts()}")

X = df.drop('Diabetes_binary', axis=1).values
y = df['Diabetes_binary'].values.astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

sp = "D:\\ReliableMultimodalAI\\data\\splits"
os.makedirs(sp, exist_ok=True)
np.save(f"{sp}\\glucose_X_train.npy", X_train)
np.save(f"{sp}\\glucose_X_val.npy",   X_val)
np.save(f"{sp}\\glucose_X_test.npy",  X_test)
np.save(f"{sp}\\glucose_y_train.npy", y_train)
np.save(f"{sp}\\glucose_y_val.npy",   y_val)
np.save(f"{sp}\\glucose_y_test.npy",  y_test)

import joblib
joblib.dump(scaler, "D:\\ReliableMultimodalAI\\models_saved\\glucose_scaler.pkl")

print(f"\nTrain: {X_train.shape}, class balance: 0={(y_train==0).sum()}, 1={(y_train==1).sum()}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")
print("Done!")