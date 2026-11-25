import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import joblib, os

print("="*50)
print("Training BP Model — Cleveland Only (Final, Consistent Scaler)")
print("="*50)

df = pd.read_csv("D:\\ReliableMultimodalAI\\data\\raw\\uci_heart.csv")
df['ca'] = df['ca'].fillna(df['ca'].median())
df['thal'] = df['thal'].fillna(df['thal'].median())
df['target'] = (df['target'] > 0).astype(int)

feature_cols = [c for c in df.columns if c != 'target']
X = df[feature_cols].values
y = df['target'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    random_state=42, class_weight='balanced', verbose=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['No Disease','Disease']))

os.makedirs("D:\\ReliableMultimodalAI\\models_saved", exist_ok=True)
joblib.dump(model, "D:\\ReliableMultimodalAI\\models_saved\\bp_model.pkl")
joblib.dump(scaler, "D:\\ReliableMultimodalAI\\models_saved\\bp_scaler.pkl")
print("\nBoth model AND matching scaler saved — consistent pair.")
print(f"Scaler expects {scaler.n_features_in_} features (should be 13).")