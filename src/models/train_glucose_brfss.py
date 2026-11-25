import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import joblib, os

print("="*50)
print("Training Glucose Model — BRFSS Optimized")
print("="*50)

sp = "D:\\ReliableMultimodalAI\\data\\splits"
X_train = np.load(f"{sp}\\glucose_X_train.npy")
X_val   = np.load(f"{sp}\\glucose_X_val.npy")
X_test  = np.load(f"{sp}\\glucose_X_test.npy")
y_train = np.load(f"{sp}\\glucose_y_train.npy")
y_val   = np.load(f"{sp}\\glucose_y_val.npy")
y_test  = np.load(f"{sp}\\glucose_y_test.npy")

# Combine train+val for final training
X_all = np.vstack([X_train, X_val])
y_all = np.concatenate([y_train, y_val])

print(f"Train: {X_all.shape}, Test: {X_test.shape}")

model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.005,
    max_depth=10,
    num_leaves=127,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=0.05,
    random_state=42,
    verbose=-1
)

print("Training... this takes 2-3 minutes...")
model.fit(X_all, y_all)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Test Results ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred,
      target_names=['No Diabetes', 'Diabetes']))

os.makedirs("D:\\ReliableMultimodalAI\\models_saved", exist_ok=True)
joblib.dump(model, "D:\\ReliableMultimodalAI\\models_saved\\glucose_model.pkl")
print("Model saved!")