# train_best_fraud_model.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc
)

import xgboost as xgb
import lightgbm as lgb
import kagglehub

# -----------------------------
# 0. Repro + Output dirs
# -----------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# 1. Load dataset from Kaggle
# -----------------------------
print("Loading creditcardfraud dataset from Kaggle...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_path = os.path.join(path, "creditcard.csv")
df = pd.read_csv(csv_path)
print(f"Dataset loaded successfully! Shape: {df.shape}")

# -----------------------------
# 2. Feature Engineering
# -----------------------------
# Keep PCA features (V1..V28) — they contain most of the signal
pca_features = [f"V{i}" for i in range(1, 29)]

# Log transform for amount (robust to heavy tails)
df["log_amount"] = np.log1p(df["Amount"].astype(float))

# Build datetime features from 'Time' (seconds since first transaction)
base_date = datetime(2023, 1, 1)
df["datetime"] = df["Time"].astype(float).apply(lambda s: base_date + timedelta(seconds=s))
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Cyclical encodings (preserve circularity)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

# Simple short-term rolling stats on Amount (sorted by time)
df = df.sort_values("datetime").reset_index(drop=True)
df["amount_roll_mean_10"] = df["Amount"].rolling(10, min_periods=1).mean()
df["amount_roll_std_10"] = df["Amount"].rolling(10, min_periods=1).std().fillna(0)

# Final feature set (no scaling needed for tree models)
features = (
    pca_features
    + [
        "log_amount",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "amount_roll_mean_10",
        "amount_roll_std_10",
    ]
)
target = "Class"

X = df[features].copy()
y = df[target].astype(int).copy()

# -----------------------------
# 3. Train/Val/Test Split (Stratified)
# -----------------------------
# First: hold out a test set
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
# Then: from trainval, create a validation set for early stopping & model selection
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.20, random_state=RANDOM_STATE, stratify=y_trainval
)

print(
    f"Class balance (train): {np.bincount(y_train)} | "
    f"(val): {np.bincount(y_val)} | "
    f"(test): {np.bincount(y_test)}"
)

# -----------------------------
# 4. Models with Imbalance Handling
# -----------------------------
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = max(1, int(neg / max(1, pos)))
print(f"Computed scale_pos_weight for XGBoost: {scale_pos_weight}")

xgb_model = xgb.XGBClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    reg_lambda=1.0,
    reg_alpha=0.0,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    objective="binary:logistic",
    tree_method="hist",
    eval_metric="aucpr",  # optimize for PR-AUC
    n_jobs=-1,
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=-1,
    num_leaves=64,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_samples=20,
    reg_lambda=0.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    # Imbalance handling
    is_unbalance=True,
    objective="binary",
    metric="auc_pr",
)

# -----------------------------
# 5. Fit with Early Stopping on Val
# -----------------------------
print("Training XGBoost...")
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
# xgb_best_ntree is not available without early stopping
# print(f"XGBoost best_iteration: {xgb_best_ntree}") # Commented out

print("Training LightGBM...")
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc_pr"
    # Removed callbacks for early stopping
)
lgb_best_iter = lgb_model.best_iteration_ if hasattr(lgb_model, 'best_iteration_') else lgb_model.n_estimators
print(f"LightGBM best_iteration: {lgb_best_iter}")


# -----------------------------
# 6. Model Selection by Val PR-AUC
# -----------------------------
def pr_auc_score(y_true, y_prob):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return auc(rec, prec)

xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
lgb_val_prob = lgb_model.predict_proba(X_val)[:, 1]

xgb_val_pr_auc = pr_auc_score(y_val, xgb_val_prob)
lgb_val_pr_auc = pr_auc_score(y_val, lgb_val_prob)

print(f"Validation PR-AUC | XGB: {xgb_val_pr_auc:.4f} | LGBM: {lgb_val_pr_auc:.4f}")

if lgb_val_pr_auc >= xgb_val_pr_auc:
    best_name = "lightgbm"
    best_model = lgb_model
else:
    best_name = "xgboost"
    best_model = xgb_model

print(f"Selected best model: {best_name}")

# -----------------------------
# 7. Test Evaluation + Threshold Tuning
# -----------------------------
y_prob_test = best_model.predict_proba(X_test)[:, 1]

# PR & ROC
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_test)
pr_auc = auc(recall, precision)
roc_auc = roc_auc_score(y_test, y_prob_test)

# F1-optimal threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
f1_idx = np.nanargmax(f1_scores)
# Note: precision_recall_curve returns an extra point for precision/recall without a threshold.
# Use min to avoid index error when f1_idx points to the last PR point.
f1_thr = thresholds[min(f1_idx, len(thresholds) - 1)]

# F2-optimal threshold (recall-weighted)
beta = 2
f2_scores = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-12)
f2_idx = np.nanargmax(f2_scores)
f2_thr = thresholds[min(f2_idx, len(thresholds) - 1)]

def evaluate_at_threshold(thr, name):
    y_pred = (y_prob_test >= thr).astype(int)
    print(f"\n=== {name} @ threshold={thr:.4f} ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

evaluate_at_threshold(f1_thr, "F1-Optimal")
evaluate_at_threshold(f2_thr, "F2-Optimal (Recall-Heavy)")

print(f"\nOverall Test ROC-AUC: {roc_auc:.4f}")
print(f"Overall Test PR-AUC:  {pr_auc:.4f}")

# -----------------------------
# 8. Plots (PR, ROC, Importances)
# -----------------------------
# Precision-Recall
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"PR Curve (AUC={pr_auc:.3f})")
plt.scatter(recall[f1_idx], precision[f1_idx], label="F1-opt", s=60)
plt.scatter(recall[f2_idx], precision[f2_idx], label="F2-opt", s=60)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve ({best_name})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "precision_recall_curve.png"))
plt.close()

# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve ({best_name})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "roc_curve.png"))
plt.close()

# Feature importance
def plot_feature_importance(model, feature_names, path):
    if isinstance(model, xgb.XGBClassifier):
        importances = model.get_booster().get_score(importance_type="gain")
        # Map to provided feature order; missing -> 0
        imp_series = pd.Series({fn: importances.get(fn, 0.0) for fn in feature_names})
    elif isinstance(model, lgb.LGBMClassifier):
        imp_series = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=feature_names)
    else:
        return
    imp_series = imp_series.sort_values(ascending=False).head(20)
    plt.figure(figsize=(8, 6))
    imp_series.iloc[::-1].plot(kind="barh")
    plt.title(f"Top 20 Feature Importances ({best_name}, gain)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Only plot feature importance if the selected model has the method
if hasattr(best_model, 'get_booster') or hasattr(best_model, 'booster_'):
    plot_feature_importance(best_model, features, os.path.join(MODEL_DIR, "feature_importance.png"))
else:
    print(f"Feature importance plotting is not available for model type: {best_name}")


# -----------------------------
# 9. Persist artifacts
# -----------------------------
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_fraud_model.pkl"))
joblib.dump(features, os.path.join(MODEL_DIR, "features.pkl"))
joblib.dump(
    {
        "model_type": best_name,
        "f1_threshold": float(f1_thr),
        "f2_threshold": float(f2_thr),
        "val_pr_auc": float(max(xgb_val_pr_auc, lgb_val_pr_auc)),
        "test_pr_auc": float(pr_auc),
        "test_roc_auc": float(roc_auc),
        "random_state": RANDOM_STATE,
    },
    os.path.join(MODEL_DIR, "metadata.pkl"),
)

print("\nTraining complete!")
print(f"Best model: {best_name}")
print(f"Saved to: {MODEL_DIR}/best_fraud_model.pkl")
print("Also saved: features.pkl, metadata.pkl, precision_recall_curve.png, roc_curve.png, feature_importance.png")