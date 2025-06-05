import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, r2_score,
    precision_recall_curve, roc_auc_score
)
from sklearn.feature_selection import VarianceThreshold
from lightgbm import LGBMRegressor
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

# -----------------------------
# Step 1: Load and prepare data
# -----------------------------
df_final = pd.read_csv(f'final_training_dataset.csv')

def sanitize_feature_names(df):
    clean_cols = [re.sub(r'[{}[\]":,\'`]', '_', col) for col in df.columns]
    df.columns = clean_cols
    return df
# -----------------------------
# Step 1: Prepare data
# -----------------------------
df_final = sanitize_feature_names(df_final)  # ensure safe column names
df_clin = df_final[['sample', 'pfi', 'pfi_time']].copy()
X_raw = df_final.drop(columns=['sample', 'early_progression_label', 'pfi', 'pfi_time'])

# -----------------------------
# Step 2: Train-test split
# -----------------------------
X_train_raw, X_test_raw, df_train_clin, df_test_clin = train_test_split(
    X_raw, df_clin, test_size=0.2, random_state=42, stratify=df_final['early_progression_label']
)

# -----------------------------
# Step 3: Variance filter on training only
# -----------------------------
selector = VarianceThreshold(threshold=0.01)
X_train_selected = selector.fit_transform(X_train_raw)
selected_features = X_train_raw.columns[selector.get_support()]

X_train_filtered = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train_raw.index)
X_test_filtered = pd.DataFrame(selector.transform(X_test_raw), columns=selected_features, index=X_test_raw.index)

# -----------------------------
# Step 4: StandardScaler fit on training only
# -----------------------------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_filtered), columns=selected_features, index=X_train_filtered.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_filtered), columns=selected_features, index=X_test_filtered.index)

# -----------------------------
# Step 5: Survival formatting
# -----------------------------
y_train_surv = Surv.from_dataframe("pfi", "pfi_time", df_train_clin)
y_test_surv = Surv.from_dataframe("pfi", "pfi_time", df_test_clin)

# -----------------------------
# Step 6: CoxPH model (training only)
# -----------------------------
cox_model = CoxPHSurvivalAnalysis(alpha=0.01)
cox_model.fit(X_train_scaled, y_train_surv)

train_risk_scores = cox_model.predict(X_train_scaled)
test_risk_scores = cox_model.predict(X_test_scaled)

# -----------------------------
# Step 7: scale_pos_weight calculation
# -----------------------------
threshold = np.percentile(train_risk_scores, 66)
risk_train_labels = (train_risk_scores >= threshold).astype(int)
pos_weight = (risk_train_labels == 0).sum() / (risk_train_labels == 1).sum()

# -----------------------------
# Step 8: SHAP-based feature selection on train
# -----------------------------
model_for_shap = LGBMRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, scale_pos_weight=pos_weight
)
model_for_shap.fit(X_train_scaled, train_risk_scores)

explainer = shap.TreeExplainer(model_for_shap, X_train_scaled)
shap_values = explainer.shap_values(X_train_scaled)
mean_shap = np.abs(shap_values).mean(axis=0)
shap_ranking = pd.Series(mean_shap, index=X_train_scaled.columns).sort_values(ascending=False)

top_shap_df = shap_ranking.head(250).reset_index()
top_shap_df.columns = ['feature', 'mean_abs_shap']
top_shap_df.to_csv("top_shap_features.csv", index=False)

top_features = top_shap_df['feature'].tolist()
X_train_final = X_train_scaled[top_features]
X_test_final = X_test_scaled[top_features]

# -----------------------------
# Step 9: Final LGBM regression model with pos_weight
# -----------------------------
final_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.9,
    scale_pos_weight=pos_weight,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42
)
final_model.fit(X_train_final, train_risk_scores)

# -----------------------------
# Step 10: Predict + F1-based Threshold Tuning
# -----------------------------
y_pred_risk = final_model.predict(X_test_final)
y_true_binary = (test_risk_scores >= threshold).astype(int)

precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred_risk)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]

# Apply best threshold
y_pred_binary = (y_pred_risk >= best_thresh).astype(int)

# -----------------------------
# Step 11: Evaluation
# -----------------------------
print(f"Best F1 threshold: {best_thresh:.2f}")
print(f"R² (regression): {r2_score(test_risk_scores, y_pred_risk):.3f}")
print(f"AUC: {roc_auc_score(y_true_binary, y_pred_risk):.3f}")
print(classification_report(y_true_binary, y_pred_binary, target_names=["Low Risk", "High Risk"]))

# Optional: Show strong thresholds
for p, r, t in zip(precision, recall, thresholds):
    if r >= 0.7 and p >= 0.5:
        print(f"  Threshold: {t:.2f} | Precision: {p:.2f} | Recall: {r:.2f}")
