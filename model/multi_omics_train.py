import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import shap

# -----------------------------
# Step 1: Load and prepare data
# -----------------------------
df_final = pd.read_csv('final_training_dataset.csv')

def sanitize_feature_names(df):
    clean_cols = [re.sub(r'[{}[\]":,\'`]', '_', col) for col in df.columns]
    df.columns = clean_cols
    return df

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    classification_report, precision_recall_curve,
    roc_auc_score
)
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis

def permutation_validation(model_template, X_train, y_train, X_test, y_test,
                           eval_func=roc_auc_score, n_permutations=100, verbose=True):
    # ---------------------------------------------
    # 1. Fit real model with a validation set
    # ---------------------------------------------
    model_real = deepcopy(model_template)
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )
    model_real.fit(X_tr_sub, y_tr_sub, eval_set=[(X_val_sub, y_val_sub)], verbose=False)
    y_prob_real = model_real.predict_proba(X_test)[:, 1]
    real_score = eval_func(y_test, y_prob_real)

    if verbose:
        print(f"🎯 Real Test Score: {real_score:.4f}")
        print(f"🔁 Running {n_permutations} permutations...")

    # ---------------------------------------------
    # 2. Permutation loop with eval_set
    # ---------------------------------------------
    permutation_scores = []
    for i in range(n_permutations):
        y_perm = shuffle(y_train, random_state=i)
        model_perm = deepcopy(model_template)
        X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
            X_train, y_perm, test_size=0.1, stratify=y_perm, random_state=i
        )
        model_perm.fit(X_tr_sub, y_tr_sub, eval_set=[(X_val_sub, y_val_sub)], verbose=False)
        y_prob_perm = model_perm.predict_proba(X_test)[:, 1]
        score = eval_func(y_test, y_prob_perm)
        permutation_scores.append(score)

        if verbose and (i+1) % 10 == 0:
            print(f"  ✅ Done {i+1}/{n_permutations}")
            
# -----------------------------
# Step 1: Prepare data
# -----------------------------
df_final = sanitize_feature_names(df_final)
df_clin = df_final[['sample', 'pfi', 'pfi_time']].copy()
X_raw = df_final.drop(columns=['sample', 'early_progression_label', 'pfi', 'pfi_time'])

# -----------------------------
# Step 2: Train-test split
# -----------------------------
X_train_raw, X_test_raw, df_train_clin, df_test_clin = train_test_split(
    X_raw, df_clin, test_size=0.2, random_state=42,
    stratify=df_final['early_progression_label']
)

# -----------------------------
# Step 3: Variance threshold
# -----------------------------
selector = VarianceThreshold(threshold=0.01)
X_train_selected = selector.fit_transform(X_train_raw)
selected_features = X_train_raw.columns[selector.get_support()]
X_train_filtered = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train_raw.index)
X_test_filtered = pd.DataFrame(selector.transform(X_test_raw), columns=selected_features, index=X_test_raw.index)

# -----------------------------
# Step 4: StandardScaler
# -----------------------------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_filtered), columns=selected_features, index=X_train_filtered.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_filtered), columns=selected_features, index=X_test_filtered.index)

# -----------------------------
# Step 5: CoxPH survival modeling
# -----------------------------
y_train_surv = Surv.from_dataframe("pfi", "pfi_time", df_train_clin)
y_test_surv = Surv.from_dataframe("pfi", "pfi_time", df_test_clin)

cox_model = CoxPHSurvivalAnalysis(alpha=0.01)
cox_model.fit(X_train_scaled, y_train_surv)
train_risk_scores = cox_model.predict(X_train_scaled)
test_risk_scores = cox_model.predict(X_test_scaled)

# -----------------------------
# Step 6: Create binary label for classification
# -----------------------------
threshold = np.percentile(train_risk_scores, 66)

# Align indices explicitly for safety
risk_train_labels = pd.Series((train_risk_scores >= threshold).astype(int), index=X_train_scaled.index)
risk_test_labels = pd.Series((test_risk_scores >= threshold).astype(int), index=X_test_scaled.index)


# -----------------------------
# Step 7: LogisticRegressionCV (ElasticNet)
# -----------------------------
log_cv = LogisticRegressionCV(
    penalty='elasticnet',
    solver='saga',
    l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
    Cs=[0.01, 0.1, 1, 10],
    max_iter=5000,
    class_weight='balanced',
    cv=5,
    random_state=42
)
log_cv.fit(X_train_scaled, risk_train_labels)

# -----------------------------
# Step 8: Feature selection by ElasticNet coefficients
# -----------------------------
coef_series_main = pd.Series(np.abs(log_cv.coef_.flatten()), index=X_train_scaled.columns)
top_features_main = coef_series.sort_values(ascending=False).head(250).index.tolist()

X_train_final = X_train_scaled[top_features]
X_test_final = X_test_scaled[top_features]

# -----------------------------
# Step 9: Final Logistic Model (same ElasticNetCV)
# -----------------------------
final_model = LogisticRegressionCV(
    penalty='elasticnet',
    solver='saga',
    l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
    Cs=[0.01, 0.1, 1, 10],
    max_iter=5000,
    class_weight='balanced',
    cv=5,
    random_state=42
)
final_model.fit(X_train_final, risk_train_labels)

# -----------------------------
# Step 10: Prediction and threshold tuning
# -----------------------------
# Make sure X_test_final is subset using the same index
X_test_final_aligned = X_test_final.loc[risk_test_labels.index]

y_prob = final_model.predict_proba(X_test_final_aligned)[:, 1]

assert len(y_prob) == len(risk_test_labels)

precision, recall, thresholds = precision_recall_curve(risk_test_labels, y_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]


for p, r, t in zip(precision, recall, thresholds):
    if r >= 0.7 and p >= 0.5:
        print(f"  Threshold: {t:.2f} | Precision: {p:.2f} | Recall: {r:.2f}")

real_score, perm_scores, p_val = permutation_validation(
    model, X_train_sel, y_train, X_test_sel, y_test, n_permutations=100
)
