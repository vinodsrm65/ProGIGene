import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, precision_recall_curve, roc_auc_score
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import shuffle
from copy import deepcopy

# -----------------------------
# Step 1: Extract RNA-only data
# -----------------------------
df_final = sanitize_feature_names(df_final)
X_rna_raw = df_final[[col for col in df_final.columns if col.startswith("rna_")]].copy()
y_rna = df_final["early_progression_label"]

# -----------------------------
# Step 2: Train-test split
# -----------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_rna_raw, y_rna, test_size=0.2, random_state=42, stratify=y_rna
)

# -----------------------------
# Step 3: Variance Threshold
# -----------------------------
selector = VarianceThreshold(threshold=0.01)
X_train_sel = pd.DataFrame(
    selector.fit_transform(X_train_raw),
    columns=X_train_raw.columns[selector.get_support()],
    index=X_train_raw.index
)
X_test_sel = pd.DataFrame(
    selector.transform(X_test_raw),
    columns=X_train_sel.columns,
    index=X_test_raw.index
)

# -----------------------------
# Step 4: Standard Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_sel),
    columns=X_train_sel.columns,
    index=X_train_sel.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_sel),
    columns=X_test_sel.columns,
    index=X_test_sel.index
)

# -----------------------------
# Step 5: ElasticNetCV model
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
log_cv.fit(X_train_scaled, y_train)

# -----------------------------
# Step 6: Feature selection
# -----------------------------
coef_series = pd.Series(np.abs(log_cv.coef_.flatten()), index=X_train_scaled.columns)
top_features = coef_series.sort_values(ascending=False).head(250).index.tolist()

X_train_final = X_train_scaled[top_features]
X_test_final = X_test_scaled[top_features]

# -----------------------------
# Step 7: Final ElasticNet model
# -----------------------------
log_cv_rna = LogisticRegressionCV(
    penalty='elasticnet',
    solver='saga',
    l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
    Cs=[0.01, 0.1, 1, 10],
    max_iter=5000,
    class_weight='balanced',
    cv=5,
    random_state=42
)
log_cv_rna.fit(X_train_final, y_train)

# -----------------------------
# Step 8: Prediction & F1 threshold
# -----------------------------
y_prob = log_cv_rna.predict_proba(X_test_final)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]


# -----------------------------
# Step 9: Permutation validation
# -----------------------------
def permutation_validation(model_template, X_train, y_train, X_test, y_test,
                           eval_func=roc_auc_score, n_permutations=100, verbose=True):
    model_real = deepcopy(model_template)
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )
    model_real.fit(X_tr_sub, y_tr_sub)
    y_prob_real = model_real.predict_proba(X_test)[:, 1]
    real_score = eval_func(y_test, y_prob_real)

    permutation_scores = []
    for i in range(n_permutations):
        y_perm = shuffle(y_train, random_state=i)
        model_perm = deepcopy(model_template)
        X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
            X_train, y_perm, test_size=0.1, stratify=y_perm, random_state=i
        )
        model_perm.fit(X_tr_sub, y_tr_sub)
        y_prob_perm = model_perm.predict_proba(X_test)[:, 1]
        score = eval_func(y_test, y_prob_perm)
        permutation_scores.append(score)
        if verbose and (i + 1) % 10 == 0:
            print(f"Done {i+1}/{n_permutations}")

    p_val = (np.sum(np.array(permutation_scores) >= real_score) + 1) / (n_permutations + 1)
    print(f"Permutation AUC Mean: {np.mean(permutation_scores):.4f}")
    print(f"Real AUC: {real_score:.4f}")
    print(f"Empirical p-value: {p_val:.4f}")
    return real_score, permutation_scores, p_val

# Run permutation test
real_score, perm_scores, p_val = permutation_validation(
    log_cv_rna, X_train_final, y_train, X_test_final, y_test, n_permutations=100
)
