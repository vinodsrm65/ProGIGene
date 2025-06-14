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

df_final = pd.read_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/elastic/final_training_dataset.csv')

def sanitize_feature_names(df):
    clean_cols = [re.sub(r'[{}[\]":,\'`]', '_', col) for col in df.columns]
    df.columns = clean_cols
    return df

# ---------
# Full Model train --
# ---------
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
risk_train_labels = (train_risk_scores >= threshold).astype(int)
risk_test_labels = (test_risk_scores >= threshold).astype(int)

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
coef_series = pd.Series(np.abs(log_cv.coef_.flatten()), index=X_train_scaled.columns)
top_features = coef_series.sort_values(ascending=False).head(250).index.tolist()

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
y_prob = final_model.predict_proba(X_test_final)[:, 1]
precision, recall, thresholds = precision_recall_curve(risk_test_labels, y_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]

# -----------------------------
# Step 11: Evaluation
# -----------------------------
y_pred = (y_prob >= best_thresh).astype(int)
print(f"Best F1 threshold: {best_thresh:.2f}")
print(f"AUC: {roc_auc_score(risk_test_labels, y_prob):.3f}")
print(classification_report(risk_test_labels, y_pred, target_names=["Low Risk", "High Risk"]))

# Optional: Print useful thresholds
for p, r, t in zip(precision, recall, thresholds):
    if r >= 0.7 and p >= 0.5:
        print(f"  Threshold: {t:.2f} | Precision: {p:.2f} | Recall: {r:.2f}")

# ---------
# Top Feature Extracton 
# ---------

top_feature_df = coef_series.sort_values(ascending=False).head(250).reset_index()
top_feature_df.columns = ['feature', 'absolute_coefficient']
filtered_top_df = top_feature_df[~top_feature_df['feature'].str.startswith('tumor_stage')]
print(filtered_top_df.head(20))  # show top 20 for review


# ----------
# Gene Only Model Train 
# ----------

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

# Step 1: Extract RNA-only features
X_rna = df_final[[col for col in df_final.columns if col.startswith("rna_")]].copy()

# Step 2: Target variable (binary: early progression label)
y = df_final['early_progression_label']

# Step 3: Handle missing values
X_rna = X_rna.fillna(X_rna.mean())

# Step 4: Standardize features
scaler = StandardScaler()
X_rna_scaled = pd.DataFrame(
    scaler.fit_transform(X_rna),
    columns=X_rna.columns,
    index=X_rna.index
)

# Step 5: Train ElasticNet-penalized logistic regression
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
log_cv_rna.fit(X_rna_scaled, y)

# ✅ You now have a trained RNA-only model: log_cv_rna
print("✅ RNA-only ProGIGene model trained successfully.")


# ----------
# Geo Data Validation 
# ----------

import GEOparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import re

# Step 1: Load expression data from GSE14333
print("🔄 Downloading GSE87211...")
gse = GEOparse.get_GEO(geo="GSE14333", destdir="./geo_data")
geo_expr_raw = gse.pivot_samples("VALUE")  # Probes x Samples

# Step 2: Load GPL570 for probe → gene mapping
gpl = GEOparse.get_GEO(geo="GPL570", destdir="./geo_data")
platform_table = gpl.table[['ID', 'Gene Symbol']].dropna()
platform_table.columns = ['probe_id', 'gene_symbol']

# Step 3: Map probes to gene symbols
geo_expr = geo_expr_raw.copy()
geo_expr['probe_id'] = geo_expr.index
geo_expr = geo_expr.merge(platform_table, on='probe_id', how='left')
geo_expr = geo_expr.dropna(subset=['gene_symbol'])
geo_expr = geo_expr.drop(columns='probe_id').groupby('gene_symbol').mean().T
geo_expr.columns = geo_expr.columns.str.upper()

# Step 4: Prepare RNA feature set from trained model
rna_features = [f for f in log_cv_rna.feature_names_in_ if f.startswith("rna_")]
rna_gene_symbols = [f.replace("rna_", "").upper() for f in rna_features]
common_genes = geo_expr.columns.intersection(rna_gene_symbols)

X_geo = geo_expr[common_genes].copy()
X_geo.columns = ['rna_' + g for g in X_geo.columns]

# Fill missing model features with 0 (safe default after z-scoring)
for f in rna_features:
    if f not in X_geo.columns:
        X_geo[f] = 0
X_geo = X_geo[rna_features]  # ensure correct order

# Step 5: Standardize
X_geo = X_geo.fillna(X_geo.mean())
scaler = StandardScaler()
X_geo_std = pd.DataFrame(
    scaler.fit_transform(X_geo),
    columns=X_geo.columns,
    index=X_geo.index
)

# Step 6: Predict risk scores
risk_scores = log_cv_rna.decision_function(X_geo_std)

# Step 7: Extract DFS time + event from metadata
clinical_data = []
for gsm, sample in gse.gsms.items():
    metadata_lines = sample.metadata.get("characteristics_ch1", [])
    entry = {}
    for line in metadata_lines:
        parts = [kv.strip() for kv in line.split(";")]
        for part in parts:
            if "dfs_time" in part.lower():
                try:
                    dfs_time_months = float(part.split(":")[1].strip())
                    entry["DFS_time"] = int(dfs_time_months * 30.44)  # months → days
                except:
                    continue
            if "dfs_cens" in part.lower():
                try:
                    cens = int(part.split(":")[1].strip())
                    entry["DFS_event"] = 1 - cens  # Convert: 1=censored → 0 event, 0=event → 1
                except:
                    continue
    if "DFS_time" in entry and "DFS_event" in entry:
        entry["sample_id"] = gsm
        clinical_data.append(entry)

clinical_df = pd.DataFrame(clinical_data).set_index("sample_id")
print(f"✅ Extracted DFS data for {len(clinical_df)} samples.")

# Step 8: Merge risk scores with clinical
pred_df = pd.DataFrame({'risk_score': risk_scores}, index=X_geo_std.index)
merged = clinical_df.join(pred_df)
merged = merged.dropna(subset=['risk_score', 'DFS_time', 'DFS_event'])
print(f"✅ Merged data: {merged.shape[0]} samples")

# Step 9: Stratify risk groups by 66th percentile
threshold = np.percentile(merged['risk_score'], 66)
merged['risk_group'] = np.where(merged['risk_score'] >= threshold, 'High Risk', 'Low Risk')

# Step 10: Plot Kaplan–Meier Curve
kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 6))

for group in ['Low Risk', 'High Risk']:
    mask = merged['risk_group'] == group
    kmf.fit(merged.loc[mask, 'DFS_time'], merged.loc[mask, 'DFS_event'], label=group)
    kmf.plot_survival_function(ci_show=False)

plt.title("ProGIGene RNA Validation on GSE14333\n(Disease-Free Survival)")
plt.xlabel("Time (days)")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.savefig("ProGIGene_GEO_Validaton.png", dpi=300)
plt.show()

# Step 11: Compute AUC
auc = roc_auc_score(merged['DFS_event'], merged['risk_score'])
print(f"🎯 AUC (DFS prediction): {auc:.3f}")



