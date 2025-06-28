### -- Processing CNV data --- ###
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# -----------------------------
# Config
# -----------------------------
CNV_PATH = 'cnv.csv'
CLINICAL_PATH = 'early_stage_clinical_labeled_dataset.csv'
OUTPUT_PATH = 'copy_number_variation.csv'

# -----------------------------
# Step 1: Load CNV and clinical data
# -----------------------------
df_cnv = pd.read_csv(CNV_PATH)
df_cnv.rename(columns={df_cnv.columns[0]: "gene"}, inplace=True)
df_cnv_t = df_cnv.set_index("gene").T.reset_index().rename(columns={"index": "sample"})
df_cnv_t.columns = ["sample"] + [f"cnv_{col}" for col in df_cnv_t.columns[1:]]

df_clinical = pd.read_csv(CLINICAL_PATH, usecols=['sample', 'early_progression_label'])
df_cnv_filtered = df_cnv_t[df_cnv_t['sample'].isin(df_clinical['sample'])].drop_duplicates(subset='sample')

# -----------------------------
# Step 2: CNV thresholding (-1, 0, 1)
# -----------------------------
cnv_vals = df_cnv_filtered.drop(columns='sample').astype(float).values
cnv_vals = np.where(cnv_vals <= -0.3, -1, np.where(cnv_vals >= 0.3, 1, 0))
df_cnv_filtered.iloc[:, 1:] = cnv_vals

# -----------------------------
# Step 3: Filter CNVs with variation in ≥ 10 samples
# -----------------------------
X_all = df_cnv_filtered.drop(columns='sample')
variable = X_all.loc[:, X_all.nunique() > 1]
common = (variable != 0).sum()
selected_cols = common[common >= 10].index.tolist()

df_cnv_final = df_cnv_filtered[['sample'] + selected_cols]

# -----------------------------
# Step 4: Merge, preprocess, and scale
# -----------------------------
df_merged = df_cnv_final.merge(df_clinical, on='sample')
X = df_merged.drop(columns=['sample', 'early_progression_label'])
y = df_merged['early_progression_label']

X_filtered = pd.DataFrame(VarianceThreshold(threshold=0.01).fit_transform(X), index=X.index)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# -----------------------------
# Step 5: Train fast ElasticNet-style classifier
# -----------------------------
clf = SGDClassifier(
    loss='log_loss',
    penalty='elasticnet',
    alpha=0.001,
    l1_ratio=0.5,
    max_iter=2000,
    tol=1e-3,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_scaled, y)

# -----------------------------
# Step 6: Select top 300 CNVs
# -----------------------------
coef_series = pd.Series(np.abs(clf.coef_.flatten()), index=X.columns)
top_cnv_feats = coef_series.sort_values(ascending=False).head(300).index.tolist()

# -----------------------------
# Step 7: Export top CNVs
# -----------------------------
df_output = df_cnv_t[['sample'] + list(top_cnv_feats)]
df_output.columns = ['sample'] + [f"{str(c).replace('.', '_').replace('-', '_')}" for c in df_output.columns[1:]]
df_output.to_csv(OUTPUT_PATH, index=False)
