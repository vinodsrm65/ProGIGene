import GEOparse
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# ----------------------------
# Step 1: Download and load expression data from GSE62452
# ----------------------------
print("🔄 Downloading GSE62452...")
gse = GEOparse.get_GEO(geo="GSE62452", destdir="./geo_data")
geo_expr_raw = gse.pivot_samples("VALUE")  # Probes x Samples

# ----------------------------
# Step 2: Load platform annotation for GPL6244 and parse gene symbols
# ----------------------------
gpl = GEOparse.get_GEO(geo="GPL6244", destdir="./geo_data")

# Function to extract clean gene symbol from gene_assignment string
def extract_gene_symbol(x):
    if not isinstance(x, str):
        return np.nan
    # Split by '//' or '///'
    parts = re.split(r'//+|\s*///\s*', x)
    if len(parts) == 0:
        return np.nan
    # Return first part that looks like a gene symbol (letters/numbers/hyphen only)
    for part in parts:
        part = part.strip()
        if part and re.match(r'^[A-Za-z0-9\-]+$', part):
            return part.upper()
    return np.nan

gpl.table['gene_symbol'] = gpl.table['gene_assignment'].apply(extract_gene_symbol)

platform_table = gpl.table[['ID', 'gene_symbol']].dropna()
platform_table.columns = ['probe_id', 'gene_symbol']

# ----------------------------
# Step 3: Map probes → gene symbols and collapse duplicates
# ----------------------------
geo_expr = geo_expr_raw.copy()
geo_expr['probe_id'] = geo_expr.index
geo_expr = geo_expr.merge(platform_table, on='probe_id', how='left')
geo_expr = geo_expr.dropna(subset=['gene_symbol'])
geo_expr = geo_expr.drop(columns='probe_id').groupby('gene_symbol').mean().T
geo_expr.columns = geo_expr.columns.str.upper()

print("geo_expr shape:", geo_expr.shape)
print("geo_expr columns sample:", geo_expr.columns[:10])

# ----------------------------
# Step 4: Match features from your trained model (log_cv_rna)
# ----------------------------
# Example placeholder: replace with your actual trained model object
# log_cv_rna = <your trained model>

rna_features = [f for f in log_cv_rna.feature_names_in_ if f.startswith("rna_")]
rna_gene_symbols = [f.replace("rna_", "").upper() for f in rna_features]
common_genes = geo_expr.columns.intersection(rna_gene_symbols)

X_geo = geo_expr[common_genes].copy()
X_geo.columns = ['rna_' + g for g in X_geo.columns]

# Fill missing model features with 0 (safe post-zscore default)
for f in rna_features:
    if f not in X_geo.columns:
        X_geo[f] = 0
X_geo = X_geo[rna_features]  # ensure order matches model

# ----------------------------
# Step 5: Standardize features
# ----------------------------
X_geo = X_geo.fillna(X_geo.mean())
scaler = StandardScaler()
X_geo_std = pd.DataFrame(
    scaler.fit_transform(X_geo),
    columns=X_geo.columns,
    index=X_geo.index
)

# ----------------------------
# Step 6: Predict risk scores
# ----------------------------
risk_scores = log_cv_rna.decision_function(X_geo_std)

# ----------------------------
# Step 7: Extract survival time + status from metadata
# ----------------------------
clinical_data = []
for gsm, sample in gse.gsms.items():
    metadata_lines = sample.metadata.get("characteristics_ch1", [])
    entry = {"sample_id": gsm}
    
    for line in metadata_lines:
        if m := re.search(r'survival months\s*:\s*([\d.]+)', line, flags=re.IGNORECASE):
            entry["DFS_time"] = int(float(m.group(1)) * 30.44)  # convert months to days
        if m := re.search(r'survival status\s*:\s*([01])', line, flags=re.IGNORECASE):
            entry["DFS_event"] = int(m.group(1))  # 1 = event, 0 = censored
        if m := re.search(r'tissue\s*:\s*(pancreatic tumor)', line, flags=re.IGNORECASE):
            entry["is_tumor"] = True

    if "DFS_time" in entry and "DFS_event" in entry and entry.get("is_tumor", False):
        clinical_data.append(entry)

clinical_df = pd.DataFrame(clinical_data)
if "sample_id" in clinical_df.columns:
    clinical_df = clinical_df.set_index("sample_id")
    print(f"✅ Extracted DFS data for {len(clinical_df)} tumor samples.")
else:
    raise ValueError("❌ No valid clinical entries found with both DFS_time and DFS_event.")

# ----------------------------
# Step 8: Merge risk scores with clinical data
# ----------------------------
pred_df = pd.DataFrame({'risk_score': risk_scores}, index=X_geo_std.index)
merged = clinical_df.join(pred_df)
merged = merged.dropna(subset=['risk_score', 'DFS_time', 'DFS_event'])
print(f"✅ Merged data: {merged.shape[0]} samples")

# ----------------------------
# Step 9: Stratify by 80th percentile risk score (adjust if needed)
# ----------------------------
threshold = np.percentile(merged['risk_score'], 80)
merged['risk_group'] = np.where(merged['risk_score'] >= threshold, 'High Risk', 'Low Risk')


# ----------------------------
# Step 10: Plot Kaplan–Meier Curve
# ----------------------------
kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 6))

for group in ['Low Risk', 'High Risk']:
    mask = merged['risk_group'] == group
    if merged.loc[mask, 'DFS_time'].empty or merged.loc[mask, 'DFS_event'].empty:
        print(f"⚠️ No data for group '{group}', skipping KM plot")
        continue
    kmf.fit(merged.loc[mask, 'DFS_time'], merged.loc[mask, 'DFS_event'], label=group)
    kmf.plot_survival_function(ci_show=False)

plt.title("ProGIGene RNA Validation on GSE62452\n(Disease-Free Survival)")
plt.xlabel("Time (days)")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.savefig("ProGIGene_GSE62452_KM.png", dpi=300)
plt.show()

# ----------------------------
# Step 11: Compute AUC
# ----------------------------
auc = roc_auc_score(merged['DFS_event'], merged['risk_score'])
print(f"🎯 AUC (DFS prediction): {auc:.3f}")
