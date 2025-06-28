import GEOparse
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# ----------------------------
# Helper: extract gene symbols from GPL6244
# ----------------------------
def extract_gene_symbol(x):
    if not isinstance(x, str):
        return np.nan
    parts = re.split(r'//+|\s*///\s*', x)
    for part in parts:
        part = part.strip()
        if part and re.match(r'^[A-Za-z0-9\-]+$', part):
            return part.upper()
    return np.nan

# ----------------------------
# Download and parse GSE14333 (Colorectal)
# ----------------------------
print("Downloading GSE14333...")
gse1 = GEOparse.get_GEO(geo="GSE14333", destdir="./geo_data")
geo_expr_raw1 = gse1.pivot_samples("VALUE")
gpl1 = GEOparse.get_GEO(geo="GPL570", destdir="./geo_data")
platform_table1 = gpl1.table[['ID', 'Gene Symbol']].dropna()
platform_table1.columns = ['probe_id', 'gene_symbol']
geo_expr1 = geo_expr_raw1.copy()
geo_expr1['probe_id'] = geo_expr1.index
geo_expr1 = geo_expr1.merge(platform_table1, on='probe_id', how='left')
geo_expr1 = geo_expr1.dropna(subset=['gene_symbol']).drop(columns='probe_id')
geo_expr1 = geo_expr1.groupby('gene_symbol').mean().T
geo_expr1.columns = geo_expr1.columns.str.upper()

# ----------------------------
# Download and parse GSE62452 (Pancreatic)
# ----------------------------
print("Downloading GSE62452...")
gse2 = GEOparse.get_GEO(geo="GSE62452", destdir="./geo_data")
geo_expr_raw2 = gse2.pivot_samples("VALUE")
gpl2 = GEOparse.get_GEO(geo="GPL6244", destdir="./geo_data")
gpl2.table['gene_symbol'] = gpl2.table['gene_assignment'].apply(extract_gene_symbol)
platform_table2 = gpl2.table[['ID', 'gene_symbol']].dropna()
platform_table2.columns = ['probe_id', 'gene_symbol']
geo_expr2 = geo_expr_raw2.copy()
geo_expr2['probe_id'] = geo_expr2.index
geo_expr2 = geo_expr2.merge(platform_table2, on='probe_id', how='left')
geo_expr2 = geo_expr2.dropna(subset=['gene_symbol']).drop(columns='probe_id')
geo_expr2 = geo_expr2.groupby('gene_symbol').mean().T
geo_expr2.columns = geo_expr2.columns.str.upper()

# ----------------------------
# Match RNA model features
# ----------------------------
rna_features = [f for f in log_cv_rna_2.feature_names_in_ if f.startswith("rna_")]
rna_gene_symbols = [f.replace("rna_", "").upper() for f in rna_features]

def prepare_geo_expr(expr_df, gene_list, rna_features):
    common = expr_df.columns.intersection(gene_list)
    X = expr_df[common].copy()
    X.columns = ['rna_' + g for g in X.columns]
    for f in rna_features:
        if f not in X.columns:
            X[f] = 0
    X = X[rna_features].fillna(X.mean())
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)

X_geo1_std = prepare_geo_expr(geo_expr1, rna_gene_symbols, rna_features)
X_geo2_std = prepare_geo_expr(geo_expr2, rna_gene_symbols, rna_features)

risk_scores1 = log_cv_rna_2.decision_function(X_geo1_std)
risk_scores2 = log_cv_rna_2.decision_function(X_geo2_std)

# ----------------------------
# Parse clinical metadata (Colorectal)
# ----------------------------
clinical_data1 = []
for gsm, sample in gse1.gsms.items():
    metadata_lines = sample.metadata.get("characteristics_ch1", [])
    entry = {}
    for line in metadata_lines:
        for part in line.split(";"):
            if "dfs_time" in part.lower():
                try:
                    entry["DFS_time"] = int(float(part.split(":")[1].strip()) * 30.44)
                except: continue
            if "dfs_cens" in part.lower():
                try:
                    entry["DFS_event"] = 1 - int(part.split(":")[1].strip())
                except: continue
    if "DFS_time" in entry and "DFS_event" in entry:
        entry["sample_id"] = gsm
        clinical_data1.append(entry)
clinical_df1 = pd.DataFrame(clinical_data1).set_index("sample_id")

# ----------------------------
# Parse clinical metadata (Pancreatic)
# ----------------------------
clinical_data2 = []
for gsm, sample in gse2.gsms.items():
    metadata_lines = sample.metadata.get("characteristics_ch1", [])
    entry = {"sample_id": gsm}
    for line in metadata_lines:
        if m := re.search(r'survival months\s*:\s*([\d.]+)', line, flags=re.IGNORECASE):
            entry["DFS_time"] = int(float(m.group(1)) * 30.44)
        if m := re.search(r'survival status\s*:\s*([01])', line, flags=re.IGNORECASE):
            entry["DFS_event"] = int(m.group(1))
        if re.search(r'tissue\s*:\s*(pancreatic tumor)', line, flags=re.IGNORECASE):
            entry["is_tumor"] = True
    if "DFS_time" in entry and "DFS_event" in entry and entry.get("is_tumor", False):
        clinical_data2.append(entry)
clinical_df2 = pd.DataFrame(clinical_data2).set_index("sample_id")

# ----------------------------
# Merge clinical + risk
# ----------------------------
merged1 = clinical_df1.join(pd.DataFrame({'risk_score': risk_scores1}, index=X_geo1_std.index)).dropna()
merged2 = clinical_df2.join(pd.DataFrame({'risk_score': risk_scores2}, index=X_geo2_std.index)).dropna()

merged1['risk_group'] = np.where(merged1['risk_score'] >= np.percentile(merged1['risk_score'], 66), 'High Risk', 'Low Risk')
merged2['risk_group'] = np.where(merged2['risk_score'] >= np.percentile(merged2['risk_score'], 80), 'High Risk', 'Low Risk')

# ----------------------------
# Plot Kaplan–Meier (Combined)
# ----------------------------
plt.figure(figsize=(10, 6))
kmf = KaplanMeierFitter()

for title, df in [("GSE14333 (Colorectal)", merged1), ("GSE62452 (Pancreatic)", merged2)]:
    for risk_group in ['Low Risk', 'High Risk']:
        mask = df['risk_group'] == risk_group
        kmf.fit(df.loc[mask, 'DFS_time'], df.loc[mask, 'DFS_event'], label=f"{title} - {risk_group}")
        kmf.plot_survival_function(ci_show=False)

plt.title("Combined RNA Validation (Disease-Free Survival)")
plt.xlabel("Time (days)")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.savefig("Combined_GEO_KM.png", dpi=300)
plt.show()

# ----------------------------
# AUC Scores
# ----------------------------
auc1 = roc_auc_score(merged1['DFS_event'], merged1['risk_score'])
auc2 = roc_auc_score(merged2['DFS_event'], merged2['risk_score'])
print(f"Colorectal AUC (GSE14333): {auc1:.3f}")
print(f"Pancreatic AUC (GSE62452): {auc2:.3f}")
