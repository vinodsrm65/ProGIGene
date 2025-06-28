import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

# ------------------------------
# Config
# ------------------------------
bucket = 'shelfspace-alpha-sandbox'
EXPR_PATH = 'users/tyshaikh/progigene/raw_data/gene_expression_v2.csv'
CLINICAL_PATH = 'users/tyshaikh/progigene/processed_progigene/elastic/early_stage_clinical_labeled_dataset.csv'
OUTPUT_PATH = 'users/tyshaikh/progigene/processed_progigene/elastic/rna_exp.csv' 

# ------------------------------
# Step 1: Load expression and clinical data
# ------------------------------
df_expr = pd.read_csv('gene_expression.csv')
df_expr.rename(columns={df_expr.columns[0]: "gene_id"}, inplace=True)
df_expr_t = df_expr.set_index("gene_id").T.reset_index().rename(columns={"index": "sample"})

# Log2 transform with clipping to avoid invalid log values
expr_numeric = df_expr_t.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').clip(lower=0)
expr_log2 = np.log2(expr_numeric + 1)
df_expr_t.iloc[:, 1:] = expr_log2
df_expr_t.columns = ['sample'] + [f"rna_{col}" for col in df_expr_t.columns[1:]]

# Load clinical labels
df_clinical = pd.read_csv('early_stage_clinical_labeled_dataset.csv', usecols=['sample', 'early_progression_label'])
df_expr_t = df_expr_t[df_expr_t['sample'].isin(df_clinical['sample'])].drop_duplicates(subset='sample')

# ------------------------------
# Step 2: Variance filtering - top 1000 genes
# ------------------------------
gene_vars = df_expr_t.drop(columns=["sample"]).var()
top_genes = gene_vars.sort_values(ascending=False).head(1000).index.tolist()
df_top_expr = df_expr_t[["sample"] + top_genes]

# ------------------------------
# Step 3: WGCNA-like Module Detection
# ------------------------------
def wgcna_module_genes(expression_df, clinical_df, max_modules=10):
    df = expression_df.set_index("sample")
    df_std = pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)
    corr = df_std.corr()
    dissimilarity = 1 - np.abs(corr)
    # Convert square matrix to condensed for linkage
    Z = linkage(squareform(dissimilarity), method='average')
    module_labels = fcluster(Z, t=max_modules, criterion='maxclust')
    module_df = pd.DataFrame({'gene': corr.columns, 'module': module_labels})
    
    # Compute eigengenes
    eigengenes = {}
    for m in sorted(set(module_labels)):
        genes = module_df[module_df['module'] == m]['gene'].tolist()
        eigengenes[f"Module_{m}"] = df_std[genes].mean(axis=1)
    
    eigengenes_df = pd.DataFrame(eigengenes)
    eigengenes_df["sample"] = eigengenes_df.index
    traits = clinical_df[['sample', 'early_progression_label']]
    eigengenes_df = eigengenes_df.reset_index(drop=True).merge(traits, on="sample")
    
    module_corr = eigengenes_df.drop(columns=["sample", "early_progression_label"]).corrwith(
        eigengenes_df["early_progression_label"])
    top_module = module_corr.abs().sort_values(ascending=False).index[0]
    top_module_index = int(top_module.replace("Module_", ""))
    return module_df[module_df['module'] == top_module_index]['gene'].tolist()

top_wgcna_genes = wgcna_module_genes(df_top_expr, df_clinical)

# ------------------------------
# Step 4: Merge features for model input
# ------------------------------
all_genes = list(set(top_genes).union(set(top_wgcna_genes)))
df_model_input = df_expr_t[["sample"] + all_genes].merge(df_clinical, on='sample')

X = df_model_input.drop(columns=["sample", "early_progression_label"])
y = df_model_input["early_progression_label"]

# ------------------------------
# Step 5: Remove features with >50% NaNs
# ------------------------------
nan_fraction = X.isna().mean()
X = X.loc[:, nan_fraction < 0.5]

# ------------------------------
# Step 6: Impute remaining NaNs and scale
# ------------------------------
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ------------------------------
# Step 7: Train LogisticRegressionCV (ElasticNet-style)
# ------------------------------
log_cv = LogisticRegressionCV(
    cv=5,
    penalty='elasticnet',
    solver='saga',
    max_iter=5000,
    l1_ratios=[0.1, 0.5, 0.9],
    class_weight='balanced',
    random_state=42
)
log_cv.fit(X_scaled, y)

# ------------------------------
# Step 8: Select Top 500 Genes by Absolute Coefficients
# ------------------------------
coef_series = pd.Series(np.abs(log_cv.coef_.flatten()), index=X.columns)
top_genes_log = coef_series.sort_values(ascending=False).head(500).index.tolist()

# ------------------------------
# Step 9: Save Final Dataset
# ------------------------------
df_final = df_expr_t[['sample'] + top_genes_log]
df_final.to_csv('rna_exp.csv', index=False)
