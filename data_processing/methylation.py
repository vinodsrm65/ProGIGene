    ## --- methylation --- ###

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMClassifier
    import shap

    # ------------------------------
    # Config
    # ------------------------------
    clinical_key = 'early_stage_clinical_labeled_dataset.csv'
    output_path = 'methylation_top_shap.csv'

    # ------------------------------
    # Helper: Transpose and clean
    # ------------------------------
    def transpose_meth_data(df, first_col_name='cpg_id'):
        df = df.copy()
        df.rename(columns={df.columns[0]: first_col_name}, inplace=True)
        df_t = df.set_index(first_col_name).T
        df_t.index.name = 'sample'
        df_t.reset_index(inplace=True)
        return df_t

    # ------------------------------
    # Step 1: Load clinical
    # ------------------------------
    df_clinical = pd.read_csv('clinical_key')
    clinical_samples = set(df_clinical['sample'])

    # ------------------------------
    # Step 2: Load and filter methylation by sample match
    # ------------------------------
    def load_filtered(file_path):
        df = pd.read_csv(file_path, sep='\t')
        df_t = transpose_meth_data(df)
        return df_t[df_t['sample'].isin(clinical_samples)].drop_duplicates(subset='sample')

    df_esca = load_filtered('ESCA_methylation450.tsv')
    df_coad = load_filtered('COAD_methylation450.tsv')
    df_lihc = load_filtered('LIHC_methylation450.tsv')
    df_paad = load_filtered('PAAD_methylation450.tsv')
    df_read = load_filtered('READ_methylation450.tsv')
    df_stad = load_filtered('STAD_methylation450.tsv')

# ------------------------------
# Step 3: Merge all cancer types
# ------------------------------
df_all_meth = pd.concat([df_esca, df_coad, df_lihc, df_paad, df_read, df_stad], ignore_index=True)
df_meth_filtered = df_all_meth.drop_duplicates(subset='sample')

# ------------------------------
# Step 4: Select high-variance CpGs (top 2000)
# ------------------------------
cpg_vars = df_meth_filtered.drop(columns='sample').var()
top_cpgs = cpg_vars.sort_values(ascending=False).head(2000).index.tolist()
df_meth_top = df_meth_filtered[['sample'] + top_cpgs]

# ------------------------------
# Step 5: Merge with clinical labels
# ------------------------------
df_labeled = df_meth_top.merge(df_clinical[['sample', 'early_progression_label']], on='sample')
X = df_labeled.drop(columns=['sample', 'early_progression_label'])
y = df_labeled['early_progression_label']

# ------------------------------
# Step 6: Drop features with too many NaNs (>50%)
# ------------------------------
nan_fraction = X.isna().mean()
X = X.loc[:, nan_fraction < 0.5]

# ------------------------------
# Step 7: Impute missing values and scale
# ------------------------------
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ------------------------------
# Step 8: Train logistic model with ElasticNet penalty
# ------------------------------
model = LogisticRegressionCV(
    penalty='elasticnet',
    solver='saga',
    max_iter=5000,
    l1_ratios=[0.1, 0.5, 0.9],
    class_weight='balanced',
    cv=5,
    random_state=42
)
model.fit(X_scaled, y)

# ------------------------------
# Step 9: Select top CpGs by absolute coefficient
# ------------------------------
coef_series = pd.Series(np.abs(model.coef_.flatten()), index=X.columns)
top_logreg_cpgs = coef_series.sort_values(ascending=False).head(500).index.tolist()

# ------------------------------
# Step 10: Final export
# ------------------------------
df_meth_final = df_meth_filtered[['sample'] + top_logreg_cpgs]
df_meth_final.columns = ['sample'] + [f"meth_{str(c).replace('.', '_').replace('-', '_')}" for c in df_meth_final.columns[1:]]
df_meth_final.to_csv(output_path, index=False)
