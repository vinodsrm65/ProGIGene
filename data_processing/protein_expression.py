## -- protein_expression-- ##
# -----------------------------
# Config
# -----------------------------
input_path = 'protein_expression.tsv' 
clinical_path = 'early_stage_clinical_labeled_dataset.csv'
output_path = 'rotein_expression.csv'

# -----------------------------
# Step 1: Load and transpose protein matrix
# -----------------------------
df_raw = pd.read_csv(input_path, sep='\t')
df_prot = df_raw.set_index('SampleID').T.reset_index().rename(columns={'index': 'sample'})
df_prot.columns = ['sample'] + [f"prot_{col}" for col in df_prot.columns[1:]]

# -----------------------------
# Step 2: Load clinical sample list
# -----------------------------
df_clinical = pd.read_csv(clinical_path)
valid_samples = set(df_clinical['sample'])

# -----------------------------
# Step 3: Filter to samples used in modeling
# -----------------------------
df_prot = df_prot[df_prot['sample'].isin(valid_samples)].drop_duplicates(subset='sample')

# -----------------------------
# Step 4: Fill missing protein values
# -----------------------------
df_prot.fillna(0, inplace=True)  # Or: df_prot.fillna(df_prot.median(), inplace=True)

# -----------------------------
# Step 5: Optional — filter top variable proteins
# -----------------------------
top_vars = df_prot.drop(columns='sample').var().sort_values(ascending=False).head(300).index
df_filtered = df_prot[['sample'] + list(top_vars)]

# -----------------------------
# Step 6: Save to local path
# -----------------------------
df_filtered.to_csv(output_path, index=False)
