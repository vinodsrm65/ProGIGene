## -- Compile final training dataset --- ##

# Load all processed data files
df_immune = pd.read_csv('immune_subtype.csv') 
df_clinical = pd.read_csv('early_stage_clinical_labeled_dataset.csv')
df_clinical = df_clinical.merge(df_immune, on='sample', how='left')
clinical_cols_to_keep = [
    'cancer_type',
    'age_at_diagnosis',
    'gender',
    'race',
    'tumor_stage',
    'clinical_stage',
    'histological_grade',
    'histological_type',
    'residual_tumor',
    'margin_status',
    'immune',
    'pfi',
    'pfi_time'
]

df_clinical_feature = df_clinical.drop(columns=[col for col in df_clinical.columns if col not in clinical_cols_to_keep + ['sample','early_progression_label']])

df_model = pd.get_dummies(df_clinical_feature, columns=[
    'cancer_type',
    'gender',
    'race',
    'tumor_stage',
    'clinical_stage',
    'histological_grade',
    'histological_type',
    'residual_tumor',
    'margin_status',
    'immune'
], drop_first=False)



df_rna = pd.read_csv('rna_exp.csv')
df_mut = pd.read_csv('gene_mutation.csv')
df_cnv = pd.read_csv('copy_number_variation.csv')
df_meth = pd.read_csv('methylation_top_shap.csv')
df_pathways = pd.read_csv('early_stage_gene_pathway.csv')
df_prot_exp = pd.read_csv('protein_expression.csv') 



# Merge RNA + Clinical first (fastest)
df_base = df_model.merge(df_rna, on='sample', how='inner')

# Then CNV, Mutation
df_base = df_base.merge(df_cnv, on='sample', how='left')
df_base = df_base.merge(df_mut, on='sample', how='left')

# Then Methylation (largest)
df_base = df_base.merge(df_meth, on='sample', how='left')
df_base = df_base.merge(df_prot_exp, on='sample', how='left')

# Finally Pathways
df_final = df_base.merge(df_pathways, on='sample', how='left')


df_final.fillna(0, inplace=True)

df_final.to_csv('final_training_dataset.csv', index=False)
                         
