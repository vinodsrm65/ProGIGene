import pandas as pd
import boto3
import numpy as np

#### -- Clinical data processing --- ###
# Load from S3
key = 'users/tyshaikh/progigene/raw_data/clinical_data.csv'
all_clinical_data = pd.read_csv(f's3://{bucket}/{key}', sep=',')

# Rename columns
all_clinical_data = all_clinical_data.rename(columns={
    '_PATIENT': 'patient_id',
    'cancer type abbreviation': 'cancer_type',
    'age_at_initial_pathologic_diagnosis': 'age_at_diagnosis',
    'ajcc_pathologic_tumor_stage': 'tumor_stage',
    'OS': 'os', 'DSS': 'dss', 'DFI': 'dfi', 'PFI': 'pfi',
    'OS.time': 'os_time', 'DSS.time': 'dss_time',
    'DFI.time': 'dfi_time', 'PFI.time': 'pfi_time',
})

# Select columns
selected_columns = ['sample',
                    'patient_id', 
                    'cancer_type',
                    'age_at_diagnosis', 
                    'gender',
                    'race',
                    'tumor_stage',
                    'vital_status',
                    'tumor_status',
                    'last_contact_days_to',
                    'death_days_to',
                    'new_tumor_event_type',
                    'new_tumor_event_site',
                    'new_tumor_event_site_other',
                    'new_tumor_event_dx_days_to',
                    'treatment_outcome_first_course',
                    'os',
                    'os_time',
                    'dss',
                    'dss_time',
                    'dfi',
                    'dfi_time',
                    'pfi',
                    'pfi_time']  

all_clinical_data = all_clinical_data[selected_columns]

# Filter GI cancer types
gi_cancer_types = ['COAD', 'ESCA', 'LIHC', 'PAAD', 'READ', 'STAD']
cancer_stages = ['Stage I',
                 'Stage IA',
                 'Stage IB',
                 'Stage IC',
                 'Stage II',
                 'Stage IIA',
                 'Stage IIB',
                 'Stage IIC',
                 'Stage IIIA',
                 'Stage IIIB',
                 'Stage IIIC',
                ]

gi_clinical_data = all_clinical_data[
    (all_clinical_data['cancer_type'].isin(gi_cancer_types)) & 
    (all_clinical_data['tumor_stage'].isin(cancer_stages)) &
    (all_clinical_data['pfi_time'].notna())
]

early_stage_clinical_labeled_dataset['pfi'] = early_stage_clinical_labeled_dataset['pfi'].astype(int)

# Assign early progression label (within 365 days)
early_stage_clinical_labeled_dataset['early_progression_label'] = early_stage_clinical_labeled_dataset.apply(
    lambda row: 1 if row['pfi'] == 1 and row['pfi_time'] <= 365 else 0,
    axis=1
)

early_stage_clinical_labeled_dataset.to_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/early_stage_clinical_labeled_dataset.csv', index=False)

### --- Gene expression processing ---- ###
key = 'users/tyshaikh/progigene/raw_data/gene_expression_v2.csv'

# Step 1: Load gene expression matrix from S3
all_gene_expression_data = pd.read_csv(f's3://{bucket}/{key}', sep=',')

# Step 2: Rename first column as 'gene_id' 
all_gene_expression_data.rename(columns={all_gene_expression_data.columns[0]: "gene_id"}, inplace=True)

# Step 3: Transpose so rows = samples, columns = genes
all_gene_expression_data_T = all_gene_expression_data.set_index("gene_id").T
all_gene_expression_data_T.reset_index(inplace=True)
all_gene_expression_data_T.rename(columns={"index": "sample"}, inplace=True)

# Step 4: Safely convert expression values to float and apply log2(x + 1)
expr_data = all_gene_expression_data_T.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')  
expr_data_log = np.log2(expr_data + 1)  
all_gene_expression_data_T.iloc[:, 1:] = expr_data_log

# Step 5: Rename expression columns as rna_<gene_id>
all_gene_expression_data_T.columns = ['sample'] + [f"rna_{col}" for col in all_gene_expression_data_T.columns[1:]]

# Step 6: Filter to early-stage samples from clinical dataset
early_stage_gene_expression = all_gene_expression_data_T[
    all_gene_expression_data_T['sample'].isin(early_stage_clinical_labeled_dataset['sample'])
].drop_duplicates(subset='sample')  # drop duplicates if any
early_stage_gene_expression.to_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/early_stage_gene_expression_all.csv', index=False)

### - pathways processing - ##
key = 'users/tyshaikh/progigene/raw_data/pathways.tsv'
gene_pathways = pd.read_csv(f's3://{bucket}/{key}', sep='\t')
gene_pathways.rename(columns={"Unnamed: 0": "pathway"}, inplace=True)
gene_pathways.set_index("pathway", inplace=True)
processed_gene_pathway = gene_pathways.T.reset_index().rename(columns={"index": "sample"})
early_stage_gene_pathway = processed_gene_pathway[processed_gene_pathway["sample"].isin(early_stage_gene_expression["sample"])]
early_stage_gene_pathway.to_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/early_stage_gene_pathway.csv', index=False)

### ---- Processing Mutation data --- ##
key = 'users/tyshaikh/progigene/raw_data/mutations.csv'
all_mutation_data = pd.read_csv(f's3://{bucket}/{key}',sep=',')
all_mutation_data = all_mutation_data[all_mutation_data['effect'] != "Silent"]
df_binary = pd.crosstab(all_mutation_data['sample'], all_mutation_data['gene'])
df_binary[df_binary > 0] = 1
df_binary.columns = [f"mut_{gene}" for gene in df_binary.columns]
df_binary.reset_index(inplace=True)
gene_mutation = df_binary[df_binary['sample'].isin(early_stage_clinical_labeled_dataset['sample'])]
min_samples = 5
gene_counts = gene_mutation.drop(columns='sample').sum(axis=0)
genes_to_keep = gene_counts[gene_counts >= min_samples].index
df_filtered = gene_mutation[['sample'] + list(genes_to_keep)]
df_filtered.to_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/gene_mutation.csv', index=False)

### -- Processing CNV data --- ##
key = 'users/tyshaikh/progigene/raw_data/cnv.csv'
all_cnv_data = pd.read_csv(f's3://{bucket}/{key}',sep=',')

# Rename first column and transpose
all_cnv_data.rename(columns={all_cnv_data.columns[0]: "gene"}, inplace=True)
all_cnv_data_t = all_cnv_data.set_index("gene").T.reset_index()
all_cnv_data_t.rename(columns={"index": "sample"}, inplace=True)
all_cnv_data_t.columns = ["sample"] + [f"cnv_{col}" for col in all_cnv_data_t.columns[1:]]

# Filter by early-stage samples
cnv_filtered = all_cnv_data_t[all_cnv_data_t['sample'].isin(early_stage_clinical_labeled_dataset['sample'])]

# Apply thresholding: -1 for loss, 1 for gain, 0 otherwise
cnv_array = cnv_filtered.iloc[:, 1:].astype(float).values
cnv_array = np.where(cnv_array <= -0.3, -1, np.where(cnv_array >= 0.3, 1, 0))

# Safely update DataFrame
cnv_filtered.iloc[:, 1:] = cnv_array

cnv_filtered.to_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/copy_number_variation.csv', index=False)


### --- Methylation pre processing --- ###
def transpose_meth_data(df, first_col_name='cpg_id'):
    df = df.copy()
    df.rename(columns={df.columns[0]: first_col_name}, inplace=True)
    df_t = df.set_index(first_col_name).T
    df_t.index.name = 'sample'
    df_t.reset_index(inplace=True)
    return df_t

# 2. Load and transpose all
df_esca = transpose_meth_data(pd.read_csv(f's3://{bucket}/users/tyshaikh/progigene/raw_data/ESCA_methylation450.tsv',sep='\t'))
df_coad = transpose_meth_data(pd.read_csv(f's3://{bucket}/users/tyshaikh/progigene/raw_data/COAD_methylation450.tsv',sep='\t'))
df_lihc = transpose_meth_data(pd.read_csv(f's3://{bucket}/users/tyshaikh/progigene/raw_data/LIHC_methylation450.tsv',sep='\t'))
df_paad = transpose_meth_data(pd.read_csv(f's3://{bucket}/users/tyshaikh/progigene/raw_data/PAAD_methylation450.tsv',sep='\t'))
df_read = transpose_meth_data(pd.read_csv(f's3://{bucket}/users/tyshaikh/progigene/raw_data/READ_methylation450.tsv',sep='\t'))
df_stad = transpose_meth_data(pd.read_csv(f's3://{bucket}/users/tyshaikh/progigene/raw_data/STAD_methylation450.tsv',sep='\t'))

# 3. Concatenate all transposed datasets
df_all_meth = pd.concat([df_esca, df_coad, df_lihc, df_paad, df_read, df_stad], ignore_index=True)

df_meth_filtered = df_all_meth[df_all_meth['sample'].isin(early_stage_clinical_labeled_dataset['sample'])]
df_meth_filtered.columns = ['sample'] + [f"meth_{col}" for col in df_meth_filtered.columns[1:]]

# 6. Save result
df_meth_filtered.to_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/gene_methylation.csv', index=False)


## -- Compile final training dataset --- ##
# Load all processed data files
df_clinical = pd.read_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/early_stage_clinical_labeled_dataset.csv')
clinical_cols_to_keep = [
    'cancer_type',
    'age_at_diagnosis',
    'gender',
    'race',
    'tumor_stage'
]
df_clinical_feature = df_clinical.drop(columns=[col for col in df_clinical.columns if col not in clinical_cols_to_keep + ['sample','early_progression_label']])

df_model = pd.get_dummies(df_clinical_feature, columns=[
    'cancer_type',
    'gender',
    'race',
    'tumor_stage'
], drop_first=True)



df_rna = pd.read_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/early_stage_gene_expression_all.csv')
df_mut = pd.read_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/gene_mutation.csv')
df_cnv = pd.read_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/copy_number_variation.csv')
df_meth = pd.read_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/gene_methylation.csv')
df_pathways = pd.read_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/early_stage_gene_pathway.csv')


training_dataset = df_model.merge(df_rna, on='sample', how='inner') \
                .merge(df_mut, on='sample', how='left') \
                .merge(df_cnv, on='sample', how='left') \
                .merge(df_meth, on='sample', how='left') \
                .merge(df_pathways, on='sample', how='left')

training_dataset.fillna(0, inplace=True)

training_dataset.to_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/final_training_dataset.csv', index=False)
print(training_dataset.shape)                          

