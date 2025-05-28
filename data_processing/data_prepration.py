
import pandas as pd
import boto3
import numpy as np

# Initialize S3 client
s3 = boto3.client('s3')

# S3 bucket and file information
bucket = 'data-bucket'
key = 'progigene/raw_data/clinical_data.csv'

all_clinical_data = pd.read_csv(f's3://{bucket}/{key}', sep=',')

all_clinical_data = all_clinical_data.rename(columns={
    '_PATIENT': 'patient_id',
    'cancer type abbreviation': 'cancer_type',
    'age_at_initial_pathologic_diagnosis': 'age_at_diagnosis',

    'ajcc_pathologic_tumor_stage': 'tumor_stage',
    'OS': 'os',
    'DSS': 'dss',
    'DFI': 'dfi',
    'PFI': 'pfi',
    'OS.time': 'os_time',
    'DSS.time': 'dss_time',
    'DFI.time': 'dfi_time',
    'PFI.time': 'pfi_time',
})

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
                    'pfi_time'
                    ]
all_clinical_data = all_clinical_data[selected_columns]



gi_cancer_types = ['COAD',
                   'ESCA',
                   'LIHC',
                   'PAAD',
                   'READ',
                   'STAD']
gi_clinical_data = all_clinical_data[all_clinical_data['cancer_type'].isin(gi_cancer_types)]
early_stage_clinical_labeled_dataset = gi_clinical_data[
    gi_clinical_data['tumor_stage'].str.contains("Stage I|Stage II|Stage III", case=False, na=False) &
    gi_clinical_data['pfi_time'].notna()
].copy()



early_stage_clinical_labeled_dataset['early_progression_label'] = early_stage_clinical_labeled_dataset.apply(
    lambda row: 1 if row['pfi'] == 1 and row['pfi_time'] <= 365 else 0,
    axis=1
)

early_stage_clinical_labeled_dataset.to_csv(f's3://{bucket}/progigene/processed_progigene/early_stage_clinical_labeled_dataset.csv', index=False)


### --- gene expression processing ---- ###
key = 'progigene/raw_data/gene_expression_v2.csv'
all_gene_expression_data = pd.read_csv(f's3://{bucket}/{key}', sep=',')

all_gene_expression_data.rename(columns={all_gene_expression_data.columns[0]: "gene_id"}, inplace=True)
all_gene_expression_data_T = all_gene_expression_data.set_index("gene_id").T
all_gene_expression_data_T.reset_index(inplace=True)
all_gene_expression_data_T.rename(columns={"index": "sample"}, inplace=True)
all_gene_expression_data_T.iloc[:, 1:] = np.log2(all_gene_expression_data_T.iloc[:, 1:].astype(float) + 1)
all_gene_expression_data_T.columns = ['sample'] + [f"rna_{col}" for col in all_gene_expression_data_T.columns[1:]]


early_stage_gene_expression = all_gene_expression_data_T[all_gene_expression_data_T['sample'].isin(early_stage_clinical_labeled_dataset['sample'])]

early_stage_gene_expression.to_csv(f's3://{bucket}/progigene/processed_progigene/early_stage_gene_expression_all.csv', index=False)

early_stage_gene_expression = pd.read_csv(f's3://{bucket}/progigene/processed_progigene/early_stage_gene_expression_all.csv',sep=',')

### --- Select Top 1000 Genes by Variance --- ###

gene_columns = early_stage_gene_expression.columns.drop("sample")

variances = early_stage_gene_expression[gene_columns].var().sort_values(ascending=False)

top_genes = variances.head(1000).index

df_top_genes = early_stage_gene_expression[["sample"] + list(top_genes)]

df_top_genes.to_csv(f's3://{bucket}/progigene/processed_progigene/gene_expression_top1000.csv', index=False)

### ---- Processing Mutation data --- ##

all_mutation_data = pd.read_csv(f's3://{bucket}/progigene/raw_data/mutations.csv',sep=',')

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
df_filtered.to_csv(f's3://{bucket}/progigene/processed_progigene/gene_mutation.csv', index=False)

### -- Processing CNV data --- ##

all_cnv_data = pd.read_csv(f's3://{bucket}/progigene/raw_data/cnv.csv',sep=',')

all_cnv_data.rename(columns={all_cnv_data.columns[0]: "gene"}, inplace=True)

all_cnv_data_t = all_cnv_data.set_index("gene").T.reset_index()
all_cnv_data_t.rename(columns={"index": "sample"}, inplace=True)
all_cnv_data_t.columns = ["sample"] + [f"cnv_{col}" for col in all_cnv_data_t.columns[1:]]

cnv_filtered = all_cnv_data_t[all_cnv_data_t['sample'].isin(early_stage_clinical_labeled_dataset['sample'])]

cnv_array = cnv_filtered.iloc[:, 1:].values
cnv_array = np.where(cnv_array <= -0.3, -1, 
                     np.where(cnv_array >= 0.3, 1, 0))
cnv_filtered.iloc[:, 1:] = cnv_array

cnv_filtered.to_csv(f's3://{bucket}/progigene/processed_progigene/copy_number_variation.csv', index=False)

### --- Methylation pre processing --- ###

def transpose_meth_data(df, first_col_name='cpg_id'):
    df = df.copy()
    df.rename(columns={df.columns[0]: first_col_name}, inplace=True)
    df_t = df.set_index(first_col_name).T
    df_t.index.name = 'sample'
    df_t.reset_index(inplace=True)
    return df_t

# 2. Load and transpose all
df_esca = transpose_meth_data(pd.read_csv(f's3://{bucket}/raw_data/ESCA_methylation450.tsv',sep='\t'))
df_coad = transpose_meth_data(pd.read_csv(f's3://{bucket}/raw_data/COAD_methylation450.tsv',sep='\t'))
df_lihc = transpose_meth_data(pd.read_csv(f's3://{bucket}/raw_data/LIHC_methylation450.tsv',sep='\t'))
df_paad = transpose_meth_data(pd.read_csv(f's3://{bucket}/raw_data/PAAD_methylation450.tsv',sep='\t'))
df_read = transpose_meth_data(pd.read_csv(f's3://{bucket}/raw_data/READ_methylation450.tsv',sep='\t'))
df_stad = transpose_meth_data(pd.read_csv(f's3://{bucket}/raw_data/STAD_methylation450.tsv',sep='\t'))

# 3. Concatenate all transposed datasets
df_all_meth = pd.concat([df_esca, df_coad, df_lihc, df_paad, df_read, df_stad], ignore_index=True)

# 4. Calculate CpG variances across combined samples
cpg_variance = df_all_meth.drop(columns='sample').var().sort_values(ascending=False)
top_cpgs = cpg_variance.head(1000).index.tolist()

# 5. Filter and prefix
df_meth_filtered = df_all_meth[['sample'] + top_cpgs]
df_meth_filtered.columns = ['sample'] + [f"meth_{col}" for col in df_meth_filtered.columns[1:]]

# 6. Save result
df_meth_filtered.to_csv(f's3://{bucket}/progigene/processed_progigene/methylation_top1000_cpgs.csv', index=False)


import pandas as pd
from functools import reduce

bucket = 'shelfspace-alpha-sandbox'

# Load all processed data files
df_clinical = pd.read_csv(f's3://{bucket}/progigene/processed_progigene/early_stage_clinical_labeled_dataset.csv')
df_rna = pd.read_csv(f's3://{bucket}/progigene/processed_progigene/gene_expression_top1000.csv')
df_mut = pd.read_csv(f's3://{bucket}/progigene/processed_progigene/gene_mutation.csv')
df_cnv = pd.read_csv(f's3://{bucket}/progigene/processed_progigene/copy_number_variation.csv')
df_meth = pd.read_csv(f's3://{bucket}/progigene/processed_progigene/methylation_top1000_cpgs.csv')

# Ensure all have a 'sample' column
for df in [df_clinical, df_rna, df_mut, df_cnv, df_meth]:
    df.rename(columns={df.columns[0]: "sample"}, inplace=True)

# Merge all datasets on 'sample' using inner join
training_dataset = reduce(lambda left, right: pd.merge(left, right, on='sample', how='inner'),
                   [df_clinical, df_rna, df_mut, df_cnv, df_meth])
training_dataset.to_csv(f's3://{bucket}/progigene/processed_progigene/final_training_dataset.csv', index=False)




