### ---- Processing Mutation data --- ##
key = 'mutations.csv'
all_mutation_data = pd.read_csv('key',sep=',')
all_mutation_data = all_mutation_data[all_mutation_data['effect'] != "Silent"]
df_binary = pd.crosstab(all_mutation_data['sample'], all_mutation_data['gene'])
df_binary[df_binary > 0] = 1
df_binary.columns = [f"mut_{gene}" for gene in df_binary.columns]
df_binary.reset_index(inplace=True)
df_clinical = pd.read_csv(f's3://{bucket}/{CLINICAL_PATH}', usecols=['sample', 'early_progression_label'])
gene_mutation = df_binary[df_binary['sample'].isin(df_clinical['sample'])]
min_samples = 5
gene_counts = gene_mutation.drop(columns='sample').sum(axis=0)
genes_to_keep = gene_counts[gene_counts >= min_samples].index
df_filtered = gene_mutation[['sample'] + list(genes_to_keep)]
df_filtered = df_filtered.copy()
df_filtered['mutation_count'] = df_filtered.drop(columns='sample').sum(axis=1)
df_filtered.to_csv('gene_mutation.csv', index=False)
