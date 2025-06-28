### - pathways processing - ##
key = 'pathways.tsv'
clinical_path = 'early_stage_clinical_labeled_dataset.csv'
gene_pathways = pd.read_csv('key', sep='\t')
gene_pathways.rename(columns={"Unnamed: 0": "pathway"}, inplace=True)
gene_pathways['pathway'] = (
    gene_pathways['pathway']
    .str.lower()
    .str.replace('-', '_')
    .str.replace('(', '_')
    .str.replace(')', '_')
    .str.replace(' ', '_')
    .str.replace('__', '_')
    .str.strip('_')
)
gene_pathways.set_index("pathway", inplace=True)
processed_gene_pathway = gene_pathways.T.reset_index().rename(columns={"index": "sample"})
processed_gene_pathway.columns = ['sample'] + [f"pathway_{col}" for col in processed_gene_pathway.columns[1:]]
df_clinical = pd.read_csv(clinical_path)
early_stage_gene_pathway = processed_gene_pathway[processed_gene_pathway["sample"].isin(df_clinical["sample"])]
early_stage_gene_pathway.to_csv('early_stage_gene_pathway.csv', index=False)
