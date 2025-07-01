### - Immune subtype - ##
key = 'immune_subtype.tsv'

immune_subtype = pd.read_csv('key', sep='\t')
immune_subtype.rename(columns={"Subtype_Immune_Model_Based": "immune"}, inplace=True)

# Clean and standardize
immune_subtype['immune'] = (
    immune_subtype['immune']
    .str.lower()
    .str.replace('-', '_')
    .str.replace('(', '_')
    .str.replace(')', '_')
    .str.replace(' ', '_')
    .str.replace('__', '_')
    .str.strip('_')
)

# Add prefix
immune_subtype['immune'] = 'immune_' + immune_subtype['immune'].astype(str)

# Filter to clinical samples
df_clinical = pd.read_csv(clinical_path)
processed_immune_subtype = immune_subtype[immune_subtype["sample"].isin(df_clinical["sample"])]

# Export
processed_immune_subtype.to_csv('immune_subtype.csv',index=False)
