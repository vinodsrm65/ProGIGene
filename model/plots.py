
# -----------------------------
# ROC Curve Plot
# -----------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(risk_test_labels, y_prob)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='orange', lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------
# PCA Projection 
# --------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from mpl_toolkits.mplot3d import Axes3D


# Target variable
y = df_final['pfi_time']

# Drop non-feature columns and keep only numeric
non_feature_cols = ['sample', 'pfi', 'pfi_time', 'early_progression_label', 'risk_group']
X = df_final.drop(columns=[col for col in non_feature_cols if col in df_final.columns], errors='ignore')
X = X.select_dtypes(include=[np.number])

# Select top 500 most variable features
top_vars = X.var().nlargest(500).index
X_var = X[top_vars]

# Mean imputation
X_imputed = X_var.fillna(X_var.mean())

# Standardize
X_std = (X_imputed - X_imputed.mean()) / X_imputed.std()

# Transpose for PCA (features as rows)
X_t = X_std.T

# PCA to reduce to 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_t)

# Compute correlation of each feature with survival
feature_corrs = X_imputed.apply(lambda col: spearmanr(col, y).correlation).fillna(0)
norm_corrs = (feature_corrs - feature_corrs.min()) / (feature_corrs.max() - feature_corrs.min())

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                c=norm_corrs.loc[top_vars], cmap='coolwarm', s=30, alpha=0.8)

ax.set_title('3D PCA Projection of Top 500 Features\nColored by Survival Correlation (PFI Time)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label('Normalized Spearman Correlation with PFI Time')

plt.tight_layout()
plt.show()

# --------------------------------
# Spearman Correlation
# --------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Target survival time
y = df_final['pfi_time']

# Drop non-feature columns and keep numeric features only
non_feature_cols = ['sample', 'pfi', 'pfi_time', 'early_progression_label', 'risk_group']
X = df_final.drop(columns=[col for col in non_feature_cols if col in df_final.columns], errors='ignore')
X = X.select_dtypes(include=[np.number])

# Select top 500 most variable features
top_var_features = X.var().nlargest(500).index
X_top = X[top_var_features]

# Compute Spearman correlations with real survival
real_corrs = X_top.apply(lambda col: spearmanr(col, y).correlation)

# Permuted survival vector
np.random.seed(42)
y_permuted = np.random.permutation(y)

# Compute Spearman correlations with permuted survival
perm_corrs = X_top.apply(lambda col: spearmanr(col, y_permuted).correlation)

# Plot
plt.figure(figsize=(5, 10))
plt.hist(real_corrs.dropna(), bins=40, alpha=0.6, color='red', label='Real Survival',orientation='horizontal')
plt.hist(perm_corrs.dropna(), bins=40, alpha=0.6, color='blue', label='Permuted Survival',orientation='horizontal')
plt.title('Histogram of Correlations with PFI (Top 500 Most Variable Features)')
plt.xlabel('Spearman Correlation')
plt.ylabel('Number of Features')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------
# kaplan_meier_by_type
# --------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Copy working DataFrame
df = df_final.copy()

# List all one-hot encoded cancer type columns
cancer_type_cols = [col for col in df.columns if col.startswith("cancer_type_")]
cancer_types = [col.replace("cancer_type_", "") for col in cancer_type_cols]

# Plot settings
n_cols = 3
n_rows = (len(cancer_types) + n_cols - 1) // n_cols
fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True, sharey=True)
axs = axs.flatten()

# Generate KM curves per cancer type
for idx, cancer in enumerate(cancer_types):
    ax = axs[idx]
    colname = f"cancer_type_{cancer}"
    if colname not in df.columns:
        continue
    
    subset = df[df[colname] == 1].copy()

    # Check if both groups exist
    if subset['early_progression_label'].nunique() < 2:
        ax.set_title(f"{cancer} (Not enough groups)")
        continue

    kmf = KaplanMeierFitter()

    for group in ['Low Risk', 'High Risk']:
        label_value = 0 if group == 'Low Risk' else 1
        mask = subset['early_progression_label'] == label_value
        if mask.sum() == 0:
            continue
        kmf.fit(subset.loc[mask, 'pfi_time'], event_observed=subset.loc[mask, 'pfi'], label=group)
        kmf.plot_survival_function(ax=ax, ci_show=False)

    # Log-rank test
    try:
        results = logrank_test(
            subset[subset['early_progression_label'] == 0]['pfi_time'],
            subset[subset['early_progression_label'] == 1]['pfi_time'],
            event_observed_A=subset[subset['early_progression_label'] == 0]['pfi'],
            event_observed_B=subset[subset['early_progression_label'] == 1]['pfi']
        )
        p_value = results.p_value
        ax.set_title(f"{cancer} (log-rank P={p_value:.3e})")
    except:
        ax.set_title(f"{cancer} (log-rank error)")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Progression-Free Survival")

# Remove unused subplots
for i in range(len(cancer_types), len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("kaplan_meier_by_type.png", dpi=300)
plt.show()

# --------------------------------
# Pathway Enrichment
# --------------------------------


import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Extract top RNA gene names (strip rna_ prefix)
top_n = 250
feature_df = pd.DataFrame({
    'feature': log_cv.feature_names_in_,
    'coefficient': log_cv.coef_.flatten()
})
feature_df = feature_df[feature_df['feature'].str.startswith('rna_')]
feature_df['abs_coef'] = feature_df['coefficient'].abs()
top_genes = (
    feature_df
    .sort_values(by='abs_coef', ascending=False)
    .head(top_n)['feature']
    .str.replace('rna_', '', regex=False)
    .tolist()
)

# Step 2: Run enrichment using Enrichr (Reactome/KEGG/GO)
enr = gp.enrichr(
    gene_list=top_genes,
    gene_sets='KEGG_2021_Human',  # You can use 'KEGG_2021_Human' or 'GO_Biological_Process_2021'
    organism='Human',
    outdir=None,
    cutoff=0.05
)

# Step 3: Prepare dot plot
top_results = enr.results.sort_values('Adjusted P-value').head(15)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=top_results,
    x='Combined Score',
    y='Term',
    size='Overlap',
    hue='Adjusted P-value',
    palette='coolwarm_r',
    sizes=(30, 200),
    edgecolor='black'
)
plt.title('Pathway Enrichment of Top 250 RNA Features')
plt.xlabel('Combined Enrichment Score')
plt.ylabel('Pathway')
plt.legend(title='Adj. P-value', loc='lower right')
plt.tight_layout()
plt.show()

# --------------------------------
# omics_summary
# --------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def classify_omics_layer(feature):
    if feature.startswith('rna_'):
        return 'Transcriptomics'
    elif feature.startswith('meth_'):
        return 'Methylation'
    elif feature.startswith('mut_'):
        return 'Mutation'
    elif feature.startswith('cnv_'):
        return 'CNV'
    elif feature.startswith('prot_'):
        return 'Proteomics'
    elif feature.startswith('pathway_'):
        return 'Pathway'
    else:
        return 'Other'

feature_layer_map = {f: classify_omics_layer(f) for f in top_features}
layer_series = pd.Series(feature_layer_map)
layer_counts = layer_series.value_counts()
layer_counts = layer_counts[layer_counts.index != 'Other']  # drop 'Other'

# -----------------------------
# Step 2: Compute Per-Layer Z-Scores
# -----------------------------
X_test_labeled = X_test_final.copy()
X_test_labeled['sample'] = df_test_clin['sample'].values
X_test_labeled = X_test_labeled.set_index('sample')

layer_df = pd.DataFrame.from_dict(feature_layer_map, orient='index', columns=['OmicsLayer'])
layer_df = layer_df[layer_df['OmicsLayer'] != 'Other']  # drop 'Other'

layer_signals = {}
for layer in layer_df['OmicsLayer'].unique():
    features = layer_df[layer_df['OmicsLayer'] == layer].index.tolist()
    if features:
        layer_data = X_test_labeled[features]
        layer_z = pd.DataFrame(StandardScaler().fit_transform(layer_data),
                               columns=features, index=layer_data.index)
        layer_signals[layer] = layer_z.clip(-2, 2).median(axis=1)

df_layerwise_z = pd.DataFrame(layer_signals)
df_layerwise_z = df_layerwise_z.reset_index()  # 'sample' column created

# -----------------------------
# Step 3: Bin Patients by Risk Score
# -----------------------------
df_risk = pd.DataFrame({
    'sample': df_test_clin['sample'].values,
    'risk_score': y_prob
})
df_risk['risk_bin'] = pd.qcut(df_risk['risk_score'], q=5, labels=[
    'Very Low', 'Low', 'Medium', 'High', 'Very High'
])

# Merge risk + z-score data
df_grouped = df_risk.merge(df_layerwise_z, on='sample')

# Group by risk_bin and get median z-score per omics layer
heatmap_data = df_grouped.drop(columns=['sample', 'risk_score']).groupby('risk_bin').median()

# -----------------------------
# Step 4: Plot & Save Combined Figure
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 2]})

# Bar plot for layer contribution
layer_counts.plot(kind='bar', ax=axes[0], edgecolor='black')
axes[0].set_title('Omics Layer Contribution (Top Features)')
axes[0].set_ylabel('Number of Features')
axes[0].set_xlabel('Omics Layer')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', linestyle='--', alpha=0.6)

# Heatmap
sns.heatmap(
    heatmap_data,
    cmap='coolwarm',
    ax=axes[1],
    linewidths=0.5,
    linecolor='gray',
    vmin=-2, vmax=2,
    annot=True, fmt=".2f",
    cbar_kws={"label": "Z-Scored Median Signal"}
)
axes[1].set_title('Omics Activation Across Binned Risk Groups')
axes[1].set_xlabel('Omics Layer')
axes[1].set_ylabel('Risk Group')

plt.tight_layout()
plt.savefig("omics_summary.png", dpi=300, bbox_inches='tight')
plt.show()

# --------------------------------
# feature_analysis_summary
# --------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
from sklearn.preprocessing import StandardScaler

# -----------------------------
# CONFIG
# -----------------------------
top_n = 30      # Top features to show in bar & heatmap
enrich_n = 250  # RNA features to send to enrichment

# -----------------------------
# Helper: Abbreviate & classify
# -----------------------------
def abbreviate_feature(name):
    name = name.replace('rna_', '').replace('meth_', '').replace('mut_', '')\
               .replace('cnv_', '').replace('prot_', '').replace('pathway_', 'p_')\
               .replace('_', ' ')
    return name[:25] + '...' if len(name) > 28 else name

def classify_omics_layer(feature):
    if feature.startswith('rna_'): return 'Transcriptomics'
    elif feature.startswith('meth_'): return 'Methylation'
    elif feature.startswith('mut_'): return 'Mutation'
    elif feature.startswith('cnv_'): return 'CNV'
    elif feature.startswith('prot_'): return 'Proteomics'
    elif feature.startswith('pathway_'): return 'Pathway'
    else: return 'Other'

clinical_keywords = ['tumor_stage', 'histological', 'gender', 'age', 'clinical_stage', 'race', 'margin', 'residual']

# -----------------------------
# Step 1: Filter feature coefficients (exclude clinical)
# -----------------------------
coef_series = pd.Series(np.abs(log_cv.coef_.flatten()), index=log_cv.feature_names_in_)
feature_df = pd.DataFrame({
    'feature': log_cv.feature_names_in_,
    'coefficient': log_cv.coef_.flatten()
})
feature_df['abs_coef'] = feature_df['coefficient'].abs()
feature_df['omics'] = feature_df['feature'].apply(classify_omics_layer)
is_clinical = feature_df['feature'].apply(lambda f: any(kw in f for kw in clinical_keywords))
feature_df_filtered = feature_df[~is_clinical].copy()

top_plot = feature_df_filtered.sort_values('abs_coef', ascending=False).head(top_n)
top_plot['short_name'] = top_plot['feature'].apply(abbreviate_feature)

# -----------------------------
# Step 2: Omics Composition (filtered)
# -----------------------------
omics_counts = feature_df_filtered.sort_values('abs_coef', ascending=False).head(250)['omics'].value_counts()

# -----------------------------
# Step 3: Correlation Heatmap
# -----------------------------
X_corr = X_test_final[top_plot['feature'].tolist()]
X_corr_z = pd.DataFrame(StandardScaler().fit_transform(X_corr), columns=top_plot['short_name'])

# -----------------------------
# Step 4: Pathway Enrichment (on filtered RNA only)
# -----------------------------
rna_genes = feature_df_filtered[feature_df_filtered['feature'].str.startswith('rna_')]
top_rna = (
    rna_genes
    .sort_values(by='abs_coef', ascending=False)
    .head(enrich_n)['feature']
    .str.replace('rna_', '', regex=False)
    .tolist()
)

enr = gp.enrichr(
    gene_list=top_rna,
    gene_sets='KEGG_2021_Human',
    organism='Human',
    outdir=None,
    cutoff=0.05
)
top_results = enr.results.sort_values('Adjusted P-value').head(10)
top_results['Term'] = top_results['Term'].apply(lambda x: x[:35] + '...' if len(x) > 38 else x)

# -----------------------------
# Step 5: Plot All Panels
# -----------------------------
sns.set_context("paper", font_scale=0.8)
plt.rcParams.update({'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7})

fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [1, 2]})

# --- Panel 1: Coefficient Bar Plot ---
sns.barplot(
    data=top_plot,
    y='short_name', x='abs_coef',
    hue='omics',
    dodge=False, ax=axes[0, 0]
)
axes[0, 0].set_title('Top Feature Coefficients (No Clinical)')
axes[0, 0].set_xlabel('Abs. Coefficient')
axes[0, 0].set_ylabel('Feature')
axes[0, 0].legend(title='Omics Layer', loc='lower right', fontsize=7, title_fontsize=8)
axes[0, 0].invert_yaxis()

# --- Panel 2: Omics Pie Chart ---
axes[0, 1].pie(
    omics_counts.values,
    labels=omics_counts.index,
    autopct='%1.1f%%',
    colors=sns.color_palette('Set3'),
    startangle=140,
    textprops={'fontsize': 8}
)
axes[0, 1].set_title('Omics Composition (Top 250 Non-Clinical)')

# --- Panel 3: Correlation Heatmap ---
sns.heatmap(
    X_corr_z.corr(), ax=axes[1, 0],
    cmap='coolwarm', center=0,
    cbar_kws={'label': 'Pearson Correlation'},
    xticklabels=True, yticklabels=True
)
axes[1, 0].set_title('Top Feature Correlation (Non-Clinical)')
axes[1, 0].tick_params(axis='x', labelrotation=90, labelsize=6)
axes[1, 0].tick_params(axis='y', labelsize=6)

# --- Panel 4: Pathway Enrichment ---
sns.scatterplot(
    data=top_results,
    x='Combined Score',
    y='Term',
    size='Overlap',
    hue='Adjusted P-value',
    palette='coolwarm_r',
    sizes=(40, 300),
    edgecolor='black',
    ax=axes[1, 1]
)
axes[1, 1].set_title('Pathway Enrichment (Top RNA)')
axes[1, 1].set_xlabel('Combined Score')
axes[1, 1].set_ylabel('KEGG Pathway')
axes[1, 1].legend(title='Adj. P-value', fontsize=7, title_fontsize=8, loc='lower right')

# Save and Show
plt.tight_layout()
plt.savefig("feature_analysis_summary_no_clinical.png", dpi=300)
plt.show()

Top Multi-Omics Feature Distributions by Risk Group
# -----------------------------
# Top Multi-Omics Feature Distributions by Risk Group
# -----------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define feature mappings (from functional annotations)
matched_features = {
    'DAZ1': 'rna_DAZ1',
    'PIWIL1': 'rna_PIWIL1', 
    'CST2': 'rna_CST2',
    'PRAMEF10': 'rna_PRAMEF10',
    'FOXE1': 'rna_FOXE1',
    'CLDN7': 'prot_CLAUDIN7',
    'SQSTM1_P62_Ligand': 'prot_P62LCKLIGAND',
    'C_MYC_Pathway': 'pathway_c_myc_pathway',
    'Glypican3_Network': 'pathway_glypican_3_network',
    'GTF2IP12_CNV': 'cnv_GTF2IP12',
}

# Prepare a DataFrame with selected features and risk labels
selected_columns = list(matched_features.values())
df_selected = df_final[selected_columns + ['early_progression_label']].copy()

# Melt the data for seaborn
df_long = df_selected.melt(id_vars='early_progression_label', 
                           var_name='Feature', 
                           value_name='Value')

# Label risk groups
df_long['Risk Group'] = df_long['early_progression_label'].map({0: 'Low Risk', 1: 'High Risk'})

# Set font sizes
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

# Plot using seaborn
plt.figure(figsize=(18, 10))
sns.boxplot(data=df_long, x='Feature', y='Value', hue='Risk Group')
plt.xticks(rotation=45, ha='right')
plt.title('Top Multi-Omics Feature Distributions by Risk Group', pad=20)
plt.xlabel('Feature', labelpad=15)
plt.ylabel('Value', labelpad=15)
plt.grid(True)
plt.legend(title='Risk Group', title_fontsize=12)
plt.tight_layout()
plt.savefig("feature_distribution.png", dpi=300)
plt.show()

