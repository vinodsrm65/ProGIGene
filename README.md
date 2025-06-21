# ProGIGene
## 🧬 ProGIGene RNA Model Validation on GSE62452

This analysis validates a trained RNA-based prognostic model (**ProGIGene**) using the public gene expression dataset **[GSE62452](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE62452)**, which consists of pancreatic ductal adenocarcinoma (PDAC) samples.

---

### 📦 Dataset Information

- **Dataset ID**: GSE62452  
- **Platform**: Affymetrix Human Gene 1.0 ST Array (GPL6244)  
- **Cancer Type**: Pancreatic Ductal Adenocarcinoma (PDAC)  
- **Sample Count**: 130 total samples  
- **Tumor Samples with Valid DFS Data**: 65

---

### 📊 Model Prediction and Survival Stratification

### Kaplan–Meier Curve

![KM Plot](ProGIGene_GSE62452_KM.png)

---

### 📌 Summary Output

- `geo_expr` shape: **(130, 23605)**
- `geo_expr` columns (sample): Index(['---', '6Q27', '730394', 'A1BG', 'A1CF', 'A2M', 'A2ML1', 'A3GALT2',
       'A4GALT', 'A4GNT'],
      dtype='object', name='gene_symbol')
- ✅ Extracted DFS data for **65 tumor samples**
- ✅ Merged data: **65 samples**
- 🎯 **AUC (DFS prediction)**: **0.603**

---

### 📂 Files

- `ProGIGene_GSE62452_KM.png`: Kaplan–Meier survival curve comparing high vs. low risk groups
- `pancreatic_paad.py`: Script used to perform preprocessing, prediction, and survival analysis

---

### 🧠 Notes

- Risk stratification threshold was selected based on grid search for optimal separation (lowest log-rank p-value).
- AUC reflects moderate predictive performance and can potentially improve with additional omics integration or model tuning.

---

### 🎯 Final Metric

**AUC (Disease-Free Survival prediction)**: `0.603`

---

### 📎 Citation



