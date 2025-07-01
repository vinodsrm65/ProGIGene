

# 🧬 ProGIGene: Multi-Omics Risk Stratification Pipeline

**ProGIGene** is a machine learning pipeline for **progression risk modeling in early-stage gastrointestinal (GI) cancers**, integrating multi-omics features including transcriptomics, DNA methylation, copy number variations, somatic mutations, and proteomics. It implements a transparent and interpretable two-stage modeling strategy using survival analysis and ElasticNet classification.

---

## 🚀 Features

- Preprocessing of multi-omics data with variance filtering and standardization  
- Cox proportional hazards survival modeling (time-to-event analysis)  
- Binary risk label creation from continuous survival risk scores  
- ElasticNet-regularized logistic regression for feature selection and classification  
- Automated top-250 feature extraction based on coefficient magnitude  
- Performance evaluation with precision–recall and permutation testing  
- Threshold tuning to balance precision and recall  
- SHAP-ready feature outputs for model explainability

---

## 🛠️ Tech Stack

- Python 3.8+
- Pandas, NumPy
- Scikit-learn
- LightGBM
- imbalanced-learn (SMOTE)
- scikit-survival
- SHAP

---

## 📁 Dataset

Place a file named `final_training_dataset.csv` in the root directory. The dataset must include:
- `sample`: unique sample ID  
- `early_progression_label`: binary progression label  
- `pfi`: progression-free interval event indicator  
- `pfi_time`: progression-free survival time (in days)  
- Molecular features across omics platforms

---

## ⚙️ Pipeline Overview

```text
1. Load and sanitize feature names
2. Split data into train/test (stratified)
3. Apply variance threshold filtering
4. Standardize features with z-score scaling
5. Train CoxPH survival model → generate risk scores
6. Convert risk scores into binary labels (top 33% high-risk)
7. Use ElasticNet logistic regression to select top features
8. Re-train classifier on top features
9. Tune threshold for best F1 score
10. Validate robustness via permutation testing
```

---

## 📈 Example Metrics

- **C-index**: ~0.82  
- **AUC (ROC)**: ~0.81  
- **Precision/Recall tuning**: enables flexible sensitivity/specificity balancing  
- **Permutation testing**: verifies statistical significance beyond chance

---

## 🧪 How to Run

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Or import the code in a notebook for step-by-step execution and visualization.

---

## 🧠 Example Snippet

```python
from sksurv.linear_model import CoxPHSurvivalAnalysis
cox_model = CoxPHSurvivalAnalysis(alpha=0.01)
cox_model.fit(X_train_scaled, y_train_surv)
risk_scores = cox_model.predict(X_test_scaled)
```

---

## 📊 Output

- Predicted probabilities for early progression
- Risk stratification thresholds
- Precision-recall curves
- SHAP-ready feature importance values (optional)

