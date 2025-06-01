from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import shap

df_training = pd.read_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/final_training_dataset.csv')


### SHAP top features ###

X = df_training.drop(columns=['sample', 'early_progression_label'])
X.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_') for col in X.columns]
y = df_training['early_progression_label']

# Step 1: Train base model to get SHAP feature ranking
base_model = LGBMClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    random_state=42
)
base_model.fit(X, y)

# Step 2: SHAP feature ranking
explainer = shap.Explainer(base_model)
shap_values = explainer(X)
mean_shap = np.abs(shap_values.values).mean(axis=0)
shap_ranking = pd.Series(mean_shap, index=X.columns).sort_values(ascending=False)

top_n = 300
top_features = shap_ranking.head(top_n).index.tolist()
X_top = X[top_features]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_top, y, stratify=y, test_size=0.2, random_state=42)

# Step 5: Apply SMOTE to training set
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)


# Step 6: Train LightGBM model on SMOTE-balanced data
model = LGBMClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    # publish both results with and without
    scale_pos_weight = (y == 0).sum() / (y == 1).sum(), 
    random_state=42
)
model.fit(X_train_bal, y_train_bal)

# Step 7: Predict probabilities on test set
y_prob = model.predict_proba(X_test)[:, 1]

# Step 7b: Use PR curve to find best threshold by F1
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
best_idx = np.argmax(f1)
best_threshold = thresholds[best_idx]
print(f"Best threshold: {best_threshold:.2f}")

# Apply optimal threshold
y_pred = (y_prob >= best_threshold).astype(int)


# Step 8: Evaluate
print("AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))



#### --- XG Boost -- ###

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report
import numpy as np

# Step 1: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_top, y, stratify=y, test_size=0.2, random_state=42
)

# Step 2: SMOTE for class balancing
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Step 3: Define and train XGBoost model
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    scale_pos_weight=(y_train_bal == 0).sum() / (y_train_bal == 1).sum(),  # optional
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42
)

xgb_model.fit(X_train_bal, y_train_bal)

# Step 4: Predict probabilities
y_prob = xgb_model.predict_proba(X_test)[:, 1]

# Step 5: Tune threshold using PR curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
best_idx = np.argmax(f1)
best_threshold = thresholds[best_idx]
print(f"Best threshold: {best_threshold:.2f}")

# Step 6: Final prediction and evaluation
y_pred = (y_prob >= best_threshold).astype(int)

print("AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

#### -- Linear Regression --- ###

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Optional: scale features (required for ElasticNet, helps Logistic)
scaler = StandardScaler()

# -----------------------
# 1. Logistic Regression (L2 only)
logreg = make_pipeline(
    scaler,
    LogisticRegression(penalty='l2', class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)
)
logreg.fit(X_train_bal, y_train_bal)
y_prob_log = logreg.predict_proba(X_test)[:, 1]

# -----------------------
# 2. ElasticNet (L1 + L2)
elastic = make_pipeline(
    scaler,
    LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5,
                       class_weight='balanced', max_iter=2000, random_state=42)
)
elastic.fit(X_train_bal, y_train_bal)
y_prob_elastic = elastic.predict_proba(X_test)[:, 1]

# -----------------------
# Step 3: Pick best threshold by F1 (PR curve) — apply to both
def tune_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_idx = f1.argmax()
    return thresholds[best_idx]

threshold_log = tune_threshold(y_test, y_prob_log)
threshold_elastic = tune_threshold(y_test, y_prob_elastic)

# -----------------------
# Step 4: Evaluate
print("\n🔎 Logistic Regression:")
print("AUC:", roc_auc_score(y_test, y_prob_log))
print(classification_report(y_test, (y_prob_log >= threshold_log).astype(int)))

print("\n🔎 ElasticNet:")
print("AUC:", roc_auc_score(y_test, y_prob_elastic))
print(classification_report(y_test, (y_prob_elastic >= threshold_elastic).astype(int)))


