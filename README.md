
# The Repository contains 
---

## Repository Structure

- `Fraud Detection.ipynb` — main notebook with EDA, preprocessing, modeling, evaluation, and visualizations  
- `README.md` — project README 
- `data/` - contains data (not uploaded due to confidentiality purposes and size of data)

---

## Dataset

- **Source:** Provided by IDFC First Bank

---

## Notebook Walkthrough

- **Section 1 — Data Loading:** load dataset from local `data/` or cloud storage  
- **Section 2 — EDA:** class imbalance checks, distributions, correlation heatmap  
- **Section 3 — Preprocessing:** missing-value handling, scaling, encoding, outlier handling  
- **Section 4 — Resampling:** SMOTE / undersampling / class-weight strategies  
- **Section 5 — Modeling:** baseline and advanced learners, hyperparameter tuning  
- **Section 6 — Evaluation:** ROC, confusion matrix, threshold selection  
- **Section 7 — Explainability:** SHAP or feature importance visualizations  
- **Section 8 — Conclusions & Next Steps**

---

## Data Processing Pipeline

- Load raw CSV/parquet into `pandas.DataFrame`  
- Inspect and log basic stats (nulls, unique counts)  
- Clean: drop duplicates, handle missing values, cap outliers  
- Transform: scale numeric features (StandardScaler/RobustScaler), log-transform skewed features if needed  
- Split: stratified train/validation/test to preserve class ratios  
- Balance: apply SMOTE or class-weighted models depending on approach

Example:
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)
```

---

## Modeling Approach

- Baseline: Logistic Regression with class weighting  
- Tree-based: Random Forest, XGBoost, LightGBM (good defaults for tabular data)  
- Ensembles: stacking or blending complementary models  
- Neural nets: small dense networks if dataset size/complexity warrants it  
- Hyperparameter tuning: `RandomizedSearchCV` or `GridSearchCV` with `StratifiedKFold`  
- Use `sklearn.pipeline.Pipeline` to combine preprocessing and modeling

Example pipeline:
```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

clf = Pipeline([
  ('scaler', StandardScaler()),
  ('model', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
])
clf.fit(X_train, y_train)
```

---

## Evaluation & Metrics

Focus on metrics that reflect imbalanced classification:

- Primary: ROC-AUC, PR-AUC  
- Operational: Precision, Recall, F1, confusion matrix at chosen thresholds  
- Business KPIs: False Positive Rate (cost of manual review) vs False Negative Rate (missed fraud)  
- Calibration: Brier score, reliability diagrams

Useful plots: ROC curve, Precision-Recall curve, confusion matrix, precision/recall vs threshold.

---

## Results 

- Best model: XGBoost (with ROC Thresholding)  
- ROC-AUC (test): 99.904%  
- Precision / Recall / F1 at chosen threshold: 92.039% / 92.50% / 92.27% 
- Notes: A balanced Precision and Recall shows a more credible financial decision.

---

## Explainability & Analysis

- Use `shap` for global and local explanations (feature importance, dependence plots)  
- Perform error analysis on false positives/false negatives to identify blind spots  
- Validate feature importance with permutation importance and SHAP

---

## Deployment & Inference (Optional)

- Batch scoring: periodic scoring (cron, Airflow)  
- Real-time scoring: FastAPI service exposing a `/score` endpoint with input validation and feature extraction  
- Monitoring: track prediction drift, feature drift, and performance metrics after deployment  
- Retraining: schedule periodic retraining or trigger on detected drift

Example FastAPI skeleton:
```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/score")
def score(payload: dict):
    features = preprocess_for_model(payload)
    prob = model.predict_proba([features])[0, 1]
    return {"fraud_score": float(prob)}
```

---




## References
  
- Scikit-learn docs — `https://scikit-learn.org`  
- Imbalanced-learn docs — `https://imbalanced-learn.org`  
- SHAP docs — `https://shap.readthedocs.io`

---

## License

This project is provided under the MIT License. See `LICENSE` for details.

---