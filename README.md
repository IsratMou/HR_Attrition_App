# 💼 Employee Attrition Risk Predictor

## Overview

An end-to-end machine learning system that predicts an employee's risk of attrition (leaving the company) and explains _why_, using SHAP. Built on the IBM HR Analytics dataset, deployed as an interactive Streamlit app.

Unlike a simple Stay/Leave classifier, this project focuses on three things recruiters and HR teams actually need: a **calibrated risk score**, a **tuned decision threshold** that reflects the real cost of missing a flight risk, and a **per-employee explanation** of which factors are driving the prediction.

🌐 **Live demo:** _update this link after redeploying_

## Business Problem

Replacing an employee costs significant time and money in recruiting, onboarding, and lost productivity. A model that flags at-risk employees early — with a clear, explainable reason — lets HR intervene before a resignation, not after.

## Dataset

**IBM HR Analytics Employee Attrition & Performance** (Kaggle), 1,470 employee records, 35 original features. Target variable `Attrition` (Yes/No), heavily imbalanced: **1,233 stayed vs. 237 left (~84% / 16%)**.

Four columns were dropped before modeling — confirmed via `nunique()` to carry zero predictive value:

- `EmployeeCount`, `StandardHours`, `Over18` — constant across all rows
- `EmployeeNumber` — a row identifier, not a feature

## Tech Stack

- **Language:** Python
- **Data manipulation:** Pandas, NumPy
- **Machine learning:** Scikit-learn (`Pipeline`, `ColumnTransformer`, `RandomForestClassifier`)
- **Imbalanced data:** imbalanced-learn (SMOTE, applied correctly — training folds only)
- **Explainability:** SHAP (`TreeExplainer`)
- **Web framework:** Streamlit

## ML Pipeline

1. **Preprocessing:** dropped uninformative columns, encoded target as binary, split into train/test (80/20, **stratified** to preserve the 84/16 ratio in both sets).
2. **Encoding:** `ColumnTransformer` + `OneHotEncoder(handle_unknown='ignore')` for the 7 categorical features, numeric features passed through unchanged — fit only on training data, wrapped inside the pipeline (no manual `pd.get_dummies` + column-reindexing at inference time).
3. **Class imbalance:** SMOTE applied **inside an `imblearn.Pipeline`**, so oversampling only happens during training and never leaks into evaluation or live predictions.
4. **Model:** Random Forest Classifier.
5. **Evaluation:** 5-fold stratified cross-validation (not a single train/test split) plus a held-out test set, using precision/recall/F1/ROC-AUC — accuracy alone is a misleading metric on an 84/16-imbalanced target.
6. **Threshold tuning:** the default 0.5 decision threshold was replaced with a threshold chosen from the precision-recall curve on the test set (see below).
7. **Explainability:** SHAP `TreeExplainer` for both global feature importance and per-employee explanations.
8. **Deployment:** the entire pipeline (preprocessing + SMOTE + model) is saved as a single object and loaded directly in the Streamlit app — no encoding logic duplicated in the app layer.

## Model Performance

**5-fold cross-validation** (training set, "Leave" class):

| Metric | Score         |
| ------ | ------------- |
| F1     | 0.383 ± 0.140 |
| Recall | 0.263 ± 0.114 |

The wide standard deviation reflects the small number of positive cases (237 "Leave" examples total) — a known limitation of this dataset, not the model.

**Held-out test set** (default 0.5 threshold):

| Class     | Precision | Recall | F1   |
| --------- | --------- | ------ | ---- |
| Stay (0)  | 0.87      | 0.98   | 0.92 |
| Leave (1) | 0.71      | 0.21   | 0.33 |

ROC-AUC: **0.775** — the model separates risk levels reasonably well, but the default threshold is far too conservative for this use case.

**Tuned decision threshold: 0.30** (chosen by maximizing F1 on the precision-recall curve):

| Class     | Precision | Recall | F1   |
| --------- | --------- | ------ | ---- |
| Stay (0)  | 0.91      | 0.85   | 0.88 |
| Leave (1) | 0.43      | 0.57   | 0.49 |

**Why 0.30 and not 0.5:** at the default threshold, the model only catches 21% of employees who actually leave. For a retention tool, a missed flight risk (false negative) is more costly than an unnecessary check-in conversation (false positive) — so the threshold was deliberately tuned to trade some precision for meaningfully higher recall (57%).

**Important caveat:** because training data is SMOTE-balanced (~50/50), raw predicted probabilities are not calibrated to the true ~16% real-world attrition rate — a "high risk" score should be read as a relative ranking between employees, not a literal frequency.

## What Actually Drives Attrition (Feature Importance)

Top features by Random Forest importance, consistent with both correlation analysis and SHAP:

| Feature                    | Importance |
| -------------------------- | ---------- |
| OverTime = Yes             | 0.098      |
| Marital Status = Single    | 0.087      |
| Stock Option Level         | 0.060      |
| Marital Status = Married   | 0.044      |
| Monthly Income             | 0.039      |
| Job Satisfaction           | 0.032      |
| Years With Current Manager | 0.029      |

Numeric correlation with attrition (independent check) confirms the same pattern from a different angle — the strongest negative correlations are `TotalWorkingYears` (-0.171), `JobLevel` (-0.169), `YearsInCurrentRole` (-0.161), `MonthlyIncome` (-0.160), and `Age` (-0.159): **younger, less senior, less tenured, lower-income employees are the primary attrition risk**, amplified sharply by overtime.

## Explainability (SHAP)

Beyond global feature importance, the app generates a **per-employee SHAP explanation** for every prediction — showing exactly which factors pushed that individual's risk score up or down, not just which features matter on average across the dataset. This is what turns the model from "a black-box score" into something an HR user can act on and an interviewer can interrogate.

## App Features

- Full input form covering all 30 model features, organized into logical sections (Employee Info, Compensation & Work, Tenure & History, Satisfaction)
- Attrition risk probability with a 🟢🟡🔴 risk level indicator, using the tuned 0.30 threshold
- Live SHAP bar chart of the top factors behind each individual prediction
- Retention suggestions generated dynamically from that employee's actual top risk drivers — not a static generic list

## How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/IsratMou/HR_Attrition_App.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Open `http://localhost:8501`.

## Known Limitations

- Small positive class (237 of 1,470 records) limits how stable cross-validation scores are across folds.
- SMOTE-balanced training probabilities are not calibrated to the real-world base rate — treat scores as relative ranking, not literal likelihood.
- Trained on a single company's historical snapshot; feature relationships (e.g. income thresholds, role-specific patterns) may not generalize to other organizations without retraining.

## Future Work

- Compare Random Forest against Logistic Regression / XGBoost with the same pipeline, to justify model choice quantitatively rather than by default
- Hyperparameter tuning (`GridSearchCV` / `Optuna`)
- Package as a FastAPI backend in addition to the Streamlit frontend
- Dockerize for reproducible deployment
- Add automated tests for the preprocessing and prediction logic
