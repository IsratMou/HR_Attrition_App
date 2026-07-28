# 💼 IT Employee Attrition Predictor

## 📖 Project Overview

This is an end-to-end Machine Learning project that predicts whether an IT employee is likely to leave the company (Attrition). By identifying flight risks early, HR departments can take proactive measures to improve retention, saving the company time and recruitment costs.

🌐 Live Demo
Click the link below to try the live deployed app:https://hr-attrition-app123.streamlit.app/

## 🧠 Business Problem

IT companies spend a significant amount of money recruiting and training software engineers. When an employee leaves unexpectedly, it costs the company time and money. This model aims to predict attrition based on employee demographics, job satisfaction, and compensation metrics.

## 🛠️ Tech Stack

- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Random Forest Classifier)
- **Imbalanced Data Handling:** Imbalanced-Learn (SMOTE)
- **Web Framework:** Streamlit

## 📊 Dataset

The dataset used is the **IBM HR Analytics Employee Attrition & Performance** dataset from Kaggle. It contains 1,470 employee records with 35 features (Age, Department, MonthlyIncome, OverTime, etc.).

## ⚙️ Machine Learning Pipeline

1. **Data Loading:** Loaded IBM HR Analytics dataset using Pandas.
2. **Data Preprocessing:**
   - Dropped useless columns (EmployeeCount, StandardHours, etc.).
   - Converted target variable (`Attrition`) to binary (Yes=1, No=0).
   - Handled categorical variables using One-Hot Encoding (`pd.get_dummies`).
3. **Handling Imbalanced Data:**
   - The dataset was heavily imbalanced (84% Stay, 16% Leave).
   - Used **SMOTE** (Synthetic Minority Over-sampling Technique) on the training data to balance the classes, which improved the model's Recall and F1-score for predicting attrition.
4. **Modeling:** Trained a Random Forest Classifier.
5. **Evaluation:** Evaluated using Accuracy, Precision, Recall, and F1-Score.
6. **Feature Importance:** Extracted top drivers of attrition (StockOptionLevel, MonthlyIncome, JobSatisfaction).
7. **Deployment:** Built an interactive web UI using Streamlit to take user inputs and output live predictions.

## ▶️ How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/IsratMou/HR_Attrition_App.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open your browser to `http://localhost:8501`.
