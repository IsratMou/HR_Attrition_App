import streamlit as st
import pandas as pd
import joblib

# Load the model and columns
model = joblib.load('attrition_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("💼 IT Employee Attrition Predictor")

st.write("Enter employee details to predict if they are at risk of leaving.")

# Create some input fields
age = st.slider("Age", 18, 60, 30)
monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000)
job_satisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
overtime = st.selectbox("OverTime?", ["Yes", "No"])

# Put all inputs into a dictionary (matching your training data)
input_data = {
    'Age': age,
    'MonthlyIncome': monthly_income,
    'JobSatisfaction': job_satisfaction,
    'OverTime': overtime,
    # For a real app, you'd add all the other required fields here...
    'BusinessTravel': 'Travel_Rarely',
    'Department': 'Research & Development',
    'DistanceFromHome': 5,
    'Education': 3,
    'EducationField': 'Life Sciences',
    'EnvironmentSatisfaction': 3,
    'Gender': 'Male',
    'HourlyRate': 80,
    'JobInvolvement': 3,
    'JobLevel': 2,
    'JobRole': 'Research Scientist',
    'MaritalStatus': 'Single',
    'MonthlyRate': 15000,
    'NumCompaniesWorked': 1,
    'PercentSalaryHike': 15,
    'PerformanceRating': 3,
    'RelationshipSatisfaction': 3,
    'StockOptionLevel': 1,
    'TotalWorkingYears': 5,
    'TrainingTimesLastYear': 3,
    'WorkLifeBalance': 3,
    'YearsAtCompany': 3,
    'YearsInCurrentRole': 2,
    'YearsSinceLastPromotion': 1,
    'YearsWithCurrManager': 2
}

if st.button("Predict Attrition"):
    # Convert to DataFrame
    df_new = pd.DataFrame([input_data])
    
    # One-Hot Encode
    df_new_encoded = pd.get_dummies(df_new, drop_first=True)
    
    # Align columns
    df_new_aligned = df_new_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Predict
    prediction = model.predict(df_new_aligned)
    
    if prediction[0] == 1:
        st.error("🚨 WARNING: This employee is highly likely to LEAVE the company!")
    else:
        st.success("✅ SAFE: This employee is likely to STAY.")