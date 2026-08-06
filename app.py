import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------
# Load model artifacts (produced at the end of the notebook)
# ----------------------------


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load('attrition_pipeline.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    feature_names = joblib.load('feature_names.pkl')
    threshold = joblib.load('decision_threshold.pkl')
    metadata = joblib.load('feature_metadata.pkl')
    return pipeline, explainer, feature_names, threshold, metadata


pipeline, explainer, feature_names, DECISION_THRESHOLD, metadata = load_artifacts()

st.set_page_config(page_title="Employee Attrition Predictor",
                   page_icon="💼", layout="wide")
st.title("💼 Employee Attrition Risk Predictor")
st.write("Enter employee details below to estimate attrition risk and see the key drivers behind the prediction.")

# ----------------------------
# Input form, organized by section (real inputs for every model feature)
# ----------------------------
with st.form("employee_form"):

    st.subheader("Employee Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox(
            "Marital Status", ["Single", "Married", "Divorced"])
    with col2:
        education = st.selectbox("Education Level (1=Below College, 5=Doctorate)", [
                                 1, 2, 3, 4, 5], index=2)
        education_field = st.selectbox(
            "Education Field",
            ["Life Sciences", "Other", "Medical", "Marketing",
                "Technical Degree", "Human Resources"]
        )
        distance_from_home = st.slider("Distance From Home (km)", 1, 29, 9)
    with col3:
        department = st.selectbox(
            "Department", ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox(
            "Job Role",
            ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
             "Healthcare Representative", "Manager", "Sales Representative", "Research Director",
             "Human Resources"]
        )
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)

    st.subheader("Compensation & Work Arrangement")
    col1, col2, col3 = st.columns(3)
    with col1:
        monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 4900)
        monthly_rate = st.slider("Monthly Rate ($)", 2000, 27000, 14300)
        daily_rate = st.slider("Daily Rate ($)", 100, 1500, 800)
    with col2:
        hourly_rate = st.slider("Hourly Rate ($)", 30, 100, 65)
        percent_salary_hike = st.slider("Percent Salary Hike (%)", 11, 25, 15)
        stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    with col3:
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        business_travel = st.selectbox(
            "Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        num_companies_worked = st.slider("Number of Companies Worked", 0, 9, 2)

    st.subheader("Tenure & History")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_working_years = st.slider("Total Working Years", 0, 40, 10)
        years_at_company = st.slider("Years At Company", 0, 40, 5)
        years_in_current_role = st.slider("Years In Current Role", 0, 18, 3)
    with col2:
        years_since_last_promotion = st.slider(
            "Years Since Last Promotion", 0, 15, 1)
        years_with_curr_manager = st.slider(
            "Years With Current Manager", 0, 17, 3)
        training_times_last_year = st.slider(
            "Training Times Last Year", 0, 6, 3)
    with col3:
        performance_rating = st.selectbox(
            "Performance Rating (3=Excellent, 4=Outstanding)", [3, 4])
        job_involvement = st.selectbox(
            "Job Involvement (1=Low, 4=High)", [1, 2, 3, 4], index=2)

    st.subheader("Satisfaction")
    col1, col2 = st.columns(2)
    with col1:
        job_satisfaction = st.selectbox(
            "Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4], index=2)
        environment_satisfaction = st.selectbox(
            "Environment Satisfaction (1=Low, 4=High)", [1, 2, 3, 4], index=2)
    with col2:
        relationship_satisfaction = st.selectbox(
            "Relationship Satisfaction (1=Low, 4=High)", [1, 2, 3, 4], index=2)
        work_life_balance = st.selectbox(
            "Work Life Balance (1=Bad, 4=Best)", [1, 2, 3, 4], index=2)

    submitted = st.form_submit_button("Predict Attrition Risk")

# ----------------------------
# Prediction + explanation
# ----------------------------
if submitted:
    input_data = {
        'Age': age, 'BusinessTravel': business_travel, 'DailyRate': daily_rate,
        'Department': department, 'DistanceFromHome': distance_from_home, 'Education': education,
        'EducationField': education_field, 'EnvironmentSatisfaction': environment_satisfaction,
        'Gender': gender, 'HourlyRate': hourly_rate, 'JobInvolvement': job_involvement,
        'JobLevel': job_level, 'JobRole': job_role, 'JobSatisfaction': job_satisfaction,
        'MaritalStatus': marital_status, 'MonthlyIncome': monthly_income, 'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_companies_worked, 'OverTime': overtime,
        'PercentSalaryHike': percent_salary_hike, 'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction, 'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years, 'TrainingTimesLastYear': training_times_last_year,
        'WorkLifeBalance': work_life_balance, 'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role, 'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager
    }

    # Match training column order exactly — the pipeline's ColumnTransformer expects this
    emp_df = pd.DataFrame([input_data])[metadata['all_raw_cols']]

    # Predict probability using the full pipeline (all preprocessing happens inside it)
    risk_prob = pipeline.predict_proba(emp_df)[0, 1]

    st.divider()
    st.subheader("Prediction Result")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Attrition Risk Probability", f"{risk_prob:.1%}")
        st.caption(f"Threshold for flagging risk: {DECISION_THRESHOLD:.0%}")
        # High-risk cutoff is defined relative to the tuned decision threshold, not a fixed
        # number — otherwise a hardcoded value can end up below the threshold itself
        high_risk_cutoff = min(DECISION_THRESHOLD + 0.2, 0.95)
        if risk_prob >= high_risk_cutoff:
            st.error("🔴 High Risk")
        elif risk_prob >= DECISION_THRESHOLD:
            st.warning("🟡 Medium Risk")
        else:
            st.success("🟢 Low Risk")
        st.progress(min(float(risk_prob), 1.0))

    # SHAP explanation for this specific employee
    # LinearExplainer returns a plain 2D array (n_samples, n_features) — no per-class
    # list/3D handling needed here, unlike TreeExplainer used with the earlier RF model
    emp_transformed = pipeline.named_steps['preprocessor'].transform(emp_df)
    emp_shap = explainer.shap_values(emp_transformed)

    shap_series = pd.Series(emp_shap[0], index=feature_names).sort_values(
        key=abs, ascending=False)
    top_factors = shap_series.head(6)

    with col2:
        st.write("**Top factors behind this prediction**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ['#d62728' if v >
                  0 else '#1f77b4' for v in top_factors.values[::-1]]
        ax.barh(top_factors.index[::-1],
                top_factors.values[::-1], color=colors)
        ax.set_xlabel("SHAP value (pushes toward Leave \u2192)")
        st.pyplot(fig)
        st.caption(
            "Red = pushes toward higher risk \u00b7 Blue = pushes toward lower risk")

    # Factor-driven retention suggestions — only shown when they're actually top risk drivers
    # for THIS employee, not a static generic list
    risk_drivers = top_factors[top_factors > 0].index.tolist()
    suggestions = []
    if any('OverTime' in f for f in risk_drivers):
        suggestions.append(
            "Review workload — this employee is working overtime, a strong attrition driver.")
    if 'JobSatisfaction' in risk_drivers:
        suggestions.append(
            "Schedule a check-in on job satisfaction and role fit.")
    if 'MonthlyIncome' in risk_drivers:
        suggestions.append("Review compensation relative to role and market.")
    if 'StockOptionLevel' in risk_drivers:
        suggestions.append(
            "Consider offering or increasing stock option level.")
    if any('BusinessTravel' in f for f in risk_drivers):
        suggestions.append("Reassess travel demands for this role.")
    if any('WorkLifeBalance' in f or 'EnvironmentSatisfaction' in f for f in risk_drivers):
        suggestions.append(
            "Check in on work-life balance and team environment.")
    if any('JobRole' in f for f in risk_drivers):
        suggestions.append(
            "Role-specific factors are contributing — consider a role-focused retention conversation.")

    if suggestions:
        st.write("**Suggested retention actions**")
        for s in suggestions:
            st.write(f"- {s}")

    st.caption(
        f"Decision threshold: {DECISION_THRESHOLD:.2f} (tuned on held-out test data to maximize F1 for the "
        "minority 'Leave' class, rather than the default 0.5 — this model is trained on SMOTE-balanced data, "
        "so raw probabilities are not calibrated to the true ~16% real-world attrition rate; ranking between "
        "employees is still meaningful)."
    )
