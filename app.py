<<<<<<< HEAD
"""
Employee Attrition Risk Predictor — Enhanced UI
================================================
A polished Streamlit interface for predicting employee attrition risk
with explainable AI (SHAP) driver breakdown.

Author: Israt Jahan Mou
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import traceback
from charts import make_risk_gauge, make_shap_chart

# ============================================================
# PAGE CONFIG — must run before any other st command
# ============================================================
st.set_page_config(
    page_title="Employee Attrition Risk Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM THEME — professional HR palette
# ============================================================
# Palette: Deep Sea + Emerald/Coral semantic colors
COLOR_BG          = "#0B1C2C"
COLOR_PRIMARY     = "#1A365D"
COLOR_ACCENT      = "#2C7A7B"
COLOR_TEXT        = "#1A2330"
COLOR_MUTED       = "#64748B"
COLOR_LOW         = "#0EA5A4"   # teal — low risk
COLOR_MED         = "#F59E0B"   # amber — medium risk
COLOR_HIGH        = "#E11D48"   # rose — high risk
COLOR_PUSH_UP     = "#E11D48"   # red — pushes toward leave
COLOR_PUSH_DOWN   = "#0EA5A4"   # teal — pushes toward stay
COLOR_CARD_BG     = "#F8FAFC"
COLOR_CARD_BORDER = "#E2E8F0"

# Set matplotlib defaults for SHAP chart
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": COLOR_MUTED,
    "axes.linewidth": 0.8,
    "axes.labelcolor": COLOR_TEXT,
    "xtick.color": COLOR_MUTED,
    "ytick.color": COLOR_TEXT,
})

# ============================================================
# CUSTOM CSS — inject professional styling
# ============================================================
st.markdown(
    """
    <style>
    /* ---------- Hide default Streamlit chrome ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    /* ---------- Typography ---------- */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     "Inter", "Helvetica Neue", Arial, sans-serif;
        color: #1A2330;
    }

    /* ---------- App title banner ---------- */
    .app-banner {
        background: linear-gradient(135deg, #0B1C2C 0%, #1A365D 50%, #2C7A7B 100%);
        padding: 28px 32px;
        border-radius: 14px;
        margin-bottom: 18px;
        color: white;
        box-shadow: 0 10px 30px rgba(11, 28, 44, 0.18);
    }
    .app-banner h1 {
        margin: 0 0 6px 0;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.4px;
    }
    .app-banner p {
        margin: 0;
        font-size: 14px;
        opacity: 0.85;
        font-weight: 400;
    }
    .app-banner .pill {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 500;
        margin-right: 8px;
        letter-spacing: 0.3px;
    }

    /* ---------- Section headings ---------- */
    .section-title {
        font-size: 16px;
        font-weight: 600;
        color: #1A365D;
        margin: 18px 0 8px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #2C7A7B;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ---------- Cards ---------- */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .metric-card .label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 30px;
        font-weight: 700;
        color: #1A2330;
        line-height: 1.1;
    }

    /* ---------- Risk tier badge ---------- */
    .risk-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .risk-low    { background: rgba(14,165,164,0.12);  color: #0EA5A4; border: 1px solid rgba(14,165,164,0.3); }
    .risk-medium { background: rgba(245,158,11,0.12);  color: #B45309; border: 1px solid rgba(245,158,11,0.3); }
    .risk-high   { background: rgba(225,29,72,0.12);   color: #BE123C; border: 1px solid rgba(225,29,72,0.3); }

    /* ---------- Suggestion cards ---------- */
    .suggestion {
        background: #F8FAFC;
        border-left: 4px solid #2C7A7B;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
        font-size: 13px;
        color: #1A2330;
    }
    .suggestion strong { color: #1A365D; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    .sidebar-header {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748B;
        margin: 16px 0 8px 0;
    }
    .sidebar-stat {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .sidebar-stat .k { font-size: 11px; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; }
    .sidebar-stat .v { font-size: 16px; color: #1A365D; font-weight: 700; }

    /* ---------- Form controls ---------- */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #475569 !important;
        margin-bottom: 2px !important;
    }

    /* ---------- Predict button ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #1A365D 0%, #2C7A7B 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 12px rgba(44, 122, 123, 0.25);
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(44, 122, 123, 0.35);
    }

    /* ---------- Expanders ---------- */
    .streamlit-expanderHeader {
        background: #F8FAFC;
        font-weight: 600;
        font-size: 14px;
        color: #1A365D;
        border-radius: 8px !important;
    }
    details[data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        overflow: hidden;
        margin-bottom: 8px;
    }

    /* ---------- Footer ---------- */
    .app-footer {
        margin-top: 32px;
        padding: 18px 0;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        font-size: 12px;
        color: #94A3B8;
    }
    .app-footer a { color: #2C7A7B; text-decoration: none; }

    /* ---------- Info banner ---------- */
    .info-banner {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 4px solid #2C7A7B;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 13px;
        color: #1E3A5F;
        margin-bottom: 12px;
    }

    /* ---------- Divider ---------- */
    .app-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL ARTIFACTS (with graceful SHAP fallback)
# ============================================================
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("attrition_pipeline.pkl")
    feature_names = joblib.load("feature_names.pkl")
    threshold = joblib.load("decision_threshold.pkl")
    metadata = joblib.load("feature_metadata.pkl")

    explainer = None
    shap_error = None
    try:
        explainer = joblib.load("shap_explainer.pkl")
    except ModuleNotFoundError as e:
        shap_error = (
            f"SHAP explainer could not be loaded: missing module '{e.name}'. "
            "Predictions still work; per-employee SHAP explanations are disabled."
        )
    except Exception as e:
        shap_error = f"SHAP explainer could not be loaded: {e}"

    return pipeline, explainer, feature_names, threshold, metadata, shap_error


try:
    pipeline, explainer, feature_names, DECISION_THRESHOLD, metadata, shap_error = load_artifacts()
except Exception:
    st.error("Failed to load model artifacts — see error below.")
    st.code(traceback.format_exc())
    st.stop()


# ============================================================
# HEADER BANNER
# ============================================================
st.markdown(
    f"""
    <div class="app-banner">
        <h1>💼 Employee Attrition Risk Predictor</h1>
        <p>Estimate attrition probability for any employee profile and see the data-driven factors behind the prediction.</p>
        <div style="margin-top:10px;">
            <span class="pill">SCIKIT-LEARN PIPELINE</span>
            <span class="pill">SMOTE BALANCED</span>
            <span class="pill">SHAP EXPLAINABLE</span>
            <span class="pill">THRESHOLD {DECISION_THRESHOLD:.0%}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if shap_error:
    st.warning(shap_error)


# ============================================================
# SIDEBAR — Model context
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-header">Model Snapshot</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sidebar-stat"><div class="k">Algorithm</div><div class="v">Logistic Regression</div></div>
        <div class="sidebar-stat"><div class="k">Decision Threshold</div><div class="v">{DECISION_THRESHOLD:.2f}</div></div>
        <div class="sidebar-stat"><div class="k">Training Data</div><div class="v">IBM HR Analytics (1,470)</div></div>
        <div class="sidebar-stat"><div class="k">Features</div><div class="v">30</div></div>
        <div class="sidebar-stat"><div class="k">Balancing</div><div class="v">SMOTE (train only)</div></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-header">Risk Tiers</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sidebar-stat">
            <div class="k">Low Risk</div>
            <div class="v" style="color:#0EA5A4;">P &lt; {DECISION_THRESHOLD:.0%}</div>
        </div>
        <div class="sidebar-stat">
            <div class="k">Medium Risk</div>
            <div class="v" style="color:#B45309;">{DECISION_THRESHOLD:.0%} – {min(DECISION_THRESHOLD + 0.2, 0.95):.0%}</div>
        </div>
        <div class="sidebar-stat">
            <div class="k">High Risk</div>
            <div class="v" style="color:#BE123C;">P ≥ {min(DECISION_THRESHOLD + 0.2, 0.95):.0%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-header">About</div>', unsafe_allow_html=True)
    st.caption(
        "This model was trained on the IBM HR Analytics dataset. Probabilities "
        "are not calibrated to the real-world ~16% attrition rate but ranking "
        "across employees is meaningful."
    )


# ============================================================
# INPUT FORM
# ============================================================
st.markdown('<div class="section-title">📋 Employee Profile</div>', unsafe_allow_html=True)

with st.form("employee_form"):

    # --- Section 1: Employee Information ---
    with st.expander("👤 Employee Information", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 60, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        with c2:
            education = st.selectbox("Education Level (1=Below College, 5=Doctorate)", [1, 2, 3, 4, 5], index=2)
            education_field = st.selectbox("Education Field", [
                "Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"])
            distance_from_home = st.slider("Distance From Home (km)", 1, 29, 9)
        with c3:
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            job_role = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"])
            job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)

    # --- Section 2: Compensation & Work Arrangement ---
    with st.expander("💰 Compensation & Work Arrangement", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 4900)
            monthly_rate = st.slider("Monthly Rate ($)", 2000, 27000, 14300)
            daily_rate = st.slider("Daily Rate ($)", 100, 1500, 800)
        with c2:
            hourly_rate = st.slider("Hourly Rate ($)", 30, 100, 65)
            percent_salary_hike = st.slider("Percent Salary Hike (%)", 11, 25, 15)
            stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        with c3:
            overtime = st.selectbox("OverTime", ["Yes", "No"])
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            num_companies_worked = st.slider("Number of Companies Worked", 0, 9, 2)

    # --- Section 3: Tenure & History ---
    with st.expander("⏳ Tenure & History", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            total_working_years = st.slider("Total Working Years", 0, 40, 10)
            years_at_company = st.slider("Years At Company", 0, 40, 5)
            years_in_current_role = st.slider("Years In Current Role", 0, 18, 3)
        with c2:
            years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
            years_with_curr_manager = st.slider("Years With Current Manager", 0, 17, 3)
            training_times_last_year = st.slider("Training Times Last Year", 0, 6, 3)
        with c3:
            performance_rating = st.selectbox("Performance Rating (3=Excellent, 4=Outstanding)", [3, 4])
            job_involvement = st.selectbox("Job Involvement (1=Low, 4=High)", [1, 2, 3, 4], index=2)

    # --- Section 4: Satisfaction ---
    with st.expander("😊 Satisfaction", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            job_satisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4], index=2)
            environment_satisfaction = st.selectbox("Environment Satisfaction (1=Low, 4=High)", [1, 2, 3, 4], index=2)
        with c2:
            relationship_satisfaction = st.selectbox("Relationship Satisfaction (1=Low, 4=High)", [1, 2, 3, 4], index=2)
            work_life_balance = st.selectbox("Work Life Balance (1=Bad, 4=Best)", [1, 2, 3, 4], index=2)

    submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)


# ============================================================
# PREDICTION + EXPLANATION
# ============================================================
if submitted:
    input_data = {
        "Age": age, "BusinessTravel": business_travel, "DailyRate": daily_rate,
        "Department": department, "DistanceFromHome": distance_from_home, "Education": education,
        "EducationField": education_field, "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender, "HourlyRate": hourly_rate, "JobInvolvement": job_involvement,
        "JobLevel": job_level, "JobRole": job_role, "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status, "MonthlyIncome": monthly_income, "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked, "OverTime": overtime,
        "PercentSalaryHike": percent_salary_hike, "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction, "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years, "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance, "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role, "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
    }

    emp_df = pd.DataFrame([input_data])[metadata["all_raw_cols"]]
    risk_prob = float(pipeline.predict_proba(emp_df)[0, 1])

    high_risk_cutoff = min(DECISION_THRESHOLD + 0.2, 0.95)
    if risk_prob >= high_risk_cutoff:
        tier, tier_cls, tier_label = risk_prob, "risk-high", "HIGH RISK"
    elif risk_prob >= DECISION_THRESHOLD:
        tier, tier_cls, tier_label = risk_prob, "risk-medium", "MEDIUM RISK"
    else:
        tier, tier_cls, tier_label = risk_prob, "risk-low", "LOW RISK"

    # ---------- Results header ----------
    st.markdown('<div class="app-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)

    # ---------- Top row: gauge + metrics ----------
    col_gauge, col_metrics = st.columns([1, 1.4])

    with col_gauge:
        gauge_fig = make_risk_gauge(risk_prob, DECISION_THRESHOLD, high_risk_cutoff)
        st.pyplot(gauge_fig, use_container_width=True)

    with col_metrics:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Attrition Risk Probability</div>
                <div class="value">{risk_prob:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top:10px;">
                <div class="label">Risk Tier</div>
                <div style="margin-top:6px;">
                    <span class="risk-badge {tier_cls}">{tier_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top:10px;">
                <div class="label">Decision Threshold</div>
                <div class="value" style="font-size:18px;">{DECISION_THRESHOLD:.0%}
                    <span style="font-size:12px;color:#64748B;font-weight:400;"> (tuned for F1 on minority class)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- SHAP explanation ----------
    if explainer is not None:
        emp_transformed = pipeline.named_steps["preprocessor"].transform(emp_df)
        emp_shap = explainer.shap_values(emp_transformed)
        shap_series = pd.Series(emp_shap[0], index=feature_names).sort_values(key=abs, ascending=False)
        top_factors = shap_series.head(6)

        st.markdown(
            '<div class="section-title" style="margin-top:24px;">🎯 Top Factors Behind This Prediction</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="info-banner">Red bars push the employee toward leaving; '
            "teal bars push toward staying. The longer the bar, the stronger the influence.</div>",
            unsafe_allow_html=True,
        )
        shap_fig = make_shap_chart(top_factors)
        st.pyplot(shap_fig, use_container_width=True)

        # ---------- Retention suggestions ----------
        risk_drivers = top_factors[top_factors > 0].index.tolist()
        suggestions = []
        if any("OverTime" in f for f in risk_drivers):
            suggestions.append(("Workload", "Review workload — this employee works overtime, a strong attrition driver."))
        if "JobSatisfaction" in risk_drivers:
            suggestions.append(("Check-in", "Schedule a check-in on job satisfaction and role fit."))
        if "MonthlyIncome" in risk_drivers:
            suggestions.append(("Compensation", "Review compensation relative to role and market benchmarks."))
        if "StockOptionLevel" in risk_drivers:
            suggestions.append(("Equity", "Consider offering or increasing stock option level."))
        if any("BusinessTravel" in f for f in risk_drivers):
            suggestions.append(("Travel", "Reassess travel demands for this role."))
        if any("WorkLifeBalance" in f or "EnvironmentSatisfaction" in f for f in risk_drivers):
            suggestions.append(("Environment", "Check in on work-life balance and team environment."))
        if any("JobRole" in f for f in risk_drivers):
            suggestions.append(("Role", "Role-specific factors are contributing — consider a role-focused retention conversation."))

        if suggestions:
            st.markdown(
                '<div class="section-title" style="margin-top:24px;">✅ Suggested Retention Actions</div>',
                unsafe_allow_html=True,
            )
            for label, text in suggestions:
                st.markdown(
                    f'<div class="suggestion"><strong>{label}:</strong> {text}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info(
            "Per-employee SHAP explanation is unavailable because the SHAP explainer "
            "artifact could not be loaded in this environment. The risk probability "
            "above is still valid."
        )

    # ---------- Caption ----------
    st.caption(
        f"Decision threshold: {DECISION_THRESHOLD:.2f} — tuned on held-out test data to "
        "maximize F1 for the minority 'Leave' class. The model is trained on SMOTE-balanced "
        "data, so raw probabilities are not calibrated to the true ~16% real-world attrition "
        "rate; ranking between employees is still meaningful."
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="app-footer">
        Employee Attrition Risk Predictor · Built with Streamlit + Scikit-learn + SHAP<br>
        Model: Logistic Regression · SMOTE-balanced · Threshold-tuned · <a href="https://github.com/IsratMou">Israt Jahan Mou</a>
    </div>
    """,
    unsafe_allow_html=True,
)
=======
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import traceback

# ----------------------------
# Load model artifacts (produced at the end of the notebook).
# Wrapped in try/except so the Streamlit app surfaces a readable error
# instead of a redacted "ModuleNotFoundError" page on Streamlit Cloud.
# ----------------------------

@st.cache_resource
def load_artifacts():
    pipeline = joblib.load('attrition_pipeline.pkl')
    feature_names = joblib.load('feature_names.pkl')
    threshold = joblib.load('decision_threshold.pkl')
    metadata = joblib.load('feature_metadata.pkl')

    # shap_explainer.pkl unpickles a shap.explainers._linear.LinearExplainer
    # object. If the shap package (or its numba/llvmlite deps) is not
    # installed in the Streamlit Cloud environment, joblib.load raises
    # ModuleNotFoundError. We catch it here so the app can still serve
    # predictions without SHAP explanations.
    explainer = None
    shap_error = None
    try:
        explainer = joblib.load('shap_explainer.pkl')
    except ModuleNotFoundError as e:
        shap_error = (
            f"SHAP explainer could not be loaded: {e.name}. "
            "Predictions will still work, but per-employee SHAP explanations "
            "are disabled. Fix: pin Python 3.11 and add numba/llvmlite/shap "
            "to requirements.txt."
        )
    except Exception as e:
        shap_error = f"SHAP explainer could not be loaded: {e}"

    return pipeline, explainer, feature_names, threshold, metadata, shap_error


try:
    pipeline, explainer, feature_names, DECISION_THRESHOLD, metadata, shap_error = \
        load_artifacts()
except Exception:
    st.set_page_config(page_title="Employee Attrition Predictor",
                       page_icon="💼", layout="wide")
    st.title("💼 Employee Attrition Risk Predictor")
    st.error(
        "Failed to load model artifacts. See the error below — this is "
        "almost always a Python version / missing dependency issue on "
        "Streamlit Cloud (Python 3.14 lacks wheels for numba/llvmlite, "
        "which shap depends on)."
    )
    st.code(traceback.format_exc())
    st.stop()


st.set_page_config(page_title="Employee Attrition Predictor",
                   page_icon="💼", layout="wide")
st.title("💼 Employee Attrition Risk Predictor")
st.write("Enter employee details below to estimate attrition risk and see the key drivers behind the prediction.")

if shap_error:
    st.warning(shap_error)

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

    emp_df = pd.DataFrame([input_data])[metadata['all_raw_cols']]
    risk_prob = pipeline.predict_proba(emp_df)[0, 1]

    st.divider()
    st.subheader("Prediction Result")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Attrition Risk Probability", f"{risk_prob:.1%}")
        st.caption(f"Threshold for flagging risk: {DECISION_THRESHOLD:.0%}")
        high_risk_cutoff = min(DECISION_THRESHOLD + 0.2, 0.95)
        if risk_prob >= high_risk_cutoff:
            st.error("🔴 High Risk")
        elif risk_prob >= DECISION_THRESHOLD:
            st.warning("🟡 Medium Risk")
        else:
            st.success("🟢 Low Risk")
        st.progress(min(float(risk_prob), 1.0))

    # SHAP explanation is only shown if the explainer loaded successfully.
    if explainer is not None:
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
    else:
        with col2:
            st.info(
                "Per-employee SHAP explanation is unavailable because the "
                "SHAP explainer artifact could not be loaded in this "
                "environment. The risk probability above is still valid."
            )

    st.caption(
        f"Decision threshold: {DECISION_THRESHOLD:.2f} (tuned on held-out test data to maximize F1 for the "
        "minority 'Leave' class, rather than the default 0.5 — this model is trained on SMOTE-balanced data, "
        "so raw probabilities are not calibrated to the true ~16% real-world attrition rate; ranking between "
        "employees is still meaningful)."
    )
>>>>>>> 6d5d4af1dc710cc288e06b3359963dcb6b33880c
