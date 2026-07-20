"""
Heart Disease Prediction Streamlit App

This app uses a trained Random Forest model to predict the likelihood
of heart disease based on patient clinical measurements.
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from heart_disease.predict import (
    get_prediction_label,
    get_risk_level,
    load_model,
    predict_from_inputs,
)
from heart_disease.utils import CATEGORICAL_LABELS

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.15rem;
        padding-bottom: 1rem;
        max-width: 1160px;
    }

    .hero {
        background: linear-gradient(180deg, rgba(14, 87, 122, 0.12), rgba(14, 87, 122, 0.04));
        border: 1px solid rgba(14, 87, 122, 0.28);
        border-radius: 10px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.9rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.25;
        letter-spacing: 0.2px;
    }

    .hero p {
        margin: 0.3rem 0 0 0;
        opacity: 0.92;
        font-size: 0.94rem;
    }

    .hero-tag {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
        opacity: 0.78;
        margin-bottom: 0.35rem;
    }

    .section-card {
        border: 1px solid rgba(90, 100, 110, 0.26);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.95rem;
        background: rgba(250, 250, 250, 0.015);
    }

    .result-positive {
        background: rgba(178, 34, 34, 0.09);
        border: 1px solid rgba(178, 34, 34, 0.35);
        border-left: 5px solid #b22222;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin: 0.75rem 0;
    }

    .result-negative {
        background: rgba(20, 106, 73, 0.10);
        border: 1px solid rgba(20, 106, 73, 0.36);
        border-left: 5px solid #146a49;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin: 0.75rem 0;
    }

    .risk-note {
        font-size: 0.88rem;
        opacity: 0.88;
        margin-top: 0.3rem;
    }

    .footer {
        text-align: center;
        opacity: 0.68;
        font-size: 0.8rem;
        margin-top: 1.1rem;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 700;
        border: 1px solid rgba(14, 87, 122, 0.38);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the trained model (cached)
@st.cache_resource
def get_model():
    """Load the trained Random Forest model."""
    return load_model()


# Hero header
st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">Clinical Decision Support Prototype</div>
        <h1>Heart Disease Risk Dashboard</h1>
        <p>
            Enter patient clinical measurements to estimate heart disease risk using a trained
            Random Forest model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar - About section
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This app uses a **Random Forest Classifier** trained on the Heart Disease dataset.

        **Model summary**
        - Algorithm: Random Forest
        - Training split: 80%
        - Test accuracy: ~85%

        **Note**
        This tool supports education and screening awareness only.
        """
    )

    st.divider()
    st.subheader("Feature Guide")
    st.caption("Most fields use the same coding as common heart-disease benchmark datasets.")

# Main input form
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Patient Assessment Inputs")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Demographics & Baseline Vitals")
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50,
            help="Age in years",
        )

        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x: CATEGORICAL_LABELS["sex"][x],
            help="1 = Male, 0 = Female",
        )

        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: CATEGORICAL_LABELS["cp"][x],
            help="Type of chest pain",
        )

        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=50,
            max_value=250,
            value=120,
            help="Resting blood pressure",
        )

        chol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=50,
            max_value=600,
            value=200,
            help="Serum cholesterol",
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            options=[0, 1],
            format_func=lambda x: CATEGORICAL_LABELS["fbs"][x],
            help="1 = Yes, 0 = No",
        )

    with col2:
        st.markdown("#### ECG & Exercise Findings")
        restecg = st.selectbox(
            "Resting ECG Results",
            options=[0, 1, 2],
            format_func=lambda x: CATEGORICAL_LABELS["restecg"][x],
            help="Resting electrocardiographic results",
        )

        thalach = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=50,
            max_value=250,
            value=150,
            help="Maximum heart rate achieved",
        )

        exang = st.selectbox(
            "Exercise-Induced Angina",
            options=[0, 1],
            format_func=lambda x: CATEGORICAL_LABELS["exang"][x],
            help="1 = Yes, 0 = No",
        )

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="ST depression induced by exercise",
        )

        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[0, 1, 2],
            format_func=lambda x: CATEGORICAL_LABELS["slope"][x],
            help="Slope of the peak exercise ST segment",
        )

        ca = st.selectbox(
            "Number of Major Vessels (0-3)",
            options=[0, 1, 2, 3],
            format_func=lambda x: CATEGORICAL_LABELS["ca"][x],
            help="Number of major vessels colored by fluoroscopy",
        )

        thal = st.selectbox(
            "Thalassemia",
            options=[1, 2, 3],
            format_func=lambda x: CATEGORICAL_LABELS["thal"][x],
            help="Thalassemia status",
        )

    submitted = st.form_submit_button("Predict Heart Disease Risk", type="primary", width="stretch")

st.markdown("</div>", unsafe_allow_html=True)

# Prediction and result display
if submitted:
    model = get_model()

    result = predict_from_inputs(
        model=model,
        age=age,
        sex=sex,
        cp=cp,
        trestbps=trestbps,
        chol=chol,
        fbs=fbs,
        restecg=restecg,
        thalach=thalach,
        exang=exang,
        oldpeak=oldpeak,
        slope=slope,
        ca=ca,
        thal=thal,
    )

    prediction = result["prediction"]
    prob_disease = result["probability"] * 100
    prob_no_disease = result["probability_no_disease"] * 100
    risk_level = get_risk_level(result["probability"])
    label = get_prediction_label(prediction)

    st.markdown("---")
    st.subheader("Risk Assessment Results")

    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-positive">
                <h4 style="margin:0;">Elevated Concern: {label} ({risk_level})</h4>
                <p style="margin:0.35rem 0 0.2rem 0;">
                    Recommendation: Prompt clinical review is advised for comprehensive evaluation.
                </p>
                <p class="risk-note">Model-estimated risk: <strong>{prob_disease:.1f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-negative">
                <h4 style="margin:0;">Lower Concern: {label} ({risk_level})</h4>
                <p style="margin:0.35rem 0 0.2rem 0;">
                    Recommendation: Continue routine follow-up and maintain heart-healthy habits.
                </p>
                <p class="risk-note">Model-estimated risk: <strong>{prob_disease:.1f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk Level", risk_level)
    m2.metric("Heart Disease", f"{prob_disease:.1f}%")
    m3.metric("No Heart Disease", f"{prob_no_disease:.1f}%")

    st.caption("Probability bar corresponds to estimated probability of heart disease.")
    st.progress(min(max(prob_disease / 100, 0.0), 1.0))

    with st.expander("See submitted patient profile"):
        summary_rows = [
            ("Age", age),
            ("Sex", CATEGORICAL_LABELS["sex"][sex]),
            ("Chest Pain Type", CATEGORICAL_LABELS["cp"][cp]),
            ("Resting Blood Pressure", f"{trestbps} mm Hg"),
            ("Cholesterol", f"{chol} mg/dL"),
            ("Fasting Blood Sugar > 120", CATEGORICAL_LABELS["fbs"][fbs]),
            ("Resting ECG", CATEGORICAL_LABELS["restecg"][restecg]),
            ("Maximum Heart Rate", thalach),
            ("Exercise-Induced Angina", CATEGORICAL_LABELS["exang"][exang]),
            ("Oldpeak", oldpeak),
            ("Slope", CATEGORICAL_LABELS["slope"][slope]),
            ("Major Vessels", CATEGORICAL_LABELS["ca"][ca]),
            ("Thalassemia", CATEGORICAL_LABELS["thal"][thal]),
        ]
        summary_df = pd.DataFrame(summary_rows, columns=["Feature", "Value"])
        summary_df["Value"] = summary_df["Value"].astype(str)
        st.dataframe(summary_df, width="stretch", hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    "<div class='footer'>Heart Disease Risk Dashboard · Educational support tool only (not a diagnosis)</div>",
    unsafe_allow_html=True,
)
