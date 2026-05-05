import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# ─────────────────────────────────────────────
# Load model + training columns
# ─────────────────────────────────────────────
model = joblib.load("Heart_disease.pkl")
columns = joblib.load("columns.pkl")

# ─────────────────────────────────────────────
# UI Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
.big-title {
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#e63946;
}
.result-yes {
    font-size:28px;
    color:red;
    font-weight:bold;
}
.result-no {
    font-size:28px;
    color:green;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">❤️ Heart Disease Predictor</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.title("About")
st.sidebar.info("ML-based Heart Disease Prediction App")

# ─────────────────────────────────────────────
# Input Section (FIXED)
# ─────────────────────────────────────────────
st.subheader("📊 Enter Patient Details")

age = st.number_input("Age", 1, 120)

sex = st.selectbox("Sex", ["Male", "Female"])

# FIX 1: correct cp range (NO 4)
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])

trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol Level", 100, 600)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 1, 2])

thalach = st.number_input("Max Heart Rate Achieved", 60, 220)

exang = st.selectbox("Exercise Induced Angina", [0, 1])

oldpeak = st.number_input("ST Depression", 0.0, 10.0, step=0.1)

slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])

# FIX 2: IMPORTANT → must match training dataset
thal = st.selectbox("Thalassemia", [3, 6, 7])

# Convert sex
sex = 1 if sex == "Male" else 0

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
if st.button("🔍 Predict"):

    # Create input
    features = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    input_df = pd.DataFrame([features])

    # Ensure correct order
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]

    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)[0]

    # ─────────────────────────────────────────
    # Output
    # ─────────────────────────────────────────
    if prediction == 1:
        st.markdown('<p class="result-yes">⚠️ High Risk of Heart Disease</p>', unsafe_allow_html=True)

        if prob is not None:
            st.progress(float(prob[1]))
            st.write(f"Risk Probability: **{prob[1]*100:.2f}%**")

    else:
        st.markdown('<p class="result-no">✅ Low Risk (Healthy)</p>', unsafe_allow_html=True)

        if prob is not None:
            st.progress(float(prob[0]))
            st.write(f"Safe Probability: **{prob[0]*100:.2f}%**")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.write("Made with ❤️ using Streamlit")