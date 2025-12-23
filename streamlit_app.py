import streamlit as st
import joblib
import pandas as pd

# ---------------- PATHS ----------------
MODEL_PATH = "models/xgb_model.pkl"
SCALER_PATH = "models/scaler.pkl"
COUGH_ENCODER_PATH = "models/cough_encoder.pkl"
GENDER_ENCODER_PATH = "models/gender_encoder.pkl"
CITY_COLUMNS_PATH = "models/city_column.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"

# ---------------- LOAD OBJECTS ----------------
@st.cache_resource
def load_objects():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    cough_encoder = joblib.load(COUGH_ENCODER_PATH)
    gender_encoder = joblib.load(GENDER_ENCODER_PATH)
    city_columns = joblib.load(CITY_COLUMNS_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, scaler, cough_encoder, gender_encoder, city_columns, feature_names

model, scaler, cough_encoder, gender_encoder, city_columns, feature_names = load_objects()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="COVID Predictor",
    page_icon="🦠",
    layout="centered"
)

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align:center;color:#FF4B4B;'>🦠 COVID-19 Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#FFD700;'>Predict COVID Risk Based on Symptoms</p>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<h5 style='text-align:left;color:#89CFF0;'>👤 Patient Details</h5>",
    unsafe_allow_html=True
)

# ---------------- INPUT FIELDS ----------------
# First row
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
with col2:
    fever = st.number_input("Fever (°F)", min_value=95.0, max_value=105.0, value=98.6)
with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])

# Second row
col4, col5 = st.columns(2)
with col4:
    cough = st.selectbox("Cough Type", ["Mild", "Strong"])
with col5:
    city = st.selectbox("City", ["Mumbai", "Delhi", "Kolkata", "Bangalore"])

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button("Predict"):
    input_df = pd.DataFrame({
        "Age": [age],
        "Fever": [fever],
        "Gender": [gender],
        "Cough": [cough],
        "City": [city]
    })

    # Encode categorical features
    input_df["Cough"] = cough_encoder.transform(input_df[["Cough"]])
    input_df["Gender"] = gender_encoder.transform(input_df["Gender"])

    # One-hot encode city
    city_df = pd.get_dummies(input_df["City"], prefix="City")
    city_df = city_df.reindex(columns=city_columns, fill_value=0)

    input_df = input_df.drop("City", axis=1)
    input_df = pd.concat([input_df, city_df], axis=1)

    # Scale numeric features
    num_cols = ["Age", "Fever"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Align features with training
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    input_df = input_df.astype(float)

    # Make prediction
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Show result
    if pred == 1:
        st.error(f"⚠️ High Risk of COVID\n\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Low Risk of COVID\n\nProbability: {1 - prob:.2f}")
