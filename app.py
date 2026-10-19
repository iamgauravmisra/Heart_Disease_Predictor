import streamlit as st
import pandas as pd
import joblib 

model = joblib.load('LR_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("Heart Disease Prediction App ❤️")
st.markdown("Provide the following information to predict the likelihood of heart disease.")

age = st.slider("Age",18,100,40)
sex = st.selectbox("Sex", ["Male", "Female"])

chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)

cholesterol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
rest_ecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
max_hr = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", 0.0, 10.0, 1.0)
st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Up", "Flat", "Down"])

if st.button("Predict"):
    # Map UI inputs to the model's expected feature names (including one-hot columns)
    raw_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "Yes" else 0,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "Male" else 0,
        # Chest pain one-hot (ASY is the baseline => all zeros)
        "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
        "ChestPainType_TA": 1 if chest_pain == "TA" else 0,
        # Resting ECG one-hot (Left Ventricular Hypertrophy is baseline)
        "RestingECG_Normal": 1 if rest_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if rest_ecg == "ST-T Wave Abnormality" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,
        # ST slope one-hot (Down is baseline)
        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Up" else 0
    }
    
    input_df = pd.DataFrame([raw_data])
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_columns]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"Prediction: {'‼️ High Risk of Heart Disease' if prediction[0] == 1 else '✅ Low Risk of Heart Disease'}")