import streamlit as st
import pandas as pd 
import joblib 

model = joblib.load('LECTURE 1/Project_2_lect1/Heart_UI+Model/KNN_Heart.pkl')
scaler = joblib.load('LECTURE 1/Project_2_lect1/Heart_UI+Model/scaler.pkl')
columns = joblib.load('LECTURE 1/Project_2_lect1/Heart_UI+Model/columns.pkl')

st.title("Heart Stroke Prediction by Yug")
st.markdown("Provide the following details to predict the likelihood of a heart stroke:")
age=st.slider("Age", 0, 100, 25 )
sex=st.selectbox("Sex", ["Male", "Female"])
chest_pain_type=st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_blood_pressure=st.slider("Resting Blood Pressure", 80, 200, 120)
cholesterol=st.slider("Cholesterol", 100, 400, 200)
resting_ecg=st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
max_heart_rate=st.slider("Max Heart Rate", 60, 220, 150)
exercise_induced_angina=st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak=st.slider("Oldpeak", 0.0, 6.0, 1.0)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == "Male" else 0],
        'chest_pain_type': [chest_pain_type],
        'resting_blood_pressure': [resting_blood_pressure],
        'cholesterol': [cholesterol],
        'resting_ecg': [resting_ecg],
        'max_heart_rate': [max_heart_rate],
        'exercise_induced_angina': [1 if exercise_induced_angina == "Yes" else 0],
        'oldpeak': [oldpeak]
    })
    input_data = pd.get_dummies(input_data, columns=['chest_pain_type', 'resting_ecg'])
    input_data = input_data.reindex(columns=columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    st.write(f"Prediction: {'Heart Stroke' if prediction[0] == 1 else 'No Heart Stroke'}")