import streamlit as st
import numpy as np
import pickle

with open(r'heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open(r'scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


st.title("Heart Disease Prediction")
st.write("Fill the following details to predict the likelihood of heart disease")

age=st.number_input("Age", min_value=20, max_value=100, value=50)
sex=st.selectbox("Sex",[0,1],format_func=lambda x:"Female" if x==0 else "Male")
cp=st.selectbox("Chest Pain Type",[0,1,2,3])
treastbps=st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol=st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
fbs=st.selectbox("Fassting Blood Sugar > 120 mg/dl", [0,1])
restecg=st.selectbox("Resting ECG",[0,1,2])
thalach=st.number_input("Maximum Heart rate Achieved", min_value=60, max_value=200, value=150)
exang=st.selectbox("Exercise Induced Angina", [0,1])
olad=st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
slope=st.selectbox("Slape of the peak exercise ST segment",[0,1,2])
ca=st.selectbox("Number of major vessels",[0,1,2,3])
thal=st.selectbox("Thalassemia",[0,1,2])
if st.button("Predict Heart Disease"):
    input_data = np.array([[age, sex, cp, treastbps, chol, fbs, restecg, thalach, exang, olad, slope, ca, thal]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.write("The model predicts that the person has heart disease.")
    else:
        st.write("The model predicts that the person does not have heart disease.")