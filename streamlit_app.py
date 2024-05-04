import streamlit as st
import joblib
import numpy as np

# Correct the model path to a relative path assuming both the script and model are in the same directory
model_path = 'diabetes_prediction_model.joblib'
model = joblib.load(model_path)  # This loads the model directly from your repository

# Creating the Streamlit interface
st.title('Diabetes Prediction App')
st.write('Please enter the following data to predict your diabetes probability:')

# Form to input new data for prediction
pregnancies = st.number_input('Number of pregnancies', min_value=0, step=1)
glucose = st.number_input('Plasma glucose concentration', min_value=0, step=1)
blood_pressure = st.number_input('Diastolic blood pressure (mm Hg)', min_value=0, step=1)
skin_thickness = st.number_input('Triceps skin fold thickness (mm)', min_value=0, step=1)
insulin = st.number_input('2-Hour serum insulin (mu U/ml)', min_value=0, step=1)
bmi = st.number_input('Body mass index', min_value=0.0, step=0.1)
diabetes_pedigree = st.number_input('Diabetes pedigree function', min_value=0.0, step=0.01)
age = st.number_input('Age', min_value=0, step=1)

# Button to make prediction
if st.button('Predict Diabetes'):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    prediction_proba = model.predict_proba(input_data)[0][1]  # Probability of having diabetes
    st.write(f'The probability of having diabetes is: {prediction_proba * 100:.2f}%')
















