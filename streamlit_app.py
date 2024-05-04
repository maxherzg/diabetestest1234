import streamlit as st
import joblib
import numpy as np

# Load your trained model from the specified path
model_path = 'path_to_your_model.joblib'  # Update the path as necessary
model = joblib.load(model_path)

# Streamlit interface
st.title('Diabetes Prediction App')
st.write('Please enter the following data to predict diabetes probability:')

# Input data from the user
pregnancies = st.number_input('Number of pregnancies', min_value=0, step=1)
glucose = st.number_input('Plasma glucose concentration', min_value=0, step=1)
blood_pressure = st.number_input('Diastolic blood pressure (mm Hg)', min_value=0, step=1)
skin_thickness = st.number_input('Triceps skin fold thickness (mm)', min_value=0, step=1)
insulin = st.number_input('2-Hour serum insulin (mu U/ml)', min_value=0, step=1)
bmi = st.number_input('Body mass index', min_value=0.0, step=0.1)
diabetes_pedigree = st.number_input('Diabetes pedigree function', min_value=0.0, step=0.01)
age = st.number_input('Age', min_value=0, step=1)

# Prediction button
if st.button('Predict Diabetes'):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    probabilities = model.predict_proba(input_data)
    diabetes_probability = probabilities[0][1]  # Probability of diabetes
    
    st.success(f'The probability of diabetes is: {diabetes_probability*100:.2f}%')









