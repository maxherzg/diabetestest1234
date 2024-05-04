import streamlit as st
import joblib
import numpy as np
import shap  # Make sure to install shap with pip install shap

# Load your trained model from the specified path
model_path = 'diabetes_prediction_model.joblib'  # Update the path as necessary
model = joblib.load(model_path)

# Initialize the SHAP explainer (assuming a tree-based model, adjust as needed)
explainer = shap.TreeExplainer(model)

# Streamlit interface
st.title('Diabetes Prediction App')
st.write('Please enter the following data to predict the probability of diabetes and understand parameter impacts:')

# Input data from the user
pregnancies = st.number_input('Number of pregnancies', min_value=0, step=1)
glucose = st.number_input('Plasma glucose concentration', min_value=0, step=1)
blood_pressure = st.number_input('Diastolic blood pressure (mm Hg)', min_value=0, step=1)
skin_thickness = st.number_input('Triceps skin fold thickness (mm)', min_value=0, step=1)
insulin = st.number_input('2-Hour serum insulin (mu U/ml)', min_value=0, step=1)
bmi = st.number_input('Body mass index', min_value=0.0, step=0.1)
diabetes_pedigree = st.number_input('Diabetes pedigree function', min_value=0.0, step=0.01)
age = st.number_input('Age', min_value=0, step=1)

# Button to make prediction and show SHAP values
if st.button('Predict Diabetes'):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    shap_values = explainer.shap_values(input_data)
    prediction = model.predict_proba(input_data)[0][1]

    st.success(f'The probability of diabetes is: {prediction*100:.2f}%')

    # Display SHAP values for each feature
    st.subheader('Impact of each feature on the prediction:')
    features = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree', 'Age']
    for feature, value, shap_value in zip(features, input_data[0], shap_values[1][0]):  # Assuming binary classification, index 1 for positive class
        st.text(f"{feature}: {value} - Impact on model output: {shap_value:.2f}")








