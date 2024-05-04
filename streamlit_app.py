import streamlit as st
import joblib
import numpy as np
import shap

# Load your trained model
model_path = 'path/to/your/model.joblib'  # Adjust path as necessary
model = joblib.load(model_path)

# Check the model type and create appropriate SHAP explainer
if hasattr(model, 'predict_proba'):  # Check if model supports the predict_proba method (common for classifiers)
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        st.error(f"Error with TreeExplainer: {e}")
        # Fallback to KernelExplainer for models not supported by TreeExplainer
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(data, 100))
else:
    # Assume a non-probabilistic model or unsupported model type
    st.error("Unsupported model type for SHAP TreeExplainer.")

# Creating the Streamlit interface
st.title('Diabetes Prediction App')
st.write('Please enter the following data to predict diabetes:')

# Form to input new data for prediction
# (assuming input features same as previous example)
input_data = np.array([[
    st.number_input('Number of pregnancies', min_value=0, step=1),
    st.number_input('Plasma glucose concentration', min_value=0, step=1),
    st.number_input('Diastolic blood pressure (mm Hg)', min_value=0, step=1),
    st.number_input('Triceps skin fold thickness (mm)', min_value=0, step=1),
    st.number_input('2-Hour serum insulin (mu U/ml)', min_value=0, step=1),
    st.number_input('Body mass index', min_value=0.0, step=0.1),
    st.number_input('Diabetes pedigree function', min_value=0.0, step=0.01),
    st.number_input('Age', min_value=0, step=1)
]])

if st.button('Predict Diabetes'):
    prediction = model.predict(input_data)
    # Output the prediction
    if prediction[0] == 0:
        st.success('The prediction is: No Diabetes')
    else:
        st.error('The prediction is: Diabetes')

    # Generate SHAP values
    shap_values = explainer.shap_values(input_data)
    st.shap.plot.bar(shap_values)









