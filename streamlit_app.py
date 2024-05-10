import streamlit as st
import pandas as pd
import pickle

# Load the model safely
try:
    model = pickle.load(open("model.pkl", 'rb'))
except FileNotFoundError:
    st.error("The model file was not found. Please ensure the 'model.pkl' is placed in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

st.title('Health Risk Prediction App')
st.write('Please enter your data to predict health risks:')

# Input fields for health data
HighBP = st.number_input('High Blood Pressure (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)
HighChol = st.number_input('High Cholesterol (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)
CholCheck = st.number_input('Cholesterol Checked in last 5 years (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)

# BMI Calculator
st.header("BMI Calculator and Information")
weight = st.number_input("Enter your weight in kilograms (kg):", min_value=0.0, format="%.2f")
height = st.number_input("Enter your height in meters (m):", min_value=0.0, format="%.2f")

BMI = 0
if st.button('Calculate BMI'):
    if height > 0:
        BMI = weight / (height ** 2)
        st.write(f"Your Calculated BMI: {BMI:.2f}")
        # Category based on calculated BMI
        if BMI < 18.5:
            st.error("You are underweight.")
        elif 18.5 <= BMI <= 24.9:
            st.success("You have a normal weight.")
        elif 25 <= BMI <= 29.9:
            st.warning("You are overweight.")
        elif 30 <= BMI <= 34.9:
            st.error("You are considered obese (Class I).")
        elif 35 <= BMI <= 39.9:
            st.error("You are considered obese (Class II).")
        elif BMI >= 40:
            st.error("You are considered obese (Class III).")
    else:
        st.error("Height must be greater than zero to calculate BMI.")

# Additional health information
Smoker = st.number_input('Smoker (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)
Stroke = st.number_input('History of Stroke (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)
HeartDiseaseorAttack = st.number_input('Heart Disease or Heart Attack (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)
PhysActivity = st.number_input('Physical Activity (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)
Fruits = st.number_input('Consumes Fruits Regularly (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)
Veggies = st.number_input('Consumes Vegetables Regularly (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)
HvyAlcoholConsump = st.number_input('Heavy Alcohol Consumption (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)
AnyHealthcare = st.number_input('Has Healthcare (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)
NoDocbcCost = st.number_input('No Doctor Because of Cost (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)
GenHlth = st.number_input('General Health (1-5)', min_value=1, max_value=5, step=1)
MentHlth = st.number_input('Mental Health (Days of poor mental health in past 30 days)', min_value=0, step=1)
PhysHlth = st.number_input('Physical Health (Days of poor physical health in past 30 days)', min_value=0, step=1)
DiffWalk = st.number_input('Difficulty Walking (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)
Sex = st.radio('Sex', options=['Male', 'Female'])
Age = st.number_input('Age', min_value=0, step=1)
Education = st.selectbox('Education Level', options=['Less than High School', 'High School Graduate', 'Some College', 'College Graduate'])
Income = st.selectbox('Income Level', options=['Less than $10,000', '$10,000 to $24,999', '$25,000 to $49,999', '$50,000 to $74,999', '$75,000 or more'])
sex_numeric = 1 if Sex == 'Male' else 0
education_levels = {'Less than High School': 1, 'High School Graduate': 2, 'Some College': 3, 'College Graduate': 4}
education_numeric = education_levels[Education]
income_levels = {'Less than $10,000': 1, '$10,000 to $24,999': 2, '$25,000 to $49,999': 3, '$50,000 to $74,999': 4, '$75,000 or more': 5}
income_numeric = income_levels[Income]

# Prediction button
if st.button('Predict Health Risks'):
    input_data = [HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
                  HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, sex_numeric, Age,
                  education_numeric, income_numeric]
    input_df = pd.DataFrame([input_data], columns=['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                                                   'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                                                   'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                                                   'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
                                                   'Income'])
    prediction = model.predict(input_df)
    st.write('Prediction:', 'Higher Risk' if prediction[0] == 1 else 'Lower Risk')




































