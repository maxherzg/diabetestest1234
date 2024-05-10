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

HighBP = st.radio('Do you have High Blood Pressure', options=['Yes', 'No'])
HighChol = st.radio('Do you have High Colesterol', options=['Yes', 'No'])
CholCheck = st.radio('Did you have your Cholesterol checked in the last 5 years', options=['Yes', 'No'])

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

Smoker = st.radio('Smoker', options=['Yes', 'No'])
Stroke = st.radio('Stroke', options=['Yes', 'No'])
HeartDiseaseorAttack = st.radio('HeartDiseaseorAttack', options=['Yes', 'No'])
PhysActivity = st.radio('PhysActivity', options=['Yes', 'No'])
Fruits = st.radio('Do you consume at least 1 portion of fruits per day', options=['Yes', 'No'])
Veggies = st.radio('Do you consume at least 1 portion of veggies per day', options=['Yes', 'No'])
HvyAlcoholConsump = st.radio('Do you drink more than 7 drinks per week', options=['Yes', 'No'])
AnyHealthcare = st.radio('Do you have any kind of Healthcare', options=['Yes', 'No'])
NoDocbcCost = st.radio('Have you ever avoided going to Doctor because of the cost', options=['Yes', 'No'])

GenHlth = st.select_slider('How doyou consider your General Health, 1 being very poor and 5 being amazing',options=list(range(1, 6)),value=3)

MentHlth = st.select_slider('How many days of poor mental health in the past 30 days',options=list(range(1, 31)),value=15)

PhysHlth = st.select_slider('How many days of poor physical health in the past 30 days',options=list(range(1, 31)),value=15)

DiffWalk = st.radio('Do you have any difficulty in walking?', options=['Yes', 'No'])
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




































