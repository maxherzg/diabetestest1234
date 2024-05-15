import streamlit as st
import pandas as pd
import pickle
try:
    model = pickle.load(open("model-2.pkl", 'rb'))
except FileNotFoundError:
    st.error("The model file was not found. Please ensure the 'model-2.pkl' is placed in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()
st.title('Health Risk Prediction App')
st.write('Please enter your data to predict health risks:')
HighBP = st.radio('Do you have High Blood Pressure?', options=['Yes', 'No'])
HighBP_numeric = 1 if HighBP == 'Yes' else 0
HighChol = st.radio('Do you have High Cholesterol?', options=['Yes', 'No'])
HighChol_numeric = 1 if HighChol == 'Yes' else 0
CholCheck = st.radio('Did you have your Cholesterol checked in the last 5 years?', options=['Yes', 'No'])
CholCheck_numeric = 1 if CholCheck == 'Yes' else 0
st.header("BMI Calculator and Information")
weight = st.number_input("Enter your weight in kilograms (kg):", min_value=0.0, format="%.2f")
height = st.number_input("Enter your height in meters (m):", min_value=0.0, format="%.2f")
BMI = 0
if st.button('Calculate BMI'):
    if height > 0:
        BMI = weight / (height ** 2)
        st.write(f"Your Calculated BMI: {BMI:.2f}")
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
Smoker = st.radio('Smoker?', options=['Yes', 'No'])
Smoker_numeric = 1 if Smoker == 'Yes' else 0
Stroke = st.radio('History of Stroke?', options=['Yes', 'No'])
Stroke_numeric = 1 if Stroke == 'Yes' else 0
HeartDiseaseorAttack = st.radio('Heart Disease or Heart Attack?', options=['Yes', 'No'])
HeartDiseaseorAttack_numeric = 1 if HeartDiseaseorAttack == 'Yes' else 0
PhysActivity = st.radio('Physical Activity?', options=['Yes', 'No'])
PhysActivity_numeric = 1 if PhysActivity == 'Yes' else 0
Fruits = st.radio('Do you consume at least 1 portion of fruits per day?', options=['Yes', 'No'])
Fruits_numeric = 1 if Fruits == 'Yes' else 0
Veggies = st.radio('Do you consume at least 1 portion of veggies per day?', options=['Yes', 'No'])
Veggies_numeric = 1 if Veggies == 'Yes' else 0
HvyAlcoholConsump = st.radio('Do you drink more than 7 drinks per week?', options=['Yes', 'No'])
HvyAlcoholConsump_numeric = 1 if HvyAlcoholConsump == 'Yes' else 0
AnyHealthcare = st.radio('Do you have any kind of Healthcare?', options=['Yes', 'No'])
AnyHealthcare_numeric = 1 if AnyHealthcare == 'Yes' else 0
NoDocbcCost = st.radio('Have you ever avoided going to Doctor because of the cost?', options=['Yes', 'No'])
NoDocbcCost_numeric = 1 if NoDocbcCost == 'Yes' else 0
GenHlth = st.select_slider('How do you consider your General Health, 1 being very poor and 5 being amazing', options=list(range(1, 6)), value=3)
MentHlth = st.select_slider('How many days of poor mental health in the past 30 days', options=list(range(1, 31)), value=15)
PhysHlth = st.select_slider('How many days of poor physical health in the past 30 days', options=list(range(1, 31)), value=15)
DiffWalk = st.radio('Do you have any difficulty in walking?', options=['Yes', 'No'])
DiffWalk_numeric = 1 if DiffWalk == 'Yes' else 0
Sex = st.radio('Sex', options=['Male', 'Female'])
sex_numeric = 1 if Sex == 'Male' else 0
Age = st.number_input('Age', min_value=0, step=1)
Education = st.selectbox('Education Level', options=['Less than High School', 'High School Graduate', 'Some College', 'College Graduate'])
education_numeric = {'Less than High School': 1, 'High School Graduate': 2, 'Some College': 3, 'College Graduate': 4}[Education]
Income = st.selectbox('Income Level', options=['Less than $10,000', '$10,000 to $24,999', '$25,000 to $49,999', '$50,000 to $74,999', '$75,000 or more'])
income_numeric = {'Less than $10,000': 1, '$10,000 to $24,999': 2, '$25,000 to $49,999': 3, '$50,000 to $74,999': 4, '$75,000 or more': 5}[Income]
if st.button('Predict Health Risks'):
    input_data = [HighBP_numeric, HighChol_numeric, CholCheck_numeric, BMI, Smoker_numeric, Stroke_numeric,
                  HeartDiseaseorAttack_numeric, PhysActivity_numeric, Fruits_numeric, Veggies_numeric,
                  HvyAlcoholConsump_numeric, AnyHealthcare_numeric, NoDocbcCost_numeric, GenHlth, MentHlth,
                  PhysHlth, DiffWalk_numeric, sex_numeric, Age, education_numeric, income_numeric]
    input_df = pd.DataFrame([input_data], columns=['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                                                   'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                                                   'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                                                   'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
                                                   'Income'])
    try:
        prediction = model.predict(input_df)
        risk_level = 'Higher Risk' if prediction[0] == 1 else 'Lower Risk'
        st.write(f'Prediction: {risk_level}')
        if risk_level == 'Higher Risk':
            st.warning("Please consider consulting with a healthcare professional.")
        else:
            st.success("Keep maintaining a healthy lifestyle!")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")



















































