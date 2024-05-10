import streamlit as st
import pandas as pd
import pickle

# Load the model
model = pickle.load(open("model.pkl", 'rb'))

# Set up the app title and description
st.title('Test if you have any risk of developing Diabetes Type II')
st.write('Please enter the following data to predict health risks:')

# Input fields for Yes/No questions directly collecting as 1 or 0
def get_binary_response(option):
    return 1 if option == 'Yes' else 0

HighBP = st.radio('Do you have High Blood Pressure', options=['Yes', 'No'])
HighChol = st.radio('Do you have High Cholesterol', options=['Yes', 'No'])
CholCheck = st.radio('Did you have your Cholesterol checked in the last 5 years', options=['Yes', 'No'])
BMI = st.number_input('Body Mass Index', min_value=0.0, step=0.1)
Smoker = st.radio('Smoker', options=['Yes', 'No'])
Stroke = st.radio('Stroke', options=['Yes', 'No'])
HeartDiseaseorAttack = st.radio('Heart Disease or Attack', options=['Yes', 'No'])
PhysActivity = st.radio('Physical Activity', options=['Yes', 'No'])
Fruits = st.radio('Do you consume at least 1 portion of fruits per day', options=['Yes', 'No'])
Veggies = st.radio('Do you consume at least 1 portion of veggies per day', options=['Yes', 'No'])
HvyAlcoholConsump = st.radio('Do you drink more than 7 drinks per week', options=['Yes', 'No'])
AnyHealthcare = st.radio('Do you have any kind of Healthcare', options=['Yes', 'No'])
NoDocbcCost = st.radio('Have you ever avoided going to Doctor because of the cost', options=['Yes', 'No'])
GenHlth = st.select_slider('How do you consider your General Health, 1 being very poor and 5 being amazing', options=[1, 2, 3, 4, 5])
MentHlth = st.select_slider('How many days of poor mental health in past 30 days', options=range(1, 31))
PhysHlth = st.select_slider('How many days of poor physical health in past 30 days', options=range(1, 31))
DiffWalk = st.radio('Do you have any difficulty in walking?', options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')

Sex = st.radio('Gender', options=['Male', 'Female'])
Age = st.number_input('Age', min_value=0, step=1)
Education = st.selectbox('Education Level', options=['Less than High School', 'High School Graduate', 'Some College', 'College Graduate'])
Income = st.selectbox('Income Level', options=['Less than $10,000', '$10,000 to $24,999', '$25,000 to $49,999', '$50,000 to $74,999', '$75,000 or more'])

input_data = [
    get_binary_response(HighBP), get_binary_response(HighChol), get_binary_response(CholCheck), BMI, get_binary_response(Smoker),
    get_binary_response(Stroke), get_binary_response(HeartDiseaseorAttack), get_binary_response(PhysActivity), get_binary_response(Fruits), get_binary_response(Veggies),
    get_binary_response(HvyAlcoholConsump), get_binary_response(AnyHealthcare), get_binary_response(NoDocbcCost), GenHlth, MentHlth,
    PhysHlth, DiffWalk, 1 if Sex == 'Male' else 0, Age,
    {'Less than High School': 1, 'High School Graduate': 2, 'Some College': 3, 'College Graduate': 4}[Education],
    {'Less than $10,000': 1, '$10,000 to $24,999': 2, '$25,000 to $49,999': 3, '$50,000 to $74,999': 4, '$75,000 or more': 5}[Income]
]

input_df = pd.DataFrame([input_data], columns=[
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
    'Income'
])

if st.button('Predict Health Risks'):
    prediction = model.predict(input_df)
    st.write('Prediction:', 'Higher Risk' if prediction[0] == 1 else 'Lower Risk')











