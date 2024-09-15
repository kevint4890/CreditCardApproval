import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta, datetime

header = st.container()
features = st.container()
model = st.container()

def calc_days(date):
    today = datetime.now().date()
    days_difference = date - today

    return days_difference / timedelta(days=1)

# Collecting user input and encoding categorical variables
def user_input():
    col1, col2 = st.columns(2)

    gender = col1.selectbox('Gender?', options=['M', 'F'], index=0)
    car = col1.selectbox('Own a car? 0 for No, 1 for Yes', options=[0,1], index=0)
    realty = col1.selectbox('Own any realty? 0 for No, 1 for Yes', options=[0,1], index=0)
    income = col1.number_input('Annual Income?', min_value=0)
    income_type = col1.selectbox('Income Type?', options=["Working", "Commercial Associate", "State Servant", "Pensioner", "Student"], index=0)
    education = col1.selectbox('Education Type?', options=["Secondary / secondary special", "Higher Education", "Incomplete higher", "Lower Secondary", "Academic Degree"], index=0)
    family = col1.selectbox('Family Status', options=["Married", "Single / not married", "Civil marriage", "Separated", "Widow"], index=0)
    housing = col2.selectbox('Housing Type?', options=["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment"], index=0)
    temp_birthday = col2.date_input('Birthday', min_value=datetime(1900, 1, 1))
    if temp_birthday:
        birthday = calc_days(temp_birthday)
    temp_employment = col2.date_input('Date of Employment', min_value=datetime(1900, 1, 1))
    if temp_employment:
        employment = calc_days(temp_employment)
    else:
        employment = 365243  # Placeholder for missing employment date
    work_phone = col2.selectbox('Do you have a work phone? (0 for No, 1 for Yes)', options=[0,1], index=0)
    phone = col2.selectbox('Do you have a phone? (0 for No, 1 for Yes)', options=[0,1], index=0)
    email = col2.selectbox('Do you have an email? (0 for No, 1 for Yes)', options=[0,1], index=0)
    occupation = col2.selectbox('Occupation Type?', options=["Laborers", "Core staff", "Sales staff", "Managers", "Drivers"], index=0)
    months = 0

    data = {'Gender': gender, 'Car': car, 'Realty': realty, 'Income': income, 'Income Type': income_type, 'Education': education, 
            'Family': family, 'Housing': housing, 'Birthday': birthday, 'Employment': employment, 'Work Phone': work_phone, 
            'Phone': phone, 'Email': email, 'Occupation': occupation, 'Months': months}
    
    features = pd.DataFrame(data, index=[0])
    
    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)

    # Apply the LabelEncoders to the corresponding columns
    col_names = ['Gender', 'Income Type', 'Education', 'Family', 'Housing', 'Occupation']
    for col in col_names:
        le = label_encoders[col]
        features[col] = le.transform(features[col])

    return features

with header:
    st.title("Credit Approval Prediction App")
    
with features:
    st.header('Input')
    st.write('Please input your information')
    df = user_input()
    st.write("User Input Parameters")
    st.write(df)
    
with model:
    # Loading saved model
    filename = "rf_model.pkl"
    model = pickle.load(open(filename, 'rb'))

    # Prediction function
    def prediction():
        pred = model.predict(df)
        result = 'Approved' if pred[0] == 1 else 'Rejected'
        return result

    # Prediction button
    if st.button("Predict"): 
        result = prediction()
        st.success(f'Your request is {result}')
