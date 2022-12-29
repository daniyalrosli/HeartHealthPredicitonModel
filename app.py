
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('PimaIndiansD.csv')
df2 = pd.read_csv('Diabetes2.csv')

# Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(df[['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']], df['Outcome'], test_size=0.3, random_state=109)#Creating the model
# logisticRegr = LogisticRegression(C=1, max_iter = 1000)
# logisticRegr.fit(X_train, y_train)
# y_pred = logisticRegr.predict(X_test)


# #Saving the Model
# pickle_out = open("logisticRegr.pkl", "wb") 
# pickle.dump(logisticRegr, pickle_out) 
# pickle_out.close()


# pickle_in = open('logisticRegr2.pkl', 'rb')
# classifier = pickle.load(pickle_in)



# Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(df[['AGE', 'Urea','Cr','HbA1c','Chol','TG','HDL','LDL','VLDL','BMI']], df['CLASS'], test_size=0.3, random_state=100)#Creating the model
# logisticRegr = LogisticRegression(C=1, max_iter = 100000)
# logisticRegr.fit(X_train, y_train)
# y_pred = logisticRegr.predict(X_test)


# #Saving the Model
# pickle_out2 = open("logisticRegr2.pkl", "wb") 
# pickle.dump(logisticRegr, pickle_out) 
# pickle_out2.close()

pickle_in = open('logisticRegr2.pkl', 'rb')
classifier = pickle.load(pickle_in)

# st.sidebar.header('Diabetes Prediction')
# select = st.sidebar.selectbox('Select Form', ['Form 1'], key='1')
# if not st.sidebar.checkbox("Hide", True, key='Predict'):
#     st.title('Diabetes Prediction')
#     name = st.text_input("Name:")
#     pregnancy = st.number_input("No. of times pregnant:")
#     glucose = st.number_input("Plasma Glucose Concentration :")
#     bp =  st.number_input("Diastolic blood pressure (mm Hg):")
#     skin = st.number_input("Triceps skin fold thickness (mm):")
#     insulin = st.number_input("2-Hour serum insulin (mu U/ml):")
#     bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):")
#     dpf = st.number_input("Diabetes Pedigree Function:")
#     age = st.number_input("Age:")
    
# submit = st.button('Predict')

# if submit:
#         prediction = classifier.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
#         if prediction == 0:
#             st.write('Congratulation',name,'You are not diabetic')
#         else:
#             st.write(name," L U have diabetes")


st.sidebar.header('Diabetes Prediction')
select = st.sidebar.selectbox('Select Form', ['Form 1'], key='1')
if not st.sidebar.checkbox("Hide", True, key='Predict'):
    st.title('Diabetes Prediction')
    name = st.text_input("Name:")
    gender = st.text_input("Gender (M/F)")
    age = st.number_input("Age")
    urea =  st.number_input("Urea")
    cr = st.number_input("Creatinine:")
    hba1c = st.number_input("HbA1c")
    chol = st.number_input("Cholesterol")
    tg = st.number_input("TG")
    hdl = st.number_input("HDL")
    ldl = st.number_input("LDL")
    vldl = st.number_input("VLDL")
    bmi = st.number_input("Body Mass Index")
    
submit = st.button('Predict')

if submit:
        prediction = classifier.predict([name, gender, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi])
        if prediction == 'N':
            st.write('Congratulation',name,'You are not diabetic')
        else:
            st.write(name," L U have diabetes")