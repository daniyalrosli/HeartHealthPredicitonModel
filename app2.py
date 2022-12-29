import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.write(""" # Heart Disease Risk Prediction App
This app predicts your risk for heart disease given several parameters 
""")


##User inputs for the model
##Sliders and dropdowns 
def user_input_features():
    
    st.write("""**1. Select Age :**""") 
    age = st.slider('', 0, 100, 25)
    st.write("""**You selected this option **""",age)
    
    st.write("""**2. Select Gender :**""")
    sex_str = st.selectbox("(Male or Female)",["Male","Female"])
    st.write("""**You selected this option **""", sex_str)

    st.write("""**3. Select Chest Pain Type :**""")
    cp_str = st.selectbox("(Typical Angina, Atypical Angina, Non—anginal Pain, or Asymptotic)",["Typical Angina", "Atypical Angina", "Non—anginal Pain", "Asymptotic"])
    st.write("""**You selected this option **""",cp_str)
    
    st.write("""**4. Select Resting Blood Pressure :**""")
    trestbps = st.slider('In mm/Hg unit', 0, 200, 110)
    st.write("""**You selected this option **""",trestbps)
    
    st.write("""**5. Select Serum Cholesterol :**""")
    chol = st.slider('In mg/dl unit', 0, 600, 115)
    st.write("""**You selected this option **""",chol)
    
    st.write("""**6. Maximum Heart Rate Achieved (THALACH) :**""")
    thalach = st.slider('', 0, 220, 115)
    st.write("""**You selected this option **""",thalach)
    
    st.write("""**7. Exercise Induced Angina (Pain in chest while exersice) :**""")
    exang_str = st.selectbox("(Yes or No)", ["Yes","No"])
    st.write("""**You selected this option **""",exang_str)
    
    st.write("""**8. Oldpeak (ST depression induced by exercise relative to rest) :**""")
    oldpeak = float(st.slider('', 0.0, 10.0, 2.0))
    st.write("""**You selected this option **""",oldpeak)
    
    st.write("""**9. Slope (The slope of the peak exercise ST segment) :**""")
    slope = st.selectbox("(Select 0, 1 or 2)",["0","1","2"])
    st.write("""**You selected this option **""",slope)
    
    st.write("""**10. CA (Number of major vessels (0-3) colored by flourosopy) :**""")
    ca = st.selectbox("(Select 0, 1, 2 or 3)",["0","1","2","3"])
    st.write("""**You selected this option **""",ca)
    
    st.write("""**11. Thal :**""")
    thal = st.slider('', 0.0, 10.0, 5.0)
    st.write("""**You selected this option **""",thal)
    
    
    data = {'age': age, 'sex': sex_str, 'cp': cp_str, 'trestbps': trestbps, 'chol': chol, 'thalach': thalach, 'exang': exang_str, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal,}
    
    ##Convert to numerical inputs 
    ##Gender
    sex = ""
    if sex_str == "Male":
        sex = "1"
    else:
        sex = "0"

    ##Chest Pain    
    cp = ""
    if cp_str == "Typical Angina":
      cp = "1"
    elif cp_str == "Atypical Angina":
      cp = "2"
    elif cp_str == "Non-anginal Pain":
      cp = "3"
    else:
      cp = "4"

    ##Exersize induced angina
    exang = ""
    
    if exang_str == "Yes":
      exang = "1"
    else:
      exang = "0"

    
    data_num = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal,}
    features = pd.DataFrame(data_num, index=[0])
    return features


df = user_input_features()
st.subheader('Given Inputs : ')
st.write(df)

heart = pd.read_csv("HeartDisease.csv")
X = heart.iloc[:,0:11].values
Y = heart.iloc[:,[11]].values

model = RandomForestClassifier()
model.fit(X, Y)

prediction = model.predict(df)
st.subheader('Prediction :')
df1=pd.DataFrame(prediction,columns=['0'])
df1.loc[df1['0'] == 0, 'Chances of Heart Disease'] = 'No'
df1.loc[df1['0'] == 1, 'Chances of Heart Disease'] = 'Yes'
st.write(df1)

prediction_proba = model.predict_proba(df)
st.subheader('Prediction Probability in % :')
st.write(prediction_proba * 100)