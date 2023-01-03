import streamlit as st
# For ML models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import imblearn.over_sampling

# Filter warnings
warnings.filterwarnings('ignore')

# Add a header to the app
st.header("Asian Heart Disease Risk Calculator")

# Load the data
df = pd.read_csv('heart_2020_cleaned.csv')
data = df

# Define a function to get user input
def user_input_features():
  
  st.write("Please fill out the questionnaire below to see if you are at risk of diabetes:")

  st.write("Enter BMI:") 
  BMI = st.slider('', 0.0, 110.0, 55.0, key = "l")
  st.write("You selected this option:",BMI)

  st.write("Have you smoked over 100 cigarettes in your lifetime?") 
  smoke = st.selectbox("Yes or No", ["Yes", "No"], key = "a")
  st.write("You selected this option:",smoke)

  bin_smoke = 0
  if smoke == "Yes":
    bin_smoke = 1
    

  st.write("Are you a heavy drinker? (>14 drinks per week for men, >7 drinks per week for women)") 
  alc = st.selectbox("Yes or No", ["Yes", "No"], key = "b")
  st.write("You selected this option:",alc)

  bin_alc = 0
  if alc == "Yes":
    bin_alc = 1

  st.write("Have you ever had a stroke?") 
  stroke = st.selectbox("(Yes or No", ["Yes", "No"], key = "c")
  st.write("You selected this option:", stroke)
  
  bin_stroke = 0
  if stroke == "Yes":
    bin_stroke = 1
  
  st.write("Of the last 30 days, how many have you expereinced physical pain") 
  physical = st.slider('', 0, 30, 15, key = "1")
  st.write("You selected this option:", physical)


  st.write("Of the last 30 days, how many would you consider \'bad\' days mentally?") 
  mental = st.slider('', 0, 30, 15, key = "2")
  st.write("You selected this option:", mental)

  st.write("What is your sex?") 
  sex = st.selectbox("(Male or Female", ["Male", "Female"], key = "e")
  st.write("You selected this option:", sex)

  bin_sex = 0
  if sex == "Male":
    bin_sex= 1

  st.write("9. Enter your age:") 
  age = st.slider('', 0, 100, 50)
  st.write("You selected this option:", age)

  factor_age = 0
  if age in range(0, 30, 1):
    factor_age = 0
  elif age in range(30,45,1):
    factor_age = 1
  elif age in range(45, 60, 1):
    factor_age = 2
  elif age in range(60, 70, 1):
    factor_age = 3
  elif age in range(70,75,1):
    factor_age = 4
  elif age in range(75, 80,1):
    factor_age = 5
  elif age in range(80, 101, 1):
    factor_age = 6
  
  st.write("10. Have you ever been told you are diabetic?") 
  diabetes = st.selectbox("(Yes or No)", ["Yes", "No"], key = "f")
  st.write("You selected this option:", diabetes)  
  
  bin_diabetes = 0
  if diabetes == "Yes":
    bin_diabetes = 1

  st.write("Have you exercised in the past 30 days?") 
  exercise = st.selectbox("(Yes or No)", ["Yes", "No"], key = "g")
  st.write("You selected this option:", exercise)  

  bin_exercise = 0
  if exercise == "Yes":
    bin_exercise = 1

  st.write("How much do you sleep in a day (on avg):") 
  sleep = st.slider('', 0, 24, 12)
  st.write("You selected this option:", sleep) 
  
  st.write("Have you ever been told you have asthma?") 
  asthma = st.selectbox("(Yes or No)", ["Yes", "No"], key = "h")
  st.write("You selected this option:", asthma)  

  bin_asthma = 0
  if asthma == "Yes":
    bin_asthma = 1

  st.write("Have you ever been told you have kidney disease?") 
  kidney = st.selectbox("(Yes or No)", ["Yes", "No"], key = "i")
  st.write("You selected this option:", kidney)

  bin_kidney = 0
  if kidney == "Yes":
    bin_kidney = 1
  

  data_user = {'BMI': BMI, 'Smoking': smoke, 'AlcoholDrinking	': alc, 'Stroke': stroke, 'PhysicalHealth': physical, 'MentalHealth': mental, 'Sex': sex,  'Diabetic': diabetes, 'PhysicalActivity	': exercise, 'SleepTime': sleep,'AgeCategory': age, 'Asthma': asthma, 'KidneyDisease': kidney}
  features_str = pd.DataFrame(data_user, index=[0])
  st.subheader('Given Inputs : ')
  st.write(features_str)
  
  ##Race Factor for Asians
  bin_race = 3

  features_user = np.array([[bin_race, BMI, bin_smoke, bin_alc, bin_stroke, physical, mental, bin_sex, bin_diabetes, bin_exercise, sleep, factor_age, bin_asthma, bin_kidney]])
  features_user_2 = np.array([[BMI, bin_smoke, bin_alc, bin_stroke, physical, mental, bin_sex, bin_diabetes, bin_exercise, sleep, factor_age, bin_asthma, bin_kidney]])
  st.write(features_user)
  return features_user, features_user_2


# Replace categorical variables with numerical values
data.Smoking.replace(('Yes','No'), (1,0), inplace = True)
data.HeartDisease.replace(('Yes','No'), (1,0), inplace = True)
data.AlcoholDrinking.replace(('Yes','No'), (1,0), inplace = True)
data.Stroke.replace(('Yes', "No"), (1,0), inplace = True)
data.Sex.replace(('Male','Female'), (1,0), inplace = True)
data.Asthma.replace(('Yes','No'), (1,0), inplace = True)
data.Diabetic.replace(('Yes','No','No, borderline diabetes','Yes (during pregnancy)'), (1,0,0,0), inplace = True)
data.PhysicalActivity.replace(('Yes','No'), (1,0), inplace = True)
data.KidneyDisease.replace(('Yes','No'), (1,0), inplace = True)
data.AgeCategory.replace(('18-24','25-29','30-34','35-39','40-44','45-49','50-54',
                          '55-59','60-64','65-69','70-74','75-79','80 or older'),
                         (0,0,1,1,1,2,2,2,3,3,4,5,6), inplace = True)
data.Race.replace(('White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic'),
                          (1,2,3,1,1,2), inplace = True)

column = ['Race','BMI','Smoking','AlcoholDrinking','Stroke','PhysicalHealth','MentalHealth','Sex','Diabetic','PhysicalActivity','SleepTime','AgeCategory','Asthma','KidneyDisease']

# Split the data into training and testing sets
X = data[column]
Y = data.HeartDisease

# Use SMOTE to oversample the minority class
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 0)
X_resampled, Y_resampled = sm.fit_resample(X, Y)

# Split the resampled data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size = 0.2, random_state = 0)

# Train a Decision Tree classifier on the training data
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

# Use the model to make predictions on the user's input
result = user_input_features()
features_users = result[0]
prediction = model.predict(features_users)
prediction_proba = model.predict_proba(features_users)

# Display the prediction and probability to the user
st.subheader("Prediction:")

# Create a dataframe with the prediction
prediction_df = pd.DataFrame(prediction, columns=["Prediction"])

# Add a column with the chances of heart disease
prediction_df["Result"] = prediction_df["Prediction"].map({0: "You are not likely to have heart disease", 1: "You might be at risk for heart disease, further steps are listed below:"})

# Drop the original prediction column
prediction_df = prediction_df.drop("Prediction", axis=1)
st.dataframe(prediction_df)

column2 = ['BMI','Smoking','AlcoholDrinking','Stroke','PhysicalHealth','MentalHealth','Sex','Diabetic','PhysicalActivity','SleepTime','AgeCategory','Asthma','KidneyDisease']

# Select the subset of columns
subset = df[column2]

# Calculate the means and standard deviations of the subset of columns
means = subset.mean()
stds = subset.std()

# Create a dataframe with the means and standard deviations
stats_df = pd.concat([means, stds], axis=1)
stats_df.columns = ['Mean', 'Standard Deviation']

# Display the means and standard deviations
st.header("Further Steps")
st.write("Although our model does not produce answers at 100\% accuracy, we do suggest reaching out to medical professionals if at risk for heart disease.")
st.write("Here is a list of resources to learn more about heart disease. The first link provided offers instructions in multiple languages. Further information is provided on the Health Recommendation Pages")
st.markdown("[National Heart, Lung, and Blood Institute (NHLBI): Heart Disease](https://www.nhlbi.nih.gov/health-topics/heart-disease)")
st.markdown("[Health Resources and Services Administration (HRSA): Heart Disease](https://www.hrsa.gov/health-topics/heart-disease)")
st.markdown("[Centers for Disease Control and Prevention (CDC): Heart Disease Prevention](https://www.cdc.gov/heartdisease/prevention.htm)")
st.text("\n")

st.write("We have provided the means for each category below, we recommend you compare scores to learn which areas you can improve on. More information is provided on the Health Recommendation Page")
st.dataframe(stats_df)
