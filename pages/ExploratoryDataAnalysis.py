import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st
import sklearn as sklearn

sns.set(rc={'axes.facecolor':'#F5F6F4', 'figure.facecolor':'#F5F6F4'})


st.header("Exploratory Data Analytics")

st.write("On this page we will walkthrough our data analytics process as we worked towards building our model for the Asian Heart Health Model")

st.write("First lets load in our data set and filter by race = 'Asian'")
##Loading Data
df = pd.read_csv('heart_2020_cleaned.csv')

newdf = df[(df.Race == "Asian")]
st.write(newdf)



st.write("After taking a look at the data set, we can determine our continous and discrtete vraibles. This will help when we graph later.")
col1, col2 = st.columns(2)

continous = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
discrete = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory',
       'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth','Asthma', 'KidneyDisease', 'SkinCancer']
## Update the lists
with col1:
    st.header("Discrete")
    s = ""
    for i in discrete:
        s += "- " + i + "\n" 
    st.markdown(s) 
   


with col2:
    st.header("Continuous")
    s = ""
    for i in continous:
        s += "- " + i + "\n" 
    st.markdown(s) 


import warnings
warnings.filterwarnings('ignore')
fig,ax = plt.subplots(len(continous),2,figsize=(30,20))
for index,i in enumerate(continous):
    sns.distplot(newdf[i],ax=ax[index,0],color='#540B0E')
    stats.probplot(newdf[i],plot=ax[index,1])
    
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.suptitle("Visualizing continuous columns",fontsize=30)

st.write(fig)

##Preprocessing - Transforming Data Set using OrdinalEncoder 
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
enc.fit(newdf[discrete])
newdf[discrete] = enc.transform(newdf[discrete])
st.write(newdf)

##Chi-Square test based on the relationship of our variables
##This is just for us to understand our data more


from scipy.stats import chi2_contingency
for feature in discrete:
  stat, p, dof, expected = chi2_contingency(pd.crosstab(newdf[feature],newdf['HeartDisease']))
  alpha = 0.05
  print ( "p value is " + str (p))
  if p<=alpha:
    print ( 'Dependent (reject H0)' )
  else :
    print ( 'Independent (H0 holds true)' )

st.write('in our data set we found a statistically significant relationship between our target variable and our discrete variables')
st.write('Lets build a correlation table to better udnerstand the sterength of the relationships')

st.caption('in this case well have an undefinded correlation for race sicne we have uniformity')
correlation = newdf.corr()
st.table(correlation["HeartDisease"].sort_values(ascending = False))

##Correlation Matrix
k= 18
cols = correlation.nlargest(k,'HeartDisease')['HeartDisease'].index
print(cols)
cm = np.corrcoef(newdf[cols].values.T)
mask = np.triu(np.ones_like(newdf.corr()))
f , ax = plt.subplots(figsize = (14,12))
graph = sns.heatmap(cm,mask=mask, vmax=.8, linewidths=0.01,square=True,annot=False,cmap='rocket',
            linecolor="#F5F6F4",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)

st.pyplot(f)


st.text(" \n")
st.text(" \n")
st.text(" \n")

##Model Imports
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score

y=newdf['HeartDisease']
newdf.drop('HeartDisease',axis=1,inplace=True)

X_train, X_test, y_train, y_test=train_test_split(newdf,y,test_size=0.1,random_state=12)

models = [KNeighborsClassifier(), LogisticRegression(),ExtraTreesClassifier()]
scores = dict()

for m in models:
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)

    st.write(f'model: {str(m)}')
    st.write(f'Accuracy_score: {accuracy_score(y_test,y_pred)}')
    st.write('-'*30, '\n')

