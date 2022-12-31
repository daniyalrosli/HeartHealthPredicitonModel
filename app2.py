import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
sns.set(rc={'axes.facecolor':'#F5F6F4', 'figure.facecolor':'#F5F6F4'})



col1, col2 = st.columns([1,6])

with col1:
   st.image("HeartGif.gif")

with col2:
    new_title = '<p style="font-family:sans-serif; color:#1E000E; font-size: 70px;">Heart Health Model</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    
st.write("Hosting multiple models to help predict heeart health specifically geared towards asian amaericans")

df = pd.read_csv('DeathsbyRace.csv')
fig = plt.figure(figsize=(5, 5))
sns.barplot(data=df, y="Race", x="Percent(%) of Deaths", color="#C27070")
st.pyplot(fig)

