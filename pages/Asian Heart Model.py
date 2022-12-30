import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st

##Loading Data
train = pd.read_csv('heart_2020_cleaned.csv')

st.write(train)