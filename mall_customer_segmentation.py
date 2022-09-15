from email.policy import default
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("Mall_Customers_Data.csv")
st.subheader("Mall Customer Spending Habit Interactive Model")
st.write(df.head())
st.write(df.describe())
df.shape

st.sidebar.header("Query Parameters")
county = df['County'].unique()
st.sidebar.multiselect('Select County:', county)
gender = df['Gender'].unique()
st.sidebar.multiselect('Select Gender:', gender)
age = df['Age'].unique()
st.sidebar.multiselect('Age:', age)
annual_income = df['Annual Income (Kes)'].unique()
st.sidebar.multiselect('Select Annual Income (Kes):', annual_income)

# county_cust_data=df[['County','CustomerID']]
# county_cust_data.shape
df = df.groupby(['County'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)
df

#Data Manipulation
def manipulate_df(df):
	# Update Gender column to numerical
	df['Gender'] = df['Gender'].map(lambda x: 0 if x == 'male' else 1)
    # Update County column to numerical
	df['County'] = df['County'].map(lambda x: 0 if x == 'Nairobi' else 1)
	# Fill the nan values in the age column
	df['Age'].fillna(value = df['Age'].mean() , inplace = True)
	# Select the desired features
	df= df[['Gender' , 'Age' , 'County', 'Annual Income(Kes)']]
	return df

#Split Data into Train and Test
