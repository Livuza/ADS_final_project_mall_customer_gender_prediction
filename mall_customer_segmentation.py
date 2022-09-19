from email.policy import default
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("Mall_Customers_Data.csv")
st.subheader(f"Mall Customer Spending Habit Interactive Model")
st.write(df.head())
# st.write(df.describe())
# df.shape
# st.table(df.head())
# st.dataframe(df.style.highlight_max(axis=0))
st.sidebar.header("Query Parameters")
county = list(df["County"].drop_duplicates())
county_choice = df['County'].unique()
st.sidebar.multiselect('Select County:', county)
df = df[df['County'].isin(county_choice)]

st.table(df.head())
# st.balloons()
gender_choice = df['Gender'].unique()
st.sidebar.multiselect('Select Gender:', gender_choice)
age_choice = df['Age'].unique()
st.sidebar.multiselect('Age:', age_choice)
annual_income_choice = df['Annual Income (Kes)'].unique()
st.sidebar.multiselect('Select Annual Income (Kes):', annual_income_choice)
# Delete from here when done 

code = """for i in range(2,11,2):
		   print(i)
"""
st.code(code, language = "python")


b = st.button("Save")
if b:
	st.success("Your submission has been saved successful")
	# st.balloons()

# st.button("Save", key = "new-key")

# radio buttons
status =st.radio("What is your status", ("Attended", "Didn't Attended"))
if status == "Attended":
	st.success("Thank you for attending")
else:
	st.error("Kindly attend the next meeting without fail")


# county_cust_data=df[['County','CustomerID']]
# county_cust_data.shape
df = df.groupby(['County'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)
df
image = Image.open('mall.jpeg')
st.sidebar.image(image, caption='Mall')
#Data Manipulation



#Split Data into Train and Test

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i0.wp.com/biznakenya.com/wp-content/uploads/2018/07/Thika-Road-Mall.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 