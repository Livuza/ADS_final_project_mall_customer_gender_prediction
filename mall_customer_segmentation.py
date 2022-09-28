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

# st.set_page_config(layout="wide")

df = pd.read_csv("Mall_Customers_Data.csv")



# data_frame = sns.load_dataset("Mall_Customers_Data.csv")
# def main():
#     page = st.sidebar.selectbox(
#         "Select a Page",
#         [
#             "Line Plot"
#         ]
#     )
#     linePlot()
# def linePlot():
#     fig = plt.figure(figsize=(10, 4))
#     sns.lineplot(x = "distance", y = "mass", data = data_frame)
#     st.pyplot(fig)
# if __name__ == "__main__":
#     main()

# markdown(f"TRM Mall Customer Gender Prediction Based on Their Spending Score")
colmn1= st.sidebar
colmn2, colmn3 = st.columns((2,1))

colmn1.markdown("""
## File Uploading
""")
colmn1.markdown("""
[Check CSV File Example Template](https://raw.githubusercontent.com/Livuza/ADS_final_project_mall_customer_spending_score/master/Mall_Customers_Data.csv)
""")

uploaded_file=colmn1.file_uploader("Upload your input template file (.csv files only)", type= ["csv"])

colmn1.markdown("Mall Customer Dataset")

st.sidebar.write(df)
# st.write(df.describe())
# df.shape
# st.table(df.head())
# st.dataframe(df.style.highlight_max(axis=0))
# st.header("Query Parameters")
# st.markdown(f"by Charles Livuza")
# county = list(df["County"].drop_duplicates())

# county_choice = df['County'].unique()
# st.sidebar.multiselect('Select County:', county)
# df = df[df['County'].isin(county_choice)]
# # st.table(df.head())
# # st.balloons()
# gender_choice = df['Gender'].unique()
# st.sidebar.multiselect('Select Gender:', gender_choice)
# age_choice = df['Age'].unique()
# st.sidebar.multiselect('Age:', age_choice)
# annual_income_choice = df['Annual Income (Kes)'].unique()
# st.sidebar.multiselect('Select Annual Income (Kes):', annual_income_choice)
# st.sidebar.button("Check Score")

# Dropping the CustomerID column
# st.sns.heatmap(df.corr(),annot=True)
df.drop('CustomerID', axis = 1, inplace = True)
# df

# Encode the  ordinal categorical variable column into numbers to aid training of the model
df['County']= df['County'].map({'Nairobi':0, 'Mombasa':1, 'Kisumu':2})
df['Gender']= df['Gender'].map({'Male':0, 'Female':1})

# Remove Outliers
# df = df.drop(df[df["x"]==0].index)
# df = df.drop(df[df["y"]==0].index)
# df = df.drop(df[df["z"]==0].index)

# splitting the data into the columns which need to be trained(X) and the target column(y)
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
model_df = df.copy()
X = model_df.drop(['Gender'], axis=1)
y = model_df['Gender']

# model_df
# splitting data into training and testing data with 20 % & 80% of data as testing data and training data respectively
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# importing the random forest classifier model and training it on the dataset
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# predicting on the test dataset
y_pred = classifier.predict(X_test)
  
# finding out the accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
# score

# pickling the model
import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()

import pickle
from PIL import Image

# loading in the model to predict on the data
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

def welcome():
    return 'welcome all'

# defining the function which will make the prediction using 
# the data which the user inputs
def prediction(age, annual_income, spending_score, county):  
   
    prediction = classifier.predict(
        [[age, annual_income, spending_score, county]])
    print(prediction)
    return prediction
# this is the main function in which we define our webpage 
def main():
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:padding:13px">
    <h2 style ="color:white;text-align:center;">Customer Gender Predection Machine Learning App </h2>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)


    import base64
    image = Image.open('mall.jpeg')
    st.image(image, caption='A mall')

    # the following lines create select boxes in which the user can enter 
    # the data required to make the prediction
    col1, col2 = st.columns(2)
    with col1:
     age = st.selectbox("What is the age of the customer?", model_df['Age'].unique())
     annual_income = st.selectbox("What is the the customer's Annual Income in Kes?", model_df['Annual Income (Kes)'].unique())
    with col2:
     spending_score = st.slider("What is the customer's spending score (1-100)")
     county = st.selectbox("From which county did the customer shop?", model_df['County'].unique())
     result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(age, annual_income, spending_score, county)
        if result == 0:
            # st.success('The gender prediction is {}'.format(result))
            st.success("The predicted Gender is Male")
        else:
            # st.success('The gender prediction is {}'.format(result))
            st.success("The predicted Gender is Female")
     
if __name__=='__main__':
    main()
st.markdown("This model predicts the mall customer gender depending on age, annual_salary and county. The data used was from the counties of Nairobi, Mombasa and Kisumu.")

# plt.figure(figsize=(20,10))

# plt.subplot(2,2,1)
# sns.barplot(x=df.groupby('Gender')['Spending Score (1-100)'].mean().sort_values(ascending=True).index,y=df.groupby('Gender')['Spending Score (1-100)'].mean().sort_values(ascending=True).values)
# plt.title('Customer Spending Based on Gender')

# Delete from here when done 

# code = """for i in range(2,11,2):
# 		   print(i)
# """
# code(code, language = "python")


# b = button("Save")
# if b:
# 	success("Your submission has been saved successful")
# 	balloons()

# # st.button("Save", key = "new-key")

# # radio buttons
# status =radio("What is your status", ("Attended", "Didn't Attended"))
# if status == "Attended":
# 	success("Thank you for attending")
# else:
# 	error("Kindly attend the next meeting without fail")


# county_cust_data=df[['County','CustomerID']]
# county_cust_data.shape
# df = df.groupby(['County'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)
# df