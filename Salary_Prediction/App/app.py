#!/usr/bin/env python
# coding: utf-8

# In[2]:

# In[3]:


import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import csv
from sklearn.linear_model import LinearRegression

df=pd.read_csv("salary.csv")
df=df.drop(['Timestamp','Name','University Name'],axis=1)
#print(df.head())
cat={"Tier1":1,"Tier2":2,"Tier3":3}
df['University_Cat']=df.University_Cat.map(cat)
#print(df.head())
q={"B.Tech":1,"BE":2,"B.SC":3,"M.Tech":4}
df['Qualification']=df.Qualification.map(q)
#print(df.head())
df['University_Cat'].fillna(2,inplace=True)

x=df.drop('Salary',axis=1) #independent variables
y=df['Salary']       #dependent variable to be predicted

model=LinearRegression()
model.fit(x,y)

st.markdown("<h1 style='text-align: center; color: red;'><b>----EXPECTED SALARY PREDICTION-----</b></h>",unsafe_allow_html=True)
name=st.text_input("Enter Candidate Name :")
university=st.text_input("Enter University Name :")
cat=st.selectbox("Enter University category : ",('Tier1','Tier2','Tier3'))
if cat=='Tier1':
    cat=1
elif cat=='Tier2':
    cat=2
elif cat=='Tier3':
    cat=3
bcg=st.selectbox("Enter number of Backlogs : ",(0,1))
cgpa=st.number_input("Enter CGAP : ")
yop=st.selectbox("Year of Passout : ",(2020,2021))
quali=st.selectbox("Enter Highest Degree : ",('B.Tech','BE','B.SC','M.Tech'))
if quali=='B.Tech':
    quali=1
elif quali=='BE':
    quali=2
elif quali=='B.SC':
    quali=3
elif quali=='M.Tech':
    quali=4

if st.button("SUBMIT"):

        st.markdown(name+" from "+university)
        st.markdown("<h1 style='color: green;'><b>Expected Salary :</b></h1>",unsafe_allow_html=True)
        output=round(model.predict([[cat,bcg,cgpa,yop,quali]])[0],2)
        st.write(output)
        if cat==1:
            cat='Tier1'
        elif cat==2:
            cat='Tier2'
        elif cat==3:
            cat='Tier3'
        
        if quali==1:
            quali='B.Tech'
        elif quali==2:
            quali='BE'
        elif quali==3:
            quali='B.SC'
        elif quali==4:
            quali='M.Tech'

 
        with open('empdetails.csv', 'a', newline='') as file:
                  writer = csv.writer(file)
                  #writer.writerow(["Name", "University","U-Category","Degree","Backlogs","CGPA","YOP","Salary"])
                  writer.writerow([name,university,cat,quali,bcg,cgpa,yop,output])


# In[ ]:




