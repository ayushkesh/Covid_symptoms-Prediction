# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:08:43 2020

@author: Ayush
"""

import numpy as np
import pandas as pd
import streamlit as st
import pickle

from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('corona_data.csv')

from sklearn.preprocessing  import LabelEncoder
sc = LabelEncoder()
fever = df['Fever']
df['Fever'] = sc.fit_transform(fever)

X = df.drop('infection_Probability', axis = 1)
y = df.infection_Probability

classifier = RandomForestClassifier()
classifier.fit(X,y)

from PIL import Image
#pickle_in = open('covid19R_model.pkl', 'rb')
#classifier = pickle.load(pickle_in)

def welcome():
    return 'Stay_Home Stay_Safe'

def predict_class(Age, Fever, BodyPains, RunnyNose, Difficulty_in_Breath):
    pred = classifier.predict([[Age, Fever, BodyPains, RunnyNose, Difficulty_in_Breath]])
    return pred

def main():
    html_temp = """
    <div style="background-color:cyan;padding:10px">
    <h2 style="color:black;text-align:center;">Covid Symptoms Prediction</h2>
    </div>
    """
  
    st.markdown(html_temp , unsafe_allow_html= True)
    image = Image.open('covid_1.PNG')
    st.image(image, use_column_width=True,format='PNG')
    
    Age = st.slider('Age', 0,100)
    Fever = st.slider('Fever', 96,110)
    BodyPains = st.text_input('Body Pains 0(No)- 1(Yes)'," ")
    RunnyNose = st.text_input('RunnyNose 0(No)-1(Yes)'," ")
    Difficulty_in_Breath = st.text_input('Difficulty_in_Breath 0(No)-1(Yes)'," ")             
    result =""
    if st.button('Predict'):
         result=predict_class(int(Age),int(Fever),int(BodyPains),int(RunnyNose),int(Difficulty_in_Breath))
         if result==0:
             result="Congrats you are safe,dont have covid-19 symptoms"
         else:
             result="Sorry,you have covid-19 symptoms, please consult doctor"
            
  
    st.success('{}'.format(result))
    if st.button("About"):
        st.text("Stay_safe Stay_safe")
        st.text("Github link https://github.com/ayushkesh/Covid_symptoms-Prediction")   
        
    html_temp1 = """
    <div style="background-color:#f63366">
    <p style="color:white;text-align:center;" >Designe & Developed By: <b>Ayush Kumar</b> </p>
    </div>
    """
    st.markdown(html_temp1,unsafe_allow_html=True)
if __name__ == '__main__':
    
    main()
        
                 
