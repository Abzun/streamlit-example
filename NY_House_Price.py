#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import seaborn as sns
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pickle

#loading Model
loaded_model = pickle.load(open(r'C:\Users\Edwin\Downloads\ny_model_lm.sav','rb'))

#function for using model

def price_prediction(input_data):
    new_data = np.array(input_data, ndmin = 2)
    new_dataframe = pd.DataFrame(new_data, columns = ['postcode', 'bedroom_number', 'bathroom_number',
       'price_per_unit', 'living_space', 'sqft'])
    prediction = loaded_model.predict(new_dataframe)
    
    return prediction

def main():
    st.title('NY House Prizing!!1 EXTRAVAGANZA ')
    
    #INPUTS FROM USER
    
    postcode = st.number_input('Enter Zipcode')

    bedroom_number = st.number_input('Enter Bedrooms')
    bathroom_number = st.number_input('Enter Bathroom')
    st.info('I believe this is just dollars/Sqft. The average in LI is $400')
    price_per_unit = st.number_input('Enter something price per unit') 


    st.info('When house plan sellers refer to Total Living square feet, they are referring to the “living area” of the home. This can be thought of as the area that will be heated or cooled. It is called the living area because this is where you spend your time. An attic, while a useful storage area, is not living space.')
    living_space = st.number_input('Enter Living Space')
    sqft = st.number_input('Enter Sqft of Property')
    
    output = ''
    
    if st.button('Magic Price Predictor-gizmo'):
        output = price_prediction([postcode, bedroom_number, bathroom_number,
                                   price_per_unit, living_space, sqft])
        st.success(round(output[0],2))
    
if __name__ == '__main__':
    main()

