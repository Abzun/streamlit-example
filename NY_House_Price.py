#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import seaborn as sns
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pickle
import sklearn

#loading Model
loaded_model = pickle.load(open('https://github.com/Abzun/streamlit-example/blob/master/ny_model_lm.sav','rb'))

#loading in the data frame to usage 
data_ny = pd.read_csv(r'C:\Users\Edwin\Downloads\zillow NY for-sale properties.csv')
housey = data_ny.drop(columns = ['property_url','property_id', 'apartment'
                             ,'broker_id','property_status'
                             , 'year_build', 'total_num_units', 'listing_age'
                             ,'RunDate', 'agency_name', 'agent_name', 'agent_phone'
                             ,'is_owned_by_zillow','state','property_type']) # unnecessary columns removed
#____________ZIPCODES _LONG ISLAND__________________________________________________________________

lst = [11930.0, 11701.0, 11708.0, 11703.0,
       11704.0, 11707.0, 11933.0, 11743.0, 11963.0, 11706.0, 11751.0, 11930.0,
     11743.0, 11777.0, 11715.0, 11780.0, 11772.0, 11702.0, 11980.0, 11934.0,
     11722.0, 11749.0, 11760.0, 11782.0, 11724.0, 11743.0, 11770.0, 11768.0,
     11772.0, 11729.0, 11780.0, 11746.0, 11937.0, 11730.0, 11939.0, 11940.0,
     11731.0, 11772.0, 11942.0, 11733.0, 11967.0, 11768.0, 11706.0, 11770.0,
     11782.0, 6390.0, 11768.0, 11702.0, 11763.0, 11739.0, 11946.0, 11743.0, 11760.0,
     11788.0, 11780.0, 11757.0, 11954.0, 11743.0, 11746.0, 11747.0, 11750.0,
     11760.0, 11751.0, 11752.0, 11754.0, 11743.0, 11755.0, 11961.0, 11779.0,
     11779.0, 11743.0, 11743.0, 11950.0, 11951.0, 11750.0, 11775.0, 11805.0,
     11953.0, 11764.0, 11766.0, 11935.0, 11956.0, 11701.0, 11703.0, 11706.0,
     11713.0, 11963.0, 11757.0, 11772.0, 11968.0, 11702.0, 11702.0, 11770.0,
     11770.0, 11733.0, 11951.0, 11784.0, 11957.0, 11717.0, 11963.0, 11706.0,
     11777.0, 11776.0, 11777.0, 11930.0, 11950.0, 11778.0, 11779.0, 11963.0,
     11780.0, 11754.0, 11789.0, 11964.0, 11965.0, 11787.0, 11788.0, 11789.0,
     11722.0, 11719.0, 11746.0, 11970.0, 11720.0, 11969.0, 11790.0, 11794.0,
     11733.0, 11768.0, 11792.0, 11976.0, 11704.0, 11707.0, 11706.0, 11717.0,
     11702.0, 11743.0, 11795.0, 11796.0, 11978.0, 11978.0, 11798.0, 11507.0,
     11003.0, 11003.0, 11509.0, 11510.0, 11762.0, 11558.0, 11050.0, 11001.0,
     11001.0, 11793.0, 11548.0, 11514.0, 11771.0, 11771.0, 11561.0, 11735.0,
     11548.0, 11576.0, 11577.0, 11758.0, 11554.0, 11732.0, 11518.0, 11596.0,
     11096.0, 11736.0, 11737.0, 11774.0, 11001.0, 11002.0, 11003.0, 11010.0,
     11530.0, 11531.0, 11535.0, 11536.0, 11599.0, 11040.0, 11530.0, 11542.0,
     11545.0, 11547.0, 11020.0, 11021.0, 11022.0, 11023.0, 11024.0, 11025.0,
     11026.0, 11027.0, 11021.0, 11050.0, 11023.0, 11558.0, 11550.0, 11551.0,
     11557.0, 11557.0, 11557.0, 11598.0, 11802.0, 11803.0, 11804.0, 11815.0,
     11819.0, 11854.0, 11855.0, 11558.0, 11756.0, 11853.0, 11024.0, 11022.0,
     11020.0, 11042.0, 11570.0, 11040.0, 11771.0, 11791.0, 11561.0, 11560.0,
     11561.0, 11564.0, 11040.0, 11762.0, 11559.0, 11765.0, 11530.0, 11553.0,
     11732.0, 11753.0, 11771.0, 11791.0, 11590.0, 11040.0, 11041.0, 11042.0,
     11043.0, 11044.0, 11099.0, 11510.0, 11710.0, 11040.0, 11758.0, 11566.0,
     11040.0, 11580.0, 11793.0, 11581.0, 11804.0, 11545.0, 11548.0, 11568.0,
     11771.0, 11771.0, 11791.0, 11569.0, 11050.0, 11051.0, 11052.0, 11053.0,
     11054.0, 11055.0, 11570.0, 11571.0, 11572.0, 11592.0, 11530.0, 11531.0,
     11576.0, 11545.0, 11548.0, 11576.0, 11577.0, 11021.0, 11023.0, 11021.0,
     11050.0, 11579.0, 11735.0, 11001.0, 11550.0, 11530.0, 11559.0, 11791.0,
     11050.0, 11555.0, 11556.0, 11588.0, 11020.0, 11545.0, 11732.0, 11771.0,
     11580.0, 11581.0, 11582.0, 11583.0, 11552.0, 11590.0, 11593.0, 11594.0,
     11595.0, 11597.0, 11596.0]

#____________ZIPCODES ______LONG ISLAND _____________________________________________________________
#function for using model

#Long Island homes
li_homes = housey[housey['postcode'].isin(lst)]

def home_same_price(g):
    h = housey.loc[housey['price'] == g]
    return h
def Li_homes(g):
    h = li_homes.loc[li_homes['price'] == g]
    return h

def price_prediction(input_data):
    new_data = np.array(input_data, ndmin = 2)
    new_dataframe = pd.DataFrame(new_data, columns = ['postcode', 'bedroom_number', 'bathroom_number',
       'price_per_unit', 'living_space', 'sqft'])
    prediction = loaded_model.predict(new_dataframe)
    
    return prediction

def main():
    st.set_page_config(

         layout="wide",
    
     )
    col1, col2 = st.columns(2)
		
    st.sidebar.title('NY House Prizing!!1 EXTRAVAGANZA ')
    
    #INPUTS FROM USER
    
    postcode = st.sidebar.number_input('Enter Zipcode')

    bedroom_number = st.sidebar.number_input('Enter Bedrooms')
    bathroom_number = st.sidebar.number_input('Enter Bathroom')
    price_per_unit = st.sidebar.number_input('Enter something price per unit') 


    
    living_space = st.sidebar.number_input('Enter Living Space')
    
 
    sqft = st.sidebar.number_input('Enter Sqft of Property')
    
    with col2:
        st.write('**ZipCode** - Zipcode')
        st.write('___')
        st.write('**Bedroom** and **Bathroom** - how many rooms for each?')
        st.write('___')
        st.write('**Price per Unit** I believe this is just dollars/Sqft. For LI try $200-$400')
        st.write('___')
        st.write('**Living Space Definition** - When house plan sellers refer to Total Living square feet, they are referring to the “living area” of the home. This can be thought of as the area that will be heated or cooled. It is called the living area because this is where you spend your time. An attic, while a useful storage area, is not living space.')
        st.write('___')
        st.write('**Living Space** - Try to keep living space 60%-80% of the Sqft of the Property ')
        st.write('___')
        st.write('**SQFT** - size of the property')
        st.write('___')
	
    if st.sidebar.button('Magic Price Predictor-gizmo'):
        with col1:
            st.info('prediction magik is: ')

            output = np.array(price_prediction([postcode, bedroom_number, bathroom_number,
                                   price_per_unit, living_space, sqft]))
            st.success(round(output[0],2))
            st.info('These are houses with the similar Price point, I rounded to the nearest Ten Thousands')
            st.dataframe(home_same_price(round(output[0],-4)))

            st.info('These are homes in Long Island')
            st.dataframe(Li_homes(round(output[0],-4)))
	
    st.sidebar.info('When you get a price predicted, youll also be given other addresses based off the price point')
    st.info('To Do: Obviously to make it prettier, maybe compare learning models with different attributes. Im thinking of doing a simplier model to just do prediction based off of just Sqft')
    st.success('To Do: maybe just have options to see just LI houses. fitler panda  - completed')
    st.info('To Do: yesss')
if __name__ == '__main__':
    main()
