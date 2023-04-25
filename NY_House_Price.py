#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import seaborn as sns
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pickle
# import sklearn
import pydeck as pdk
from scipy import stats
#loading Model

loaded_model = pickle.load(open('https://rawgithub.com/Abzun/streamlit-example/blob/master/ny_model_lm.sav', 'rb'))

#loading in the data frame to usage 
data_ny = pd.read_csv(r'C:\Users\Edwin\Downloads\zillow NY for-sale properties.csv')
housey = data_ny.drop(columns = ['property_url','property_id', 'apartment'
                             ,'broker_id','property_status'
                             , 'year_build', 'total_num_units', 'listing_age'
                             ,'RunDate', 'agency_name', 'agent_name', 'agent_phone'
                             ,'is_owned_by_zillow','state']) # unnecessary columns removed

housey = housey[~housey.isnull().any(axis=1)]   #  only have rows with no null or missing values. 
ny = housey.copy()
ny = ny[(np.abs(stats.zscore(ny['price'])) < 3)]

#____________ZIPCODES _LONG ISLAND__________________________________________________________________

lst = [11930.0, 11701.0, 11708.0, 11703.0,
       11704.0, 11707.0, 11933.0, 11743.0, 11963.0, 11706.0, 11751.0, 11930.0,11520.0,
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

# g is an input being provided 

def home_same_price(g): # house prices for all NY houses
    h = housey.loc[housey['price'] == g]
    return h
def ny_homes_price(price,df):
    h = df[(price-100000 < df.price) & (df.price < price + 100000)]
    return h

def Li_homes_price(price): # li house prices
    h = li_homes[(price-50000 < li_homes.price) & (li_homes.price < price + 50000)]
    return h

def town_df(g):#Town df
    h = li_homes[li_homes.postcode == g]
    return h

def town_price(price,df):# Town prices 
    h = df[(price-50000 < df.price) & (df.price < price + 50000)]
    return h


def price_prediction(input_data):
    new_data = np.array(input_data, ndmin = 2)
    new_dataframe = pd.DataFrame(new_data, columns = ['postcode', 'bedroom_number', 'bathroom_number',
       'price_per_unit', 'living_space', 'sqft'])
    prediction = loaded_model.predict(new_dataframe)
    
    return prediction

def main():
  

    #INPUTS FROM USER
    with st.sidebar:
        st.title('NY House Form')
        with st.form("house_form"):
            postcode = st.number_input('Enter Zipcode',step = 1)
            bedroom = st.slider('Enter Bedroom number', min_value = 1, max_value = 8, value = 2, step = 1)
            bathroom = st.slider('Enter bathroom', min_value = 1, max_value = 5, value = 2, step = 1)
            
            price_per_unit = st.number_input('Enter something price per unit', min_value = 0,step = 1) 
            living_space = st.number_input('Enter Living Space',min_value = 0,step = 1)
            sqft = st.number_input('Enter SQFT of Property',min_value = 0,step = 1)
            
            radio = st.radio("Data Tabs", ('Dataframes', 'Graphs'))
            st.form_submit_button("Submit")
            st.info('Be sure to resubmit when switching tabs')
    #INPUTS FROM USER
            output = np.array(price_prediction([postcode, bedroom, bathroom,
                                   price_per_unit, living_space, sqft]))


	
    
    
            
    tab1, tab2, tab3 = st.tabs(["dataframe", "graphs", "sidebar info"])
    if radio == 'Dataframes':
        with tab1:
         
     
            st.info('Price Prediction: ')
            #output = np.array(price_prediction([postcode, bedroom, bathroom,
             #                      price_per_unit, living_space, sqft]))
	       
            st.success(round(output[0],-3))
            st.info('These are houses with the similar Price point, +/- $50,000')
            st.dataframe(Li_homes_price(round(output[0],-4)))
            st.info('These are homes in the same Town')
            st.dataframe(town_price(round(output[0],-4), town_df(postcode) ))

            xz = round(output[0],-4)
            
    if radio == 'Graphs':
        
        with tab2:
              r = st.radio('Graph df', ('All of New York','House Price in Long Island','Local House Price'))
              
              if r == 'All of New York':
                  dv = st.selectbox('Data Visiualization',('graph','map'))
                  nyny = ny_homes_price(round(output[0],-4),ny)
                  
                  

                  if dv == 'graph':
                      
                      fig = plt.figure(figsize=(10, 4)) 
                      plt.title('Homes in the State of New York with similar price')
                      sns.histplot(data =nyny.price)
                      st.pyplot(fig)
                         
                      st.markdown('The above is a distribution of houses in Ny State. What we see is a there are hundreds of home for sale around our price point')




                      fig1 = plt.figure(figsize=(10,4))
                      sns.histplot(data = nyny.property_type)
                      st.pyplot(fig1)
                          
                      st.markdown('For people that want to have multiple properties, this graph shows the types homes on sale. At this price point most of the homes are single family homes, and very little of the others.')




                  elif dv== 'map':

                      st.pydeck_chart(
                             pdk.Deck(
                             map_style='mapbox://styles/mapbox/light-v9',
                             initial_view_state=pdk.ViewState(
                                 latitude=40.7587,
                                 longitude=-73.341426,
                                 zoom=11,
                             pitch=50, 
                         ),
                        layers=[
                             pdk.Layer(
                                'HexagonLayer',
                                data=nyny,
                                get_position='[longitude, latitude]',                    
                                radius=250,
                                elevation_scale=4,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                                auto_highlight = True,
                                coverage = 1
                             ),
                            pdk.Layer(
                                 'ScatterplotLayer',
                                 data=nyny,
                                 get_position='[longitude, latitude]',
                                 get_color='[200, 30, 0, 160]',                    
                                 get_radius=250,
                             ),
                         ],
                     ))
                      







              elif r == 'House Price in Long Island':
                  fig = plt.figure(figsize=(10, 4))
                  
                  mode_zip = list(li_homes.postcode.mode())
                  count_zip = list(li_homes.postcode.value_counts())
                  ny_mean = housey.price.mean()
                  #ny_postcode = housey.post
                  li_mean = li_homes.price.mean()
                  df = Li_homes_price(round(output[0],-4))
                  st.write('The average price seems to be:',int(round(ny_mean,-3)), ' ($) in Long Island')
                  st.write('The zip code that has the most house for sale is:', mode_zip[0], 'with a', count_zip[0], 'houses')
                  
                  with plt.style.context('Solarize_Light2'):
                      dv = st.selectbox('Data Visiualization',('histograph','map'))
                      if dv == 'histograph': 
                          plt.title('Homes in Long Island with similar price')
                          sns.histplot(data =df.price)
                          st.pyplot(fig)

                      elif dv == 'map':
                          st.pydeck_chart(pdk.Deck(
                             map_style='mapbox://styles/mapbox/light-v9',
                             initial_view_state=pdk.ViewState(
                                 latitude=40.7587,
                                 longitude=-73.341426,
                                 zoom=11,
                             pitch=50,
                         ),
                        layers=[
                             pdk.Layer(
                                'HexagonLayer',
                                data=df,
                                get_position='[longitude, latitude]',                    
                                radius=200,
                                elevation_scale=4,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                             ),
                            pdk.Layer(
                                 'ScatterplotLayer',
                                 data=df,
                                 get_position='[longitude, latitude]',
                                 get_color='[200, 30, 0, 160]',                    
                                 get_radius=200,
                             ),
                         ],
                     ))
                          


              if r == 'Local House Price':
                  dv = st.selectbox('Data Visiualization ',('violin graph', 'histograph','map'))

                  my_town = town_df(postcode)
                  my_price_town = town_price(round(output[0],-4), my_town )

                  fig = plt.figure(figsize=(10, 4))
                  x = data_ny[data_ny.postcode == postcode].city.unique()[0]
                  
                  
                  with plt.style.context('Solarize_Light2'):
                      if dv == 'histograph':
    
                          plt.title(str(x))
                          sns.histplot(data = my_price_town.price)
                          st.pyplot(fig)
                          
      
                          
                          
                      if dv == 'violin graph':
                          sns.set(font_scale = 2) # sets the font size
                          fig1 = plt.figure(figsize=(20,10)) 
                          sns.violinplot(data = my_price_town.price.values )
                          st.pyplot(fig1)
                      
                      if dv == 'map':    
                          st.pydeck_chart(pdk.Deck(
                             map_style='mapbox://styles/mapbox/light-v9',
                             initial_view_state=pdk.ViewState(
                                 latitude=40.7587,
                                 longitude=-73.341426,
                                 zoom=11,
                             pitch=50,
                         ),
                        layers=[
                             pdk.Layer(
                                'HexagonLayer',
                                data=my_town,
                                get_position='[longitude, latitude]',                    
                                radius=100,
                                elevation_scale=4,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                             ),
                            pdk.Layer(
                                 'ScatterplotLayer',
                                 data=my_town,
                                 get_position='[longitude, latitude]',
                                 get_color='[200, 30, 0, 160]',                    
                                 get_radius=100,
                             ),
                         ],
                     ))








    with tab3:
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

if __name__ == '__main__':
    main()
