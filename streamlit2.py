
import streamlit as st

import matplotlib.pyplot as plt

import pandas as pd 
import numpy as np
import seaborn as sns



st.title('Iris Dataframe & Graph')

data = pd.read_csv('https://raw.githubusercontent.com/Abzun/streamlit-example/master/Iris.csv')
box_data = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
columns = data.columns.tolist()
#st.write(columns)

#SIDE BAR STUFF HERE---------------------------------------
st.sidebar.header('Iris Side Bar')

st.sidebar.info('When not using - unselect, will lag app')
if st.sidebar.checkbox('Iris Pair Plot'):
    st.markdown('___')
    st.subheader('Pair Plot for Iris')
    fig1 = sns.pairplot(data, hue='Species')
    st.pyplot(fig1)

graph = st.sidebar.radio('Pick a Graph to see data', 
	('Heatmap', 'Area Chart', 'Just the DataFrame'
        ,'Some Stats of DataFrame', 'Box Plot','Violin Plot'), disabled = False)

if graph == 'Just the DataFrame':
    # just a dataframe showing on streamlit
    st.dataframe(data)

elif graph == 'Heatmap':
    st.markdown('___')
    # lower triangle heatmap
    st.subheader('HeatMap for iris')
    corr= data.corr()
    matrix = np.triu(corr)

    fig, ax = plt.subplots()

    sns.heatmap(data.corr(), ax=ax, mask = matrix)
    st.pyplot(fig)



elif graph == 'Some Stats of DataFrame':
    df = data.describe()

    st.write(df)

elif graph == 'Area Chart':
    st.area_chart(data.drop(columns=['Species','Id']))

elif graph == 'Box Plot':
    fig33, ax = plt.subplots()
    st.write(sns.boxplot(data = box_data, ax=ax))
    st.pyplot(fig33)
elif graph == 'Violin Plot':
    fig44, ax = plt.subplots()
    st.write(sns.violinplot(data= box_data, inner ='points'))
    st.pyplot(fig44)



#-------SIDE BAR STUFF ENDS
