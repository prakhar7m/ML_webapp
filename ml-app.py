#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install wordcloud


# In[2]:

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import regex as re
warnings.filterwarnings('ignore')#to filter all the warnings
import seaborn as sns
pd.set_option('float_format', '{:.4f}'.format)# to keep the float values short
# Import for wordcloud
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#import fot plotly
import plotly.express as px


# In[38]:



url='https://drive.google.com/file/d/1N9JfsG_NbCHb4IZxClRsTXchexe9SYiq/view?usp=share_link'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
df = pd.read_csv(url)


# In[39]:


df.head()


sns.set_style("darkgrid")


st.title("Youtube Trends Analytics")

st.sidebar.title("A")

#st.subheader("Checkbox")
w1 = st.sidebar.checkbox("show table", False)
plot= st.sidebar.checkbox("show plots", False)
_3dplot=st.sidebar.checkbox("3D plots", False)
linechart=st.sidebar.checkbox("Linechart",False)
#st.write(w1)




#st.write(df)
if w1:
    st.dataframe(df,width=2000,height=500)
if linechart:
	st.subheader("Line chart")
	st.line_chart(df)

#    plt.hist(df[sel_cols])
#    plt.xlabel(sel_cols)
#    plt.ylabel("sales")
#    plt.title(f"{sel_cols} vs Sales")
    #plt.show()	
#    st.plotly_chart(f)

if plot:
    st.subheader("correlation between sales and Ad compaigns")
    options = ("TV","radio","newspaper","sales")
    w7 = st.selectbox("Ad medium", options,1)
    st.write(w7)
    f=plt.figure()
    plt.scatter(df[w7],df["sales"])
    plt.xlabel(w7)
    plt.ylabel("sales")
    plt.title(f"{w7} vs Sales")
    #plt.show()	
    st.plotly_chart(f)



if _3dplot:
	options = st.multiselect(
     'Enter columns to plot',('TV', 'radio'),('TV', 'radio', 'newspaper', 'sales'))
	st.write('You selected:', options)
	st.subheader("TV & Radio vs Sales")
	hist_data = [df["TV"].values,df["radio"].values,df["newspaper"].values]

	#x, y, z = np.random.multivariate_normal(np.array([0, 0, 0]), np.eye(3), 400).transpose()
	trace1 = go.Scatter3d(
		x=hist_data[0],
		y=hist_data[1],
		z=df["sales"].values,
		mode="markers",
		marker=dict(
			size=8,
			#color=df['sales'],  # set color to an array/list of desired values
			colorscale="Viridis",  # choose a colorscale
	#        opacity=0.,
		),
	)

	data = [trace1]
	layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
	fig = go.Figure(data=data, layout=layout)
	st.write(fig)
