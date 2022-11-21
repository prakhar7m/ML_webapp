#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install wordcloud


# In[2]:

import streamlit as st
import pandas as pd


# In[38]:



url_data = (r'https://raw.githubusercontent.com/prakhar7m/ML_webapp/main/youtube_data.csv')
df= pd.read_csv(url_data)
# In[39]:
df.head()
sns.set_style("darkgrid")
st.title("Youtube Trends Analytics")


