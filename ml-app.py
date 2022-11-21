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



url_data = (r'https://raw.githubusercontent.com/prakhar7m/ML_webapp/main/youtube_data.csv')
df= pd.read_csv(url_data)
# In[39]:
df.head()
sns.set_style("darkgrid")
st.title("Youtube Trends Analytics")


