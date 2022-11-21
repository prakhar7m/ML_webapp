#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install wordcloud


# In[2]:


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


df = pd.read_csv('youtube_data.csv')


# In[39]:


df.head()


# In[ ]:





# In[40]:


#dropping duplicates
df.drop_duplicates(subset=['video_id'], keep='last', inplace = True)


# In[41]:


#dropping all the numerical columns for text analysis
df.drop(columns=['channelTitle','video_id','publishedAt', 'categoryId','trending_date',
                 'view_count', 'likes', 'dislikes', 'comment_count'],axis=1,inplace=True)


# In[42]:


df.head()


# In[43]:


df.fillna(value = '', inplace = True)


# In[45]:


# Concat all text data in one column:

df['text'] = df.description + ' ' + df.tags + ' ' + df.title


# In[46]:


df.drop(columns=['description', 'tags', 'title'],axis=1,inplace=True)


# In[47]:


df.head()


# In[48]:


df['category_name'].unique()


# ### predicting categories

# In[49]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[50]:


#df_filtered = df[df.category_name.isin(['Entertainment', 'Sports', 'Music', 'Gaming', \
                                        'People & Blogs', 'Comedy','News & Politics'])]


# In[51]:


df_filtered = df
df_filtered.head()


# In[52]:


X = df_filtered.text


# In[53]:


Y = df_filtered.category_name


# In[54]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state = 0)


# In[55]:


# Applying bag of words to features in training and testing data
bag_of_words_creator = CountVectorizer()
X_train_bow = bag_of_words_creator.fit_transform(X_train)
X_test_bow = bag_of_words_creator.transform(X_test)


# In[56]:


#cl = RandomForestClassifier(random_state = 0, n_estimators=1000)
cl = RandomForestClassifier(random_state = 0)
cl.fit(X_train_bow,Y_train)


# In[57]:


y_pred = cl.predict(X_test_bow)


# In[58]:


from sklearn.metrics import confusion_matrix
import sklearn.metrics as met


# In[59]:


confusion_matrix(Y_test,y_pred)


# In[60]:


print(met.classification_report(Y_test,y_pred))


# In[61]:


df = pd.read_csv('youtube_data.csv')


# ### ML predicitng number of days for the video to stay trending
# 

# In[63]:


ML_df= df.groupby(['video_id','trending_date','publishedAt'],as_index=False).agg({'view_count':                                                             'max','likes':'max','dislikes':'max','comment_count':'max'})


# In[64]:


df1 = ML_df.copy()


# In[65]:


df1['trending_date'] = pd.to_datetime(df1['trending_date'])
df1['publishedAt'] = pd.to_datetime(df1['publishedAt'])


# In[66]:


df1['trending_day_no'] = df1.groupby(['video_id'])["trending_date"].rank('first',ascending=True)


# In[67]:


df1_count = df1.groupby('video_id',as_index=False)["trending_date"].count().rename(columns={                                                                'trending_date':'total_trending_days'})


# In[68]:


df2 = df1.merge(df1_count, left_on='video_id', right_on='video_id')
df2.head()


# In[69]:


df2.corr()


# In[70]:


df2['published_year'] = df2.publishedAt.dt.year
df2['published_month'] = df2.publishedAt.dt.month
df2['published_day'] = df2.publishedAt.dt.day
df2['published_hour'] = df2.publishedAt.dt.hour
df2['published_minute'] = df2.publishedAt.dt.minute
df2['published_week'] = df2.publishedAt.dt.week


# In[72]:


df2.drop(columns=['video_id', 'trending_date', 'publishedAt'],inplace=True) 


# In[73]:


df2.head()


# In[74]:


X = df2.drop('total_trending_days',axis=1)
Y = df2.total_trending_days


# In[75]:


import sklearn.tree


# In[76]:


trending_days_binned = pd.cut(Y, bins= [0,4,9,40])
trending_days_binned_count = trending_days_binned.value_counts().rename_axis('Binned_Trending_Days').reset_index(name = 'Count')


# In[77]:


sns.catplot( y='Count',x='Binned_Trending_Days',            data=trending_days_binned_count,kind='bar', aspect = 2)


# In[78]:


X = df2.drop('total_trending_days',axis=1)
Y = df2.total_trending_days


# In[79]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,
                            test_size=0.3,random_state=0)


# In[80]:


from sklearn import linear_model


# In[81]:


regr = linear_model.LinearRegression()


# In[82]:


regr = linear_model.LinearRegression()


# In[83]:


regr.fit(X_train,Y_train)


# In[84]:


y_pred = regr.predict(X_test)


# In[85]:


print("Coefficients: \n", regr.coef_)


# In[86]:


r2_score=sklearn.metrics.r2_score(Y_test, y_pred)


# In[87]:


print("Coefficient of determination: %.2f" % r2_score)


# In[88]:


(y_pred-Y_test).abs().mean()


# In[89]:


((y_pred-Y_test)**2).mean()


# In[100]:


from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


# In[103]:


#pip install lightgbm


# In[109]:


from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, RepeatedKFold


# In[105]:


def base_models():
  models = dict()
  models['lr'] = LinearRegression()
  models["Ridge"] = Ridge()
  models["Lasso"] = Lasso()
  models["Tree"] = DecisionTreeRegressor()
  models["Random Forest"] = RandomForestRegressor()
  models["Bagging"] = BaggingRegressor()
  models["GBM"] = GradientBoostingRegressor()
  
  return models


# In[106]:


def eval_models(model):
  cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
  scores = -cross_val_score(model, X_train, Y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, 
                            error_score='raise')
  return scores


# In[107]:


models = base_models()
# evaluate the models and store results
results, names = list(), list() 


# In[110]:


for name, model in models.items():
  scores = eval_models(model)
  results.append(scores)
  names.append(name)
  print('>%s %.3f (%.3f)' % (name, scores.mean(), scores.std()))


# In[111]:


regressmod = pd.DataFrame(np.transpose(results), columns = ["lr","Ridge","Lasso","Tree","Random Forest","Bagging","GBM"])
regressmod = pd.melt(regressmod.reset_index(), id_vars='index',value_vars=["lr","Ridge","Lasso","Tree","Random Forest","Bagging","GBM"])


# In[112]:


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

fig = px.box(regressmod, x="variable", y="value",color="variable",points='all',
labels={"variable": "Machine Learning Model",
        "value": "RMS Error"
        },title="Model Performance")
fig.show()


# In[113]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
import warnings
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def get_stacking():
	# define the base models
  level0 = list()
  level0.append(('Tree', DecisionTreeRegressor()))
  level0.append(('RF', RandomForestRegressor()))
  level0.append(('Bagging', BaggingRegressor()))
  level0.append(('GBM', GradientBoostingRegressor()))
	# define meta learner model
  level1 = LGBMRegressor()
	# define the stacking ensemble
  model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
  return model

def base_models():
  models = dict()
  models["Tree"] = DecisionTreeRegressor()
  models["Random Forest"] = RandomForestRegressor()
  models["Bagging"] = BaggingRegressor()
#   models["XGB"] = XGBRegressor()
  models["Stacked Model"] = get_stacking()
  return models

# Function to evaluate the list of models
def eval_models(model):
  cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
  scores = -cross_val_score(model, X_train, Y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, 
                            error_score='raise')
  return scores

models = base_models()
# evaluate the models and store results
results, names = list(), list() 

for name, model in models.items():
  scores = eval_models(model)
  results.append(scores)
  names.append(name)
  print('>%s %.3f (%.3f)' % (name, scores.mean(), scores.std()))


# In[114]:


import pickle

level0 = list()
level0.append(('Tree', DecisionTreeRegressor()))
level0.append(('RF', RandomForestRegressor()))
level0.append(('GBM', GradientBoostingRegressor()))
level0.append(('Bagging', BaggingRegressor()))

level1 = LGBMRegressor()
model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
model.fit(X_train, Y_train)

# Save to file in the current working directory
pkl_filename = "AssignmentPickle.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)


score = pickle_model.score(X_test, Y_test)
print("Test score: {0:.2f} %".format(100 * score))
Y_predict = pickle_model.predict(X_test)


import matplotlib.pyplot as plt
import seaborn as sns
predictions = pd.DataFrame(Y_predict, columns=['predictions'])
predictions['actual'] = Y_test
plt.scatter(x = Y_test, y = Y_predict, color='#336699',alpha=0.6)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Predicted vs Actual', fontsize=15, color='#336699',loc='center')


# In[ ]:




