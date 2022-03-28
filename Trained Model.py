#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# In[2]:


data = pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-2/master/CaseStudy/Advertising.csv', index_col=0)
data.head(100)


# In[3]:


features = ['TV', 'radio', 'newspaper']                # create a Python list of feature names
target = ['sales']   


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=0)


# In[5]:


#Instantiating the model
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression(fit_intercept=True)


# In[6]:


get_ipython().run_line_magic('time', '')
lr_model.fit(X_train, y_train)


# In[7]:


# Saving model to disk
pickle.dump (LinearRegression, open('model.pkl','wb'))


# In[ ]:




