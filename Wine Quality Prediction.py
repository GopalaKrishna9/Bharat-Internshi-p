#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Prediction

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[2]:


Data = pd.read_csv("C:/Users/hp/OneDrive/Desktop/4 th trisemister/WineQT.csv")
Data.head()


# In[3]:


Data.columns


# In[4]:


Data.describe()


# In[5]:


Data.isnull().mean()


# In[6]:


Data.shape


# In[7]:


x=Data.drop(columns=['Id','quality'],axis=1)
x.head() 


# In[8]:


plt.figure(figsize=(10,10)) 
sns.heatmap(x.corr(),annot =True,cmap='RdYlBu')


# In[9]:


y = Data['density']
y.head()


# In[10]:


scaler = StandardScaler()
ann = scaler.fit_transform(x)
print(ann)


# In[11]:


train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.25)


# In[12]:


lin = LinearRegression()
lin.fit(train_x,train_y)


# In[13]:


predi = lin.predict(test_x)
ww = lin.intercept_
ww


# In[14]:


model = lin.coef_
model


# In[15]:


print(predi)


# In[16]:


plt.scatter(test_y,predi)
plt.xlabel('actual score')
plt.ylabel('predicted score')
plt.show()


# In[17]:


r2 = r2_score(test_y,predi)
r2


# In[18]:


x.head()


# In[19]:


lin.predict([[7.4,0.70,0.00,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4]])


# In[20]:


error = 0.9978 - 0.9978
error


# In[21]:


if r2 >= 0.7:
    print("The model has a strong fit and performs well.")
elif r2 >= 0.5:
    print("The model has a moderate fit and gives decent predictions.")
else:
    print("The model may need further improvement as it has a weak fit.")

print("You can further refine the model by feature engineering and hyperparameter tuning to improve its performance.")


# In the provided Jupyter Notebook code, the data is loaded, preprocessed, and split into training and testing sets. A linear regression model is created, fitted to the training data, and used to make predictions. The model's performance is evaluated using metrics.
# 
# The conclusion advises considering classification techniques if the aim is to predict wine quality as an ordinal variable since linear regression is not an ideal choice for this task. Classification models are more suitable for predicting discrete categories, such as wine quality ratings.
