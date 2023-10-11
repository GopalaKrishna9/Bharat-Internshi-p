#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[2]:


Data = pd.read_csv("C:/Users/hp/OneDrive/Desktop/4 th trisemister/Iris.csv")
Data


# In[3]:


Data.columns


# In[4]:


#Determining the missing the values
Data.isnull().sum()


# In[5]:


#Data Statistics
def Stat(c,d):
    print(c)
    print(d)
Stat(Data.describe(),Data.info())  
Stat(Data.mean(),Data.median())
Stat(Data.count(),Data.std())
Stat(Data.max(),Data.min()) 


# In[6]:


x= Data.drop(['Id','Species'],axis =1)
x.head()


# In[7]:


y = Data["Species"]
y.head()


# In[8]:


Data.plot(kind = "scatter" , x='SepalLengthCm',y='SepalWidthCm')
plt.show()


# In[9]:


x.corr()


# In[10]:


plt.figure(figsize=(10,10)) 
sns.heatmap(x.corr(),annot =True,cmap='RdYlBu')


# In[11]:


stand = StandardScaler()
X = stand.fit_transform(x)
print(X)


# In[12]:


vg = LabelEncoder()
Y = vg.fit_transform(y)
print(Y)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)


# In[17]:


y_pred = knn.predict(X_test)


# In[18]:


accuracy = accuracy_score(y_test,y_pred)
print(accuracy)


# This code will load the Iris dataset, split it into training and testing sets, create a K-Nearest Neighbors classifier, make predictions, and calculate the accuracy. The accuracy will give you an idea of how well the model can predict the species of flowers based on the length of their petals and sepals.

# In[ ]:




