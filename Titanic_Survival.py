#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


titan=pd.read_csv(r'C:\Users\Vignesh Chowdary\Downloads\titanic.csv')


# In[5]:


print(titan.head(10))


# In[6]:


titan['pclass'].value_counts()


# In[7]:


titan.dtypes


# In[8]:


titan.isnull().sum()


# In[9]:


titan_new=titan.drop(['name','cabin','ticket','boat','body','home.dest'],axis=1)


# In[11]:


titan_new.head()


# In[12]:


titan_new.isnull().sum()


# In[13]:


titan_new['age'] =titan_new['age'].fillna(titan_new['age'].mean())


# In[14]:


titan_new.isnull().sum()


# In[15]:


titan_new=titan_new.dropna()


# In[16]:


titan_new.isnull().sum()


# In[17]:


titan_new.head()


# In[18]:


from sklearn.preprocessing import LabelEncoder


# In[19]:


titan_new.dtypes


# In[21]:


le=LabelEncoder()
titan_new['sex'] =le.fit_transform(titan_new['sex'])
titan_new['embarked'] =le.fit_transform(titan_new['embarked'])


# In[22]:


titan_new.dtypes


# In[23]:


titan_new.head()


# In[24]:


x=titan_new[['pclass','sex','age','sibsp','parch','fare','embarked']]
y=titan_new['survived']


# In[25]:


x.head()


# In[27]:


from sklearn.preprocessing import MinMaxScaler


# In[28]:


mms=MinMaxScaler()
x_scaled=mms.fit_transform(x)


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=4)


# In[32]:


from sklearn.linear_model import LogisticRegression


# In[2]:


model=LogisticRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)


# In[34]:


pred


# In[35]:


result=x_test


# In[36]:


result['Actual'] =y_test
result['Predicted'] =pred


# In[37]:


result.head(10)


# In[40]:


survive=model.predict([[1.0,0,16.000000,0.0,1.0,39.4000,2]])
if survive==0:print('I am Sorry, The Person is Not Survived')
else:
 print('Good News, The Person is Survived')


# In[ ]:





# In[ ]:




