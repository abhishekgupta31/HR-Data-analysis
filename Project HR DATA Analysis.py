#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('HR.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df.isnull().values.any()


# In[7]:


df['Attrition'].value_counts()


# In[8]:


sns.countplot(df['Attrition'])


# In[9]:


plt.subplots(figsize=(10,10))
sns.countplot(x='Age',hue='Attrition',data=df)


# In[10]:


corr= df.corr()
corr


# In[11]:


for columns in df.columns:
    if df[columns].dtype == 'object':
        print(str(columns) +"  "+ str(df[columns].unique()))
        print(df[columns].value_counts())


# In[12]:


df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis=1)


# In[13]:



sns.heatmap(df.isnull(),linecolor='white')


# In[14]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)


# In[15]:


from sklearn.preprocessing import LabelEncoder


# In[16]:


for columns in df.columns:
    if df[columns].dtype==np.number:
        continue
    df[columns]=LabelEncoder().fit_transform(df[columns])


# In[17]:


df['new_age']=df['Age']
df=df.drop('Age',axis=1)
df


# In[19]:


x = df.drop(['Attrition'],axis=1)
y = df['Attrition']


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=101)


# In[23]:


from sklearn.ensemble import RandomForestClassifier


# In[25]:


rc = RandomForestClassifier(n_estimators=101,criterion='entropy',random_state=101)


# In[26]:


rc.fit(x_train,y_train)


# In[28]:


pred = rc.predict(x_test)
pred


# In[29]:


rc.score(x_train,y_train)


# In[30]:


from sklearn.metrics import accuracy_score


# In[32]:


accuracy_score(y_test,pred)


# In[33]:


from sklearn.metrics import confusion_matrix


# In[36]:


cm =confusion_matrix(y_test,pred)


# In[37]:


tn = cm[0][0]
tp = cm[1][1]
fn = cm[1][0]
fp=  cm[0][1]


# In[38]:


print("Model Accuracy:{} ".format((tp+tn)/(tp+tn+fn+fp)) , )


# In[ ]:




