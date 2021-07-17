#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


path = 'C:\\Users\\أحمد محمد\\Desktop\\New folder\\section\\train.csv'
data = pd.read_csv(path)


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.rename(mapper=lambda x : str(x).lower())


# In[6]:


data.info()


# In[7]:


round((data.isnull().sum(axis = 0 )*100)/(data.shape[0]),2)


# In[8]:


data.describe()


# In[9]:


data.head()


# In[10]:


data.Sex.value_counts()


# In[11]:


data.Age.unique()


# In[12]:


data.Cabin.unique()


# In[13]:


data.Embarked.value_counts()


# In[14]:


t = data.duplicated()


# In[15]:


data[t]    # no coumn is duplicated


# In[16]:


bins  = np.arange(data.Age.min(), data.Age.max()+5,5)
plt.hist(data = data ,x = 'Age',bins = bins)
plt.xlabel('age')
plt.ylabel('count')
plt.title('histogram of Age')
("")


# In[17]:


data_copy = data.copy()


# In[21]:


data.isnull().sum()


# In[22]:


data= data.replace({'Age':{np.nan: data.Age.mean()},'Embarked':{np.nan:'S'}})


# In[23]:


round((data.isnull().sum(axis = 0 )*100)/(data.shape[0]),2)


# In[24]:


data.reset_index(drop = True).head()


# In[25]:


data = data.drop(columns =['PassengerId','Cabin','Ticket','Name'])


# In[26]:


data.head()


# In[27]:


data = pd.get_dummies(data = data , columns = ['Sex' , 'Embarked'])


# In[28]:


data = data.drop(columns=['Survived']).assign(Survived= data['Survived'])


# In[29]:


data.head()


# In[30]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split( data.iloc[:,:-1],data.iloc[:,-1],test_size=0.3, shuffle=True,
                                         random_state=23,stratify=data.iloc[:,-1])  
 
x_val, x_test, y_val, y_test = train_test_split(x_test,y_test ,test_size=0.5, shuffle=True, 
                                         random_state=23,stratify=y_test)


# In[31]:


y_train


# In[32]:


from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()


# In[33]:


from sklearn.model_selection import GridSearchCV


# In[34]:


parameters = {'max_iter':[1000,3000,4000], 'C':[0.001,0.01,0.1,1,10,100]}
clf = GridSearchCV(logistic , parameters , scoring = 'recall')


# In[35]:


clf.fit(x_train , y_train)


# In[36]:


clf.cv_results_


# In[37]:


best_logistic = clf.best_estimator_


# In[38]:


best_logistic


# In[39]:


clf.best_score_


# In[40]:


y_predict = clf.predict(x_test) 


# In[41]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_predict)


# In[42]:


from sklearn.metrics import recall_score
recall_score(y_test, y_predict)


# In[43]:


clf.predict_proba(x_test)


# In[44]:


from sklearn.tree import DecisionTreeClassifier
clf_2= DecisionTreeClassifier(max_depth = 15)


# In[45]:


clf_2.fit(x_train , y_train)


# In[46]:


y2_predict= clf_2.predict(x_test)


# In[47]:


accuracy_score(y_test , y2_predict)


# In[49]:


from sklearn.ensemble import RandomForestClassifier
clf_3 = RandomForestClassifier(max_depth=6, random_state=0)


# In[50]:


clf_3.fit(x_train, y_train )


# In[51]:


accuracy_score(y_test , clf_3.predict (x_test))

