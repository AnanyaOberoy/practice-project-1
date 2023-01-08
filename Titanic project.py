#!/usr/bin/env python
# coding: utf-8

# Titanic Project

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[2]:


df=pd.read_csv('titanic.csv')


# In[ ]:


df


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df['Survived'].value_counts()


# In[6]:


df.isnull().sum()


# In[7]:


df.isnull().sum()


# In[8]:


imp=SimpleImputer(missing_values=np.nan,strategy='mean')


# In[9]:


df['Age']=imp.fit_transform(df['Age'].values.reshape(-1,1))


# In[10]:


imp=SimpleImputer(missing_values=np.nan,strategy='most_frequent')


# In[11]:


df['Embarked']=imp.fit_transform(df['Embarked'].values.reshape(-1,1))


# In[12]:


df['Cabin']=imp.fit_transform(df['Cabin'].values.reshape(-1,1))


# In[13]:


df.isnull().sum()


# In[14]:


from sklearn.preprocessing import LabelEncoder


# In[15]:


le=LabelEncoder()


# In[16]:


for col in df.columns:
    df[col]=le.fit_transform(df[col])


# In[17]:


df


# In[18]:


plt.boxplot(df['Pclass'])
plt.show()


# In[19]:


plt.boxplot(df['Sex'])
plt.show()


# In[20]:


df.plot(kind='box',subplots=True,layout=(3,4))


# In[21]:


from scipy.stats import zscore
z=np.abs(zscore(df))
z


# In[22]:


threshold=3
print(np.where(z>3))


# In[23]:


df_new=df[(z<3).all(axis=1)]


# In[24]:


df_new.shape,df.shape


# In[25]:


df=df_new


# In[26]:


df.skew()


# In[27]:


sns.histplot(df['PassengerId'])


# In[28]:


sns.histplot(df['Sex'])


# In[29]:


sns.histplot(df['Pclass'])


# In[30]:


sns.histplot(df['Fare'])


# In[31]:


sns.pairplot(df)


# In[32]:


df


# In[33]:


from sklearn.preprocessing import PowerTransformer


# In[34]:


import warnings
warnings.filterwarnings('ignore')


# In[35]:


features=['Age','Fare']


# In[36]:


pt=PowerTransformer(method='yeo-johnson')


# In[37]:


df[features]=pt.fit_transform(df[features].values)


# In[38]:


df.skew()


# In[39]:


corr=df.corr()


# In[40]:


corr


# In[41]:


plt.figure(figsize=(15,12))
sns.heatmap(corr,annot=True)


# In[42]:


x=df.drop('Survived',axis=1)
y=df['Survived']


# In[43]:


import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[44]:


import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[45]:


calc_vif(x)


# In[ ]:


x.drop('Name',axis=1,inplace=True)


# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)


# In[48]:


x_train.shape,x_test.shape


# In[49]:


df['Survived'].value_counts()


# In[50]:


from sklearn.linear_model import LogisticRegression


# In[51]:


lr=LogisticRegression()


# In[52]:


lr.fit(x_train,y_train)


# In[53]:


pred=lr.predict(x_test)


# In[54]:


print('accuracy_score',accuracy_score(y_test,pred))
print('confusion_matrix',confusion_matrix(y_test,pred))
print('classification_report',classification_report(y_test,pred))


# In[55]:


from sklearn.model_selection import cross_val_score


# In[56]:


score=cross_val_score(lr,x,y,cv=10)


# In[57]:


print(score)
print(score.mean())
print(score.std())


# In[58]:


from sklearn.tree import DecisionTreeClassifier


# In[59]:


dtc=DecisionTreeClassifier()


# In[60]:


dtc.fit(x_train,y_train)


# In[61]:


pred=dtc.predict(x_test)


# In[62]:


print('accuracy_score',accuracy_score(pred,y_test))
print('confusion_matrix',confusion_matrix(pred,y_test))
print('classification_report',classification_report(pred,y_test))


# In[63]:


score=cross_val_score(dtc,x,y,cv=10)
print(score)
print(score.mean())
print(score.std())


# In[64]:


from sklearn.svm import SVC


# In[65]:


svc=SVC()
svc.fit(x_train,y_train)


# In[66]:


pred=svc.predict(x_test)


# In[67]:


print('accuracy_score',accuracy_score(y_test,pred))
print('confusion matrix',confusion_matrix(y_test,pred))
print('classification report',classification_report(y_test,pred))


# In[68]:


score=cross_val_score(svc,x,y,cv=10)
print(score)
print(score.mean())
print(score.std())


# In[69]:


from sklearn.ensemble import AdaBoostClassifier


# In[70]:


ad=AdaBoostClassifier()


# In[71]:


ad.fit(x_train,y_train)


# In[72]:


pred=ad.predict(x_test)


# In[73]:


print('accuracy_scoe',accuracy_score(pred,y_test))
print('confusion_matrix',confusion_matrix(pred,y_test))
print('classification_report',classification_report(y_test,pred))


# In[74]:


score=cross_val_score(ad,x,y,cv=10)
print(score)
print(score.mean())
print(score.std())


# In[75]:


from sklearn.ensemble import RandomForestClassifier


# In[76]:


rf=RandomForestClassifier()


# In[77]:


rf.fit(x_train,y_train)


# In[78]:


pred=rf.predict(x_test)


# In[79]:


print('accuracy_score',accuracy_score(y_test,pred))
print('confusion_matrix',confusion_matrix(y_test,pred))
print('classification_report',classification_report(y_test,pred))


# In[80]:


score=cross_val_score(rf,x,y,cv=100)
print(score)
print(score.mean())
print(score.std())


# In[81]:


from sklearn.model_selection import GridSearchCV


# In[82]:


model=RandomForestClassifier()


# In[83]:


parameter={'criterion':['gini','entropy'],'min_samples_split':[1,2,3,4,5],'min_samples_leaf':[1,2,3,4,5],'max_features':['auto', 'sqrt', 'log2']}


# In[84]:


grid=GridSearchCV(estimator=model,param_grid=parameter,cv=5)


# In[85]:


grid.fit(x_train,y_train)


# In[86]:


print(grid)
print(grid.best_score_)
print(grid.best_estimator_.criterion)
print(grid.best_params_)


# In[87]:


import joblib


# In[88]:


joblib.dump(RandomForestClassifier,'titanic_tragedy.obj')


# In[ ]:





# In[ ]:




