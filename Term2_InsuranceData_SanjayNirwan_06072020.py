#!/usr/bin/env python
# coding: utf-8

#  
# 
# # Project on  Insurance Data Set
# 
# Submitted By: Mr Sanjay Singh Nirwan
# 

# ## Table of Contents 
# 
# 1. Problem Statement
# 2. Data Loading and Description
# 3. Data Cleaning
# 4. Exploratory Data Analysis
# 5. Model Selection
# 6. Model Evaluation
# 8. Final Model and Recommedations

# # 1. Problem Statement
# 
# #### Health insurance is an insurance product which covers medical and surgical expenses of an insured individual. It reimburses the expenses incurred due to illness or injury or pays the care provider of the insured individual directly.
# 
# Problem Statement: we will from the Data Given in the project try to analyse the Health Risk score of the People applying for the insurance and hence be able to help the insuring party take quicker decisions on the Health Risk Score of the applting individul and decide to allocate the insurance to the applier by a informed decision from the Past data
# 

# # 2. Data Loading and Description
# 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn import metrics

# allow plots to appear directly in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output


# In[2]:


data = pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/insurance_data.csv', index_col=0)
data.head()


# In[3]:


pd.set_option('display.max_columns', None)


# In[4]:


data.shape


# The Data provided has 127 columns and 59381 Rows

# In[5]:


data.sample(10)


# The Sample data shows that there are mostly integer data , but also float data , let us analyse them below

# In[6]:


data.info()


# In[7]:


data.describe()


# # Deeper Analaysis on Categorical and Continous data
# 
# Feature details posted in data overview section - The following variables are all categorical (nominal):
# Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41
# 
# The following variables are continuous:
# Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5
# 
# The following variables are discrete:
# Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32
# 
# Medical_Keyword_1-48 are dummy variables.

# # 3. Data Cleaning
# 

# We will check for missing values .
# 
# If a categorical feature has missing values - if required will impute it with median
# 
# if a continous feature has missing values - if required will impute it with mean

# In[8]:


data.isnull().sum()[data.isnull().sum() !=0]


# Lets draw a bar graph to visualize percentage of missing features in train set

# In[9]:


data_missing= data.isnull().sum()[data.isnull().sum() !=0]
data_missing=pd.DataFrame(data_missing.reset_index())
data_missing.rename(columns={'index':'features',0:'missing_count'},inplace=True)
data_missing['missing_count_percentage']=((data_missing['missing_count'])/59381)*100
plt.figure(figsize=(20,8))
sns.barplot(y=data_missing['features'],x=data_missing['missing_count_percentage'])
data_missing


# Medical_Hist_32/24/15/10 , Family_hist_5 are top five features with huge amount of missing data ( imputaion to these might not be fruitful - hence we are going to drop these features)

# In[10]:


data.dtypes.unique()


# In[11]:


data['Product_Info_2'] = data['Product_Info_2'].astype('category').cat.codes


# In[12]:


Product_Info_2_dummies = pd.get_dummies(data.Product_Info_2)
data = data.drop('Product_Info_2',axis = 1)
data = data.join(Product_Info_2_dummies)


# In[13]:


aixs1 = plt.subplots(1,1,figsize=(10,5))
sns.countplot(x='Response',data=data)


# Employment_Info_1_4_6 Insurance_History_5 Family_Hist_2-3-4-5 are continous features .
# 
# The following variables are discrete: Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32
# 
# remove rows with missing values and see model performance
# impute missing values with mean and median or may be mode.
# 

# In[14]:


data = data.drop(['Medical_History_10','Medical_History_24','Medical_History_32'], axis=1) 


# In[15]:


plt.plot(figsize=(15,10))
sns.boxplot(data['Employment_Info_1'])


# Employment_Info_1 seems to have lots of outliers - Median should be right to impute missing value

# In[16]:


data['Employment_Info_1'].fillna(data['Employment_Info_1'].median(),inplace=True)


# In[17]:


data['Employment_Info_4'].fillna(data['Employment_Info_4'].median(),inplace=True)


# In[18]:


data_missing= data.isnull().sum()[data.isnull().sum() !=0]
data_missing=pd.DataFrame(data_missing.reset_index())
data_missing.rename(columns={'index':'features',0:'missing_count'},inplace=True)
data_missing['missing_count_percentage']=((data_missing['missing_count'])/59381)*100
data_missing


# In[19]:


Continuos = ['Employment_Info_6', 'Insurance_History_5','Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']
data[Continuos] = data[Continuos].fillna(data[Continuos].mean())


# In[20]:


Categorical = ['Medical_History_1', 'Medical_History_15']
data[Categorical] = data[Categorical].apply(lambda x:x.fillna(x.value_counts().index[0]))


# In[21]:


data_missing= data.isnull().sum()[data.isnull().sum() !=0]
data_missing=pd.DataFrame(data_missing.reset_index())
data_missing.rename(columns={'index':'features',0:'missing_count'},inplace=True)
data_missing['missing_count_percentage']=((data_missing['missing_count'])/59381)*100
data_missing


# Now since no data is missing and cleaning has been done, lets proceed to EDA...
# Please Note that I am not doing pandas Profiling as the dataset is huge and system gets hanged while running profiling on such huge dataset.

# # 4. EDA

# Let us do EDA on the data Now

# In[22]:


data.describe()


# In[23]:


data.isnull().sum()[data.isnull().sum() !=0]


# In[27]:


pd.set_option('display.max_columns', None)
data.corr()


# From the above we see that there is huge data that has low corellation and hence are removing such data
# Dropping data that has low correlation

# In[65]:


data.columns


# In[29]:


fig_size = (20, 20)
fig, ax = plt.subplots(figsize=fig_size)
sns.heatmap(data.corr(), cmap="Blues")


# In[30]:


X = data.drop(['Response'], axis=1)
y = data.Response


# In[31]:


print(type(X))
print(X.shape)


# In[32]:


y = data.Response
y.head()


# In[33]:


print(type(y))
print(y.shape)


# # Data Modelling

# In[34]:


from sklearn.model_selection import train_test_split

def split(X,y):
    return train_test_split(X, y, test_size=0.20, random_state=17)


# In[35]:


X_train, X_test, y_train, y_test=split(X,y)
print('Train cases as below')
print('X_train shape: ',X_train.shape)
print('y_train shape: ',y_train.shape)
print('\nTest cases as below')
print('X_test shape: ',X_test.shape)
print('y_test shape: ',y_test.shape)


# In[36]:


y_train.head()


# # Linear Regression

# In[37]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train) 


# In[38]:


print('Intercept:',linreg.intercept_)          # print the intercept 
print('Coefficients:',linreg.coef_)


# In[39]:


y_pred_train = linreg.predict(X_train)  


# In[40]:


y_pred_test = linreg.predict(X_test)


# In[41]:


MAE_train = metrics.mean_absolute_error(y_train, y_pred_train)
MAE_test = metrics.mean_absolute_error(y_test, y_pred_test)


# In[42]:


print('MAE for training set is {}'.format(MAE_train))
print('MAE for test set is {}'.format(MAE_test))


# In[43]:


MSE_train = metrics.mean_squared_error(y_train, y_pred_train)
MSE_test = metrics.mean_squared_error(y_test, y_pred_test)


# In[44]:


print('MSE for training set is {}'.format(MSE_train))
print('MSE for test set is {}'.format(MSE_test))


# In[45]:


RMSE_train = np.sqrt( metrics.mean_squared_error(y_train, y_pred_train))
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))


# In[46]:


print('RMSE for training set is {}'.format(RMSE_train))
print('RMSE for test set is {}'.format(RMSE_test))


# In[47]:


yhat = linreg.predict(X_train)
SS_Residual = sum((y_train-yhat)**2)
SS_Total = sum((y_train-np.mean(y_train))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
print(r_squared, adjusted_r_squared)


# In[48]:


yhat = linreg.predict(X_test)
SS_Residual = sum((y_test-yhat)**2)
SS_Total = sum((y_test-np.mean(y_test))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print(r_squared, adjusted_r_squared)


# # Logistic Regression

# In[49]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# In[50]:


y_pred_train = logreg.predict(X_train)  


# In[51]:


print(y_pred_train)


# In[52]:


y_pred_test = logreg.predict(X_test) 


# In[53]:


print(y_pred_test)


# In[54]:


from sklearn.metrics import accuracy_score
print('Accuracy score for test data using Logistic Regression is:', accuracy_score(y_test,y_pred_test))


# # Decision Tree

# In[55]:


from sklearn import tree
model = tree.DecisionTreeClassifier(random_state = 0)
model.fit(X_train, y_train)


# In[56]:


y_pred_traindt = model.predict(X_train)  


# In[57]:


y_pred_testdt = model.predict(X_test) 


# In[58]:


from sklearn.metrics import accuracy_score
print('Accuracy score for test data using Decision Tree is:', accuracy_score(y_test,y_pred_testdt))


# # Random Forest

# In[59]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state = 0)

model.fit(X_train, y_train)


# In[60]:


model1 = RandomForestClassifier(random_state = 4,
                                n_estimators = 100,
                                min_samples_split = 2,
                                n_jobs=4,
                                min_samples_leaf = 1)

model1.fit(X_train, y_train)


# In[61]:


y_pred_trainRF = model.predict(X_train)


# In[62]:


y_pred_testRF = model.predict(X_test) 


# In[63]:


from sklearn.metrics import accuracy_score
print('Accuracy score for test data using the Random Forest:', accuracy_score(y_test,y_pred_testRF))


# # Model Selection

# In[64]:


print('Accuracy score for test data using Logistic Regression is:', accuracy_score(y_test,y_pred_test))
print('Accuracy score for test data using Decision Tree is:', accuracy_score(y_test,y_pred_testdt))
print('Accuracy score for test data using the Random Forest:', accuracy_score(y_test,y_pred_testRF))


# Thus from the above models we are choosing Random forest as the acuuracy is .56 and is the best amongst the four models used.
# howvever we can use further models and check for the same data how good the accuracy score is.
