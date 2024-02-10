#!/usr/bin/env python
# coding: utf-8

# ***IMPORTING THE DEPENDENCIES***

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[4]:


# loading the dataset to a pandas dataframe
credit_card_data = pd.read_csv(r'C:\Users\yetes\OneDrive\Desktop\Excel\creditcard.csv')


# In[6]:


# first 5 rows of the dataset
credit_card_data.head()

# Int the class column 
# 0 = legit transaction
# 1 = fraud transaction


# In[9]:


# Last 5 rows of the dataset
credit_card_data.tail()                                      # tail() function is use to print the last 5 rows of the dataset


# In[10]:


# dataset information
credit_card_data.info()


# In[11]:


# checking the number of missing of each column
credit_card_data.isnull().sum()


# In[12]:


# distribution of legit transaction and fraud transaction
credit_card_data['Class'].value_counts()


# ***THIS DATASET IS HIGHLY UNBALANCED***

# In[14]:


# separating the data for the analysis                 # class column     
legit = credit_card_data[credit_card_data.Class == 0]  # if the class value is 0, then the entire row store in the legit variable
fraud = credit_card_data[credit_card_data.Class == 1]  # if the class value is 1, then the entire row store in the fraud variable


# In[17]:


print("legit data dimensions: ",legit.shape)
print("fraud data dimensions: ",fraud.shape)


# In[18]:


# statistical measures of the data for legit
legit.Amount.describe()


# In[19]:


# statistical measures of the data for fraud
fraud.Amount.describe()


# In[20]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()         # groupby() function involves some combinations of splitting the object, 
                                                 # applying a function, and combining the results


# ***UNDER-SAMPLING***

# In[21]:


# build a sample dataset containing similar distribution od normal transaction and fraudulent transactions
# the number of fraudulant transaction --> 492


# In[22]:


# here we are doing random sample..
legit_sample = legit.sample(492)             # sample() function is use to select random values from legit and 
                                             # 492 is number of sample we want


# ***CONCATENATING TWO DATAFRAME***

# In[24]:


# legit_sample, fraud are 2 dataframe
new_dataset = pd.concat([legit_sample, fraud], axis = 0)   # concat() function is use to concat the two dataframe
                                                         # if axis=0 it will adding two dataframe one-by-one


# In[25]:


new_dataset.head()


# In[27]:


new_dataset.tail()


# In[28]:


# after concatenating the 2 dataframe we counting the transaction from each group
new_dataset['Class'].value_counts()


# In[29]:


new_dataset.groupby('Class').mean()         # groupby() function involves some combinations of splitting the object, 


# ***SPLITTING THE DATA INTO FEATURE AND TARGETS***

# In[30]:


X = new_dataset.drop(columns = 'Class', axis=1)   # here we have drop the class column
Y = new_dataset['Class']                          # here we are print class column( 1st = 5 rows & last = 5 rows )


# In[31]:


print(X)


# In[32]:


print(Y)


# ***SPLITTING THE DATA INTO TRAINING DATA AND TESTING DATA***

# In[47]:


X_train ,X_test ,Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2 ) #1. test_size amount of testing data you want
                                                                                                      #2. traning data = 80% or 90% and testing data = 10% or 20%
                                                                                                      #3. features(input) are present in X and labals(output) are present int Y
                                                                                                      #4. in train_test_split() function, the stratify parameter splits the
# lebals are 0 and 1                                                                                  # dataset so that the proportion of the values in the sample matches the proportion provide with the parameter


# In[48]:


print(X.shape, X_train.shape, X_test.shape)


# ***MODEL TRANING ( Logistic Regress model )***

# In[51]:


model = LogisticRegression()


# In[52]:


# traing the logistic regression model with traning data
model.fit(X_train, Y_train)

# NOTE:
# model.fit(X_train, Y_test) and () is missing in model = LogisticRegression() error --> AttributeError: 'DataFrame' object has no attribute '_validate_params'


# ***EVALUTION OF THE MODEL BASED ON ACCURACY SCORE***

# In[57]:


# accuracy on training data
X_train_prediction = model.predict(X_train)                          # 'predict' is the attribute of LogisticRegresssion model
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) # it compare the values and give the accuracy score


# In[59]:


print("Accuracy score of traning data: ",training_data_accuracy)
# training data accuracy score : 94%
# generally the accuracy score more than 75 and 85 percentages is good


# In[61]:


# accuracy on testing data
# Y_test is real lebal
X_test_prediction = model.predict(X_test)                          # 'predict' is the attribute of LogisticRegresssion model
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[62]:


print("Accuracy score of test data: ",test_data_accuracy)
# test data accuracy score : 94%


# ***NOTE:***
# ***->if the Accuracy score of traning data is very different than Accuracy score of test data, than that model is overfitted or underfitted***

# ***high test_data_accuracy and less training_data_accuracy than, model id ' underfitted ' and vice versa*** 

# In[ ]:




