#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the dependancies
import numpy as np                # numpy is use to create numpy array
import pandas as pd               # pandas is use to create dataframe
import matplotlib.pyplot as plt   # matplotlib and seaborn those are data vistulization library 
import seaborn as sns

from sklearn.model_selection import train_test_split  # sklearn most important library contain sevral pre-processing. 
                                                      # function and machine learning algorithm.
                                                      #  here we are importing.train_test function '''
        
# train_test_split function is a powerful tool in scikit-learn's arsenal, primarily used 
# to divide dataset into traning and testing subdata

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ***DATA COLLECTION AND PROCESSING***

# In[2]:


# Load data from csv file to pandas dataframe
titanic_data  = pd.read_csv(r'C:\Users\yetes\OneDrive\Desktop\Excel\Titanic-Dataset.csv')   # read_csv is function use to read .csv file and load it in pandas dataframe


# In[3]:


# Printing the 1st file 5 rows of the dataframe
titanic_data.head()                                                                                   # NaN = not a number


# In[4]:


# Check the no. of rows and columns
titanic_data.shape                                # shape is pyhton function used to find the dimensions of the data structure, 
                                                  # such as NumPy array and Pandas Dataframe


# In[5]:


# getting some information about the data
titanic_data.info()                               # info() is a python function use to print the information of the dataframe


# In[6]:


# Cheak the number of mising values of in each colume
titanic_data.isnull().sum()                              # isnull() function is use to find the null values form each columes
                                                         # and sum calculate the sum of these


# ***HANDLING THE MISSING VALUES***

# In[7]:


# drop the 'Cabin column' from the dataframe
titanic_data = titanic_data.drop(columns='Cabin',axis=1)                   #1. rows = 0 and columes = 1
                                                                           #2. drop() method removes the specified row or column. 


# In[8]:


# replacing the missing values in 'Age' column with main value  
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace = True)     #1. here we are filling all the missing value with mean values
                                                                           #2. na = not available
                                                                           #3. mean() function is used to calculate the mean/average of the input values or input dataset
                                                                           #4. fillna() is a pandas function to fill the NA/NaN values with the specified method.


# In[9]:


# finding the mode values from 'embarked' columns         
print(titanic_data['Embarked'].mode())         # Mode is the most common number or character that appears in your set of data


# In[10]:


print(titanic_data['Embarked'].mode()[0]) 


# In[11]:


# Replacing the missing values in 'Embarked' column with mode value+
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace = True)     # [0] for 0 index


# In[12]:


# Cheak the number of mising values of in each colume
titanic_data.isnull().sum()  


# ***DATA ANALYSIS***

# In[13]:


# getting some statistical measures about the data
titanic_data.describe()                               # describe() method returns the description of the data in the Dataframe


# In[14]:


# finding the number of people survived and not survived
titanic_data['Survived'].value_counts()    # value_counts() function returns the count of the all uqunic values in the 
                                           # given index in descending order, without any null value
                                         


# ***DATA VISTUALIZATION***

# In[15]:


sns.set()                                                                # set() function creates a set object


# In[16]:


# making a count plot for 'survived' column
sns.countplot(x='Survived', data=titanic_data)         # countplot() is represent the occurrence of the
                                                       # obervation present in the categorical variable


# In[17]:


titanic_data['Sex'].value_counts() 


# In[18]:


# making a count plot for 'sex' column
sns.countplot(x='Sex', data=titanic_data)  


# In[1]:


# number of survivors gender wise
sns.countplot(x ='Sex', hue='Survived', data=titanic_data)  #1.'hue' is used to vistualize the data of 
                                                            # different categories in the plot
                                                            #2.'palette' is used to change the color of the plot  
      


# In[20]:


# making a count plot for 'Pclass' column
sns.countplot(x='Pclass', data=titanic_data)  


# In[21]:


# number of survivors based on the 'Pclass' column
sns.countplot(x ='Pclass', hue='Survived', data=titanic_data) 


# In[22]:


# Total number of people from perticular country or state
titanic_data['Embarked'].value_counts()


# In[23]:


# number of survivors based on the 'Embarked' column
sns.countplot(x ='Embarked', hue='Survived', data=titanic_data) 


# ***ENCODING THE CATEGORICAL COLUMN***

# In[24]:


titanic_data['Sex'].value_counts() 


# In[25]:


titanic_data['Embarked'].value_counts() 


# In[26]:


# coverting categorical column
# Replace the char into int for butter understanding to machine
titanic_data.replace({'Sex':{'male':0, 'female':1}, 'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)


# In[27]:


titanic_data.head()


# ***SEPARATING FEATURES(unwanted data) AND TARGET(important data)***

# In[28]:


X = titanic_data.drop(columns=['PassengerId','Name','Ticket','Survived'], axis=1)  # whenever we removing or droping column = axis =1
Y = titanic_data['Survived']                                                      # whenever we removing or droping rows = axis =0


# In[29]:


print(X)


# In[30]:


print(Y)


# ***SPLITTING THE DATA INTO TRANING DATA AND TEST DATA***

# In[31]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=2)
#1. X_train,X_test,Y_train,Y_test are 4 array.
#2. X_train,X_test from x   and  Y_train,Y_test from y
#3. X_train == Y_train   and   X_test == Y_test
#4. 2.0 represents 20%. means 20% data as test_data
#5. X_train = 80% data and X_test = 20% data
#6. In machine learning we take train = 80% or 90% data    and      test = 20% data or 10%
#7. 'random_state=2' define the data split in 2 ways otherwise it split in different ways.

# train_test_split function is a powerful tool in scikit-learn's arsenal, primarily used 
# to divide dataset into traning and testing subdata and it is a part of the sklearn.model_selection module


# In[32]:


print(X.shape, X_train.shape, X_test.shape)


# ***MODEL TRANING***
# ***( Logistic Regression model )***

# In[33]:


# Line equation: y = mx + c
# Logistic Regression work on Sigmoid function: z = w.X + b
# X - input features(  Age, SibSp, Parch, Fare, Embarked) , Y - prediction Probability(0 or 1) , w - weights , b - biases


# In[34]:


model = LogisticRegression()


# In[35]:


# traning the logistic regression models with traning data
model.fit(X_train, Y_train)


# ***MODEL EVALUATION***
# ***Accuracy Score***

# In[36]:


# we are evaluating model test data
# accuracy on traning data
X_train_prediction = model.predict(X_train)


# In[37]:


print(X_train_prediction)
# 0 = person didn't survived
# 1 = person survived


# In[38]:


traning_data_accuracy = accuracy_score(Y_train, X_train_prediction )
print("Accuracy score of traning data: ",traning_data_accuracy)


# In[39]:


# accuracy on test data
X_test_prediction = model.predict(X_test)


# In[40]:


print(X_test_prediction)


# In[41]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction )
print("Accuracy score of test data: ",test_data_accuracy)


# ***NOTE:***
# ***->if the Accuracy score of traning data is very different than Accuracy score of test data, than that model is overfitted or underfitted***

# ***high test_data_accuracy and less training_data_accuracy than, model id ' underfitted ' and vice versa*** 
