#!/usr/bin/env python
# coding: utf-8

# ***IMPORTING IMPORTANT LIBRARIES**
# 

# In[17]:


# Importing required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score


# ***IMPORTING DATASET***

# In[38]:


iris_data = pd.read_csv(r"C:\Users\yetes\OneDrive\Desktop\Excel\IRIS.csv")
# writing if the dataset is not present in csv file, using we are coverting
# column_names = ["sepal_length", "sepal_width","petal_length","petal_width","class"]  
# iris_data = pd.read_csv(url, names=column_names)


# In[39]:


iris_data.head(5)


# In[40]:


# type of the dataset
type(iris_data)


# In[41]:


iris_data.iloc[50:100]  # iloc() is use to slice the dataframe in pandas


# In[42]:


# statistical measures of the data for legit
iris_data.describe()


# In[44]:


# here we are doing exploratory data analysis
sns.pairplot(iris_data, hue="species")
plt.show()


# In[47]:


X = iris_data.drop("species",axis=1)
X


# In[84]:


Y = iris_data["species"]
Y


# ***MODEL TRANING***

# In[85]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3)
#7. 'random_state=42' define the data split in 42 ways otherwise it split in different ways.


# In[86]:


X_train


# In[87]:


knn = KNeighborsClassifier(n_neighbors=3)  # n_neighbors = number of neighbors and it also a hyperparameter
knn.fit(X_train,Y_train)


# ***MODEL EVALUTION***

# In[88]:


Y_pred = knn.predict(X_test)
print("Accuraacy: ",accuracy_score(Y_test,Y_pred))


# In[90]:


# another model evaluation
print(classification_report(Y_test,Y_pred))


# In[92]:


X_test.head(2)


# In[110]:


# here we take features of the flower as an input
new_data = pd.DataFrame({"sepal_length":[2.1],"sepal_width":1.5,"petal_length":1.4,"petal_width":0.2})


# In[111]:


prediction = knn.predict(new_data)                                # predict() is a function for predict


# In[112]:


prediction[0]

