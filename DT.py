#!/usr/bin/env python
# coding: utf-8

# -----To Read and Display the Dataset "dataset.csv"-----                                                                         
# -> Dataset is present locally in /Home/Documents/Python Examples                                                                   
# -> "dataset.csv" is a csv file which contains data about Breast Cancer patients                                             
# ->  There are 569 records(samples) each having 32 attributes(characteristics)                                              
#     

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
import urllib 
import csv
import sys
import time

#start_time=time.clock()

#-----Filename through Command line Argument-----
get_ipython().run_line_magic('run', 'CLA.py dataset.csv')



#data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
print(data)
print(data[["id","radius_mean","diagnosis"]])
print(data.shape)


# In[2]:


#-----Filename through path name-----

data=pd.read_csv("/home/shashank/Documents/Python Examples/dataset.csv")

#data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
d=(data)
print(data)
print(data[["id","radius_mean","diagnosis"]])
print(data.shape)



# In[4]:


#-----Filename through Web-----

data =pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")

#data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
print(data)
print(data[["id","radius_mean","diagnosis"]])
print(data.shape)


# In[5]:


#-----Filename through user input-----

filename = input()
data=pd.read_csv(filename)

data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
print(data)
print(data[["id","radius_mean","diagnosis"]])
print(data.shape)


# ----- To seprate Attributes and Target -----                                                                                             
# -> X - Non Target Attributes                                                                                                     
# -> Y - Target Attribute                                                                                                          
# -> In our dataset last attribute(column) is the target attribute(class label)                                                  
#            

# In[6]:


X = data.iloc[:,1:31]
Y = data.iloc[:,-1]
Y=pd.DataFrame(Y)


# In[7]:


print(X)

file=open("test.txt",'w')
file.write(X)
file.close()


# In[8]:


print(Y)


# ----- Transform class label into bits(0 and 1) -----                                                                           
# -> Malignant(M) - 1                                                                                                             
# -> Benign(B)- 0

# In[9]:


from sklearn import preprocessing
labelencoder_Y = preprocessing.LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
#Y=pd.DataFrame(Y)

print(Y)




# ----- Split dataset into train and test subset -----                                                                           
# -> X_train - Non target training subset                                                                                       
# -> Y_train - target training subset                                                                                       
# -> X_test - Non target testing subset
# -> Y_test - target testing subset                                                                                              
#                                                                                                                                    
# -> Here the Dataset is divided into 25%(0.25) Testing subset and remaining 75%(0.75) as Training subset                       

# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25,random_state=0)
print(X_train)


# -----Random Forest Classifier-----                                                                                                      
# -> Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

# In[17]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


# ----- Support Vector Machines -----                                                                                                 
# ->In SVM,given labeled training data,the algorithm outputs an optimal hyperplane which categorizes new examples. In two dimentional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side.

# In[18]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)


# ----- Creating Decision tree -----                                                                                                 
# -> DecisionTreeClassifer package is used to create a decision tree using 'entropy' and 'information gain' as criteria                                   for attribute selection 

# In[11]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
classifier.fit(X_train, Y_train)


# In[18]:


from sklearn.ensemble import BaggingClassifier
bagging_clf= BaggingClassifier()
bagging_clf.fit(X_train,Y_train.ravel())
Y_pred=bagging_clf.predict(X_test)


# In[22]:


from sklearn.ensemble import GradientBoostingClassifier
boost_clf=GradientBoostingClassifier()
boost_clf.fit(X_train,Y_train.ravel())
Y_pred=boost_clf.predict(X_test)


# In[12]:


print(X_test)
Y_pred = classifier.predict(X_test)


# -> Y_Test contains the assumed(actual) class label values                                                                         
# -> Y_pred contains the predicted class label values using Decision tree 

# In[13]:


print(Y_test)
#print(Y_pred)


# In[14]:


print(Y_pred)


# ----- Confusion Matrix -----                                                                                        
# -> To get summary of predcition results                                                                                          
# -> It tells us about the number of correct and incorrect prediciton 

# In[15]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)


# ----- Accuracy -----                                                                                                                      
# -> Accuracy - It tells us how many predictions are corrrect                                                                       

# In[16]:


acc=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
acc*=100
print(acc)
#Y=pd.DataFrame(Y)
#print(Y.columns)
#print(Y)


# In[17]:


#Y = data.iloc[:,-1]

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,  
                filled=True,feature_names=X.columns,class_names=['0','1'],rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

