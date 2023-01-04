#!/usr/bin/env python
# coding: utf-8

# In[9]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.linear_model import LogisticRegressionCV

#read one img to get size
img_0 = mpimg.imread('../datasets/celeba/img/0.jpg')
img_shape=np.shape(img_0)
img_data=np.zeros((5000,img_shape[0]*img_shape[1]))
for i in range (5000):
    img = mpimg.imread('../datasets/celeba/img/' + str(i) + '.jpg')
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_vector=img.reshape(1,img_shape[0]*img_shape[1])
    img_data[i,:]=img_vector


# In[10]:


label=pd.read_table('../datasets/celeba/labels.csv')


# In[11]:


pca = PCA(n_components = 100)
pca.fit(img_data)
pca_data=pca.transform(img_data)


# In[12]:


# Split the data
x = pca_data
y = label['gender']

# Split the data into training and testing(75% training and 25% testing data)
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=0)# Pre-process data
from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)


# In[13]:


logreg = LogisticRegression(solver='lbfgs',max_iter=1000)
logreg.fit(x_train, y_train)
y_pred= logreg.predict(x_test)
print('Accuracy on train set:'+str(logreg.score(x_train,y_train)))
print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))#text report showing the main classification metrics


# In[14]:


import pickle


pickle.dump(logreg,open("LogisticRegression_gender_split.dat","wb")) 


loaded_model = pickle.load(open("LogisticRegression_gender_split.dat","rb"))


y_pred=loaded_model.predict(x_test)


# In[ ]:




