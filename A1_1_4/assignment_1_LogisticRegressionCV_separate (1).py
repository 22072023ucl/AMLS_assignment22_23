#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
 
#read one img to get size
img_0 = mpimg.imread('../datasets/celeba/img/0.jpg')
img_shape=np.shape(img_0)
img_data=np.zeros((5000,img_shape[0]*img_shape[1]))
for i in range (5000):
    img = mpimg.imread('../datasets/celeba/img/' + str(i) + '.jpg')
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_vector=img.reshape(1,img_shape[0]*img_shape[1])
    img_data[i,:]=img_vector


# In[2]:


label=pd.read_table('../datasets/celeba/labels.csv')


# In[3]:


#The dimension of PCA will impact the result but seems that a larger dimension does not guarantee a better result
pca = PCA(n_components = 100)
pca.fit(img_data)
pca_data=pca.transform(img_data)


# In[4]:


#use separate test data
#test image
img_data_test=np.zeros((1000,img_shape[0]*img_shape[1]))
for i in range (1000):
    img_test = mpimg.imread('../datasets/celeba_test/img/' + str(i) + '.jpg')
    img_test=cv2.cvtColor(img_test,cv2.COLOR_BGR2GRAY)
    img_vector_test=img_test.reshape(1,img_shape[0]*img_shape[1])
    img_data_test[i,:]=img_vector_test


# In[5]:


#test label
label_test=pd.read_table('../datasets/celeba_test/labels.csv')


# In[6]:


# pca for test data
pca = PCA(n_components = 100)
pca.fit(img_data_test)
pca_data_test=pca.transform(img_data_test)


# In[7]:


x_train=pca_data
y_train=label['gender']
x_test=pca_data_test
y_test=label_test['gender']

from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)


# In[8]:


logregcv = LogisticRegressionCV(solver='lbfgs',cv=50,max_iter=1000)
logregcv.fit(x_train, y_train)
y_pred_cv= logregcv.predict(x_test)


# In[9]:


import pickle

#save model
pickle.dump(logregcv,open("LogisticRegressionCV_gender_separate.dat","wb"))  

# load model
loaded_model = pickle.load(open("LogisticRegressionCV_gender_separate.dat","rb"))


y_pred=loaded_model.predict(x_test)


# In[10]:


print('Accuracy on train set：%.3f'% logregcv.score(x_train,y_train))
print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))#text report showing the main classification metrics


# In[ ]:




