#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

#read one img to get size
img_0 = mpimg.imread('../datasets/cartoon_set/img/0.png')
img_shape=np.shape(img_0)
img_data=np.zeros((5000,img_shape[0]*img_shape[1]))
for i in range (5000):
    img = mpimg.imread('../datasets/cartoon_set/img/' + str(i) + '.png')
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_vector=img.reshape(1,img_shape[0]*img_shape[1])
    img_data[i,:]=img_vector


# In[8]:


label=pd.read_table('../datasets/cartoon_set/labels.csv')
print(label)


# In[9]:


pca = PCA(n_components = 100)
pca.fit(img_data)
pca_data=pca.transform(img_data)


# def IncrementalPCA(data):
#     # 使用默认的 n_components
#     pca=decomposition.IncrementalPCA(n_components=5,batch_size=10)
#     pca.partial_fit(data)
#     pca_data = pca.transform(data)
#     print(pca.n_components_)
#     return pca_data
# pca_data=IncrementalPCA(img_data)

# In[11]:


# Split the data
x = pca_data
y = label['face_shape']
y = y[:5000]
# Split the data into training and testing(75% training and 25% testing data)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 47, test_size = 0.25)

# Pre-process data

from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)
scaler = MinMaxScaler() 
# This estimator scales and translates each feature individually such that it is in the given range on the training set, default between(0,1)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[12]:


#Importing the Decision tree classifier from the sklearn library.
tree_params={
    'criterion':'entropy'
}
clf = tree.DecisionTreeClassifier( **tree_params )


# In[13]:


clf.fit(x_train,y_train)


#Predicting labels on the test set.
# Please complete the code below.
y_pred =  clf.predict(x_test)


# In[14]:


print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(x_train)))
print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))


# In[ ]:




