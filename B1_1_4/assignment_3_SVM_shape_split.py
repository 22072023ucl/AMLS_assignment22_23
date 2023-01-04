#!/usr/bin/env python
# coding: utf-8

# In[9]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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


# In[3]:


label=pd.read_table('../datasets/cartoon_set/labels.csv')


# In[4]:


pca = PCA(n_components = 100)
pca.fit(img_data)
pca_data=pca.transform(img_data)


# In[5]:


# Split the data
x = pca_data
y = label['face_shape']
y = y[:5000]

# Split the data into training and testing(75% training and 25% testing data)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.25)

# Pre-process data

from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)


# In[6]:


clf=SVC(gamma='auto')
clf.fit(x_train,y_train)
y_pred =  clf.predict(x_test)


# In[7]:


print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(x_train)))
print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))
print(classification_report(y_test,y_pred))


# In[8]:


import pickle

#save model
pickle.dump(clf,open("SVM_shape_split.dat","wb"))  

# load model
loaded_model = pickle.load(open("SVM_shape_split.dat","rb"))


y_pred=loaded_model.predict(x_test)


# In[ ]:




