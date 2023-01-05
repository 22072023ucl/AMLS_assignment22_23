#!/usr/bin/env python
# coding: utf-8

# In[20]:


import dlib
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


# In[21]:


predictor_path = "./shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
 
detector = dlib.get_frontal_face_detector() #a detector to find the faces
sp = dlib.shape_predictor(predictor_path ) #shape predictor to find face landmarks
facerec = dlib.face_recognition_model_v1(face_rec_model_path) #face recognition model
img_data=np.zeros((5000,128))


# In[1]:


for i in range (5000):
    img = dlib.load_rgb_image('../datasets/celeba/img/' + str(i) + '.jpg')
    dets = detector(img, 1)  #Extract the face area in the picture
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        img_data[i,:]=face_descriptor


# In[22]:


label=pd.read_table('../datasets/celeba/labels.csv')


# In[23]:


x = pd.read_csv('./face feature.csv')
x=np.array(x)
x=x[:,1:]
print(x)
print(np.shape(x))


# In[24]:


# Split the data
#x=img_data
y = label['gender']

# Split the data into training and testing(75% training and 25% testing data)
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=0)# Pre-process data
from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)


# In[25]:


logreg = LogisticRegression(solver='lbfgs',max_iter=1000)
logreg.fit(x_train, y_train)
y_pred= logreg.predict(x_test)
print('Accuracy on train set:'+str(logreg.score(x_train,y_train)))
print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))#text report showing the main classification metrics


# In[27]:


import pickle


pickle.dump(logreg,open("LogisticRegression_gender_split_dlib.dat","wb")) 


loaded_model = pickle.load(open("LogisticRegression_gender_split_dlib.dat","rb"))


y_pred=loaded_model.predict(x_test)


# In[ ]:




