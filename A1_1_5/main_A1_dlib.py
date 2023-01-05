#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dlib
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
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


# In[2]:


predictor_path = "./A1/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "./A1/dlib_face_recognition_resnet_model_v1.dat"
 
detector = dlib.get_frontal_face_detector() #a detector to find the faces
sp = dlib.shape_predictor(predictor_path ) #shape predictor to find face landmarks
facerec = dlib.face_recognition_model_v1(face_rec_model_path) #face recognition model
img_data=np.zeros((1000,128))


# In[3]:


for i in range (1000):
    img = dlib.load_rgb_image('./datasets/celeba_test/img/' + str(i) + '.jpg')
    dets = detector(img, 1)  #Extract the face area in the picture
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        img_data[i,:]=face_descriptor


# In[4]:


img_data_test= pd.DataFrame(img_data)
img_data_test.to_csv('face feature test.csv')


# In[5]:


#test label
label_test=pd.read_table('./datasets/celeba_test/labels.csv')


# In[6]:


x_test = img_data
y_test = label_test['gender']
x_test = StandardScaler().fit_transform(x_test)


# In[10]:


#SVM
loaded_model = pickle.load(open("./A1/SVM_gender_split_dlib.dat","rb"))
y_pred=loaded_model.predict(x_test)
print('SVM:Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))#text report showing the main classification metrics


# In[11]:


#LogisticRegression
loaded_model = pickle.load(open("./A1/LogisticRegression_gender_split_dlib.dat","rb"))
y_pred=loaded_model.predict(x_test)
print('SVM:Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))#text report showing the main classification metrics


# In[12]:


#DecisionTree
loaded_model = pickle.load(open("./A1/DecisionTree_gender_split_dlib.dat","rb"))
y_pred=loaded_model.predict(x_test)
print('SVM:Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))#text report showing the main classification metrics


# In[ ]:




