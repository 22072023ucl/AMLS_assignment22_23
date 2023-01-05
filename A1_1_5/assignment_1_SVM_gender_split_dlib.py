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
from sklearn.metrics import accuracy_score


# In[2]:


predictor_path = "./shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
 
detector = dlib.get_frontal_face_detector() #a detector to find the faces
sp = dlib.shape_predictor(predictor_path ) #shape predictor to find face landmarks
facerec = dlib.face_recognition_model_v1(face_rec_model_path) #face recognition model
img_data=np.zeros((5000,128))


# In[3]:


for i in range (5000):
    img = dlib.load_rgb_image('../datasets/celeba/img/' + str(i) + '.jpg')
    dets = detector(img, 1)  #Extract the face area in the picture
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        img_data[i,:]=face_descriptor


# In[19]:


import pandas as pd
img_data = pd.DataFrame(img_data)
img_data.to_csv('face feature.csv')


# In[7]:


label=pd.read_table('../datasets/celeba/labels.csv')


# In[8]:


# Split the data
x = img_data
y = label['gender']

# Split the data into training and testing(75% training and 25% testing data)
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=0)# Pre-process data
from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)


# In[9]:


clf=SVC(gamma='auto')
clf.fit(x_train,y_train)
y_pred =  clf.predict(x_test)


# In[10]:


print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(x_train)))
print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))


# In[11]:


import pickle

#save model
pickle.dump(clf,open("SVM_gender_split_dlib.dat","wb"))  

# load model
loaded_model = pickle.load(open("SVM_gender_split_dlib.dat","rb"))

y_pred=loaded_model.predict(x_test)


# In[ ]:




