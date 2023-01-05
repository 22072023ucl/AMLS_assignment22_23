#!/usr/bin/env python
# coding: utf-8

# In[12]:


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
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle


# In[13]:


predictor_path = "./shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
 
detector = dlib.get_frontal_face_detector() #a detector to find the faces
sp = dlib.shape_predictor(predictor_path ) #shape predictor to find face landmarks
facerec = dlib.face_recognition_model_v1(face_rec_model_path) #face recognition model
img_data=np.zeros((5000,128))


# In[ ]:


for i in range (5000):
    img = dlib.load_rgb_image('../datasets/celeba/img/' + str(i) + '.jpg')
    dets = detector(img, 1)  #Extract the face area in the picture
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        img_data[i,:]=face_descriptor


# In[14]:


label=pd.read_table('../datasets/celeba/labels.csv')


# x = pd.read_csv('./face feature.csv')
# x=np.array(x)
# x=x[:,1:]
# print(x)
# print(np.shape(x))

# In[16]:


# Split the data
#x=img_data
y = label['gender']

# Split the data into training and testing(75% training and 25% testing data)
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=0)# Pre-process data
from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)


# In[17]:


#Importing the Decision tree classifier from the sklearn library.
tree_params={
    'criterion':'entropy'
}
clf = tree.DecisionTreeClassifier( **tree_params )


# In[18]:


clf.fit(x_train,y_train)

#Predicting labels on the test set.
# Please complete the code below.
y_pred =  clf.predict(x_test)


# In[19]:


print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(x_train)))
print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))


# In[20]:


pickle.dump(clf,open("DecisionTree_gender_split_dlib.dat","wb")) 


loaded_model = pickle.load(open("DecisionTree_gender_split_dlib.dat","rb"))


y_pred=loaded_model.predict(x_test)


# In[ ]:




