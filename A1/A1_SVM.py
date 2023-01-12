#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import pickle
import dlib


# In[2]:


def load_image_to_vector(image_path,image_number):
    img_data=[]
    for i in range (image_number):
        img_color= mpimg.imread(image_path + str(i) + '.jpg')
        img=cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
        width=np.shape(img_color)[0]
        height=np.shape(img_color)[1]
        img_vector=img.reshape(width*height)
        img_data.append(img_vector)
    return img_data


# In[3]:


def get_label(label_path,label_name):
    label=pd.read_table(label_path)
    y=label[label_name]
    return y


# In[4]:


def SVM_model(x_train,y_train,x_test,y_test):
    clf=SVC(gamma='auto')
    clf.fit(x_train,y_train)
    y_pred =  clf.predict(x_test)
    print('Accuracy on train set:'+str(clf.score(x_train,y_train)))
    print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
    print(classification_report(y_test,y_pred))
    return clf


# In[5]:


def img_data_pca(img_data,dimention):
    pca = PCA(n_components = 100)
    pca.fit(img_data)
    pca_data=pca.transform(img_data)
    return pca_data


# In[11]:


def face_feature(img_path,img_number):
    predictor_path = "./shape_predictor_68_face_landmarks.dat"
    face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
    detector = dlib.get_frontal_face_detector() #a detector to find the faces
    sp = dlib.shape_predictor(predictor_path ) #shape predictor to find face landmarks
    facerec = dlib.face_recognition_model_v1(face_rec_model_path) #face recognition model
    img_data=np.zeros((img_number,128))
    for i in range (img_number):
        img = dlib.load_rgb_image('../datasets/'+img_path+'/img/' + str(i) + '.jpg')
        dets = detector(img, 1)  #Extract the face area in the picture
        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            img_data[i,:]=face_descriptor
    return img_data


# In[7]:


def face_feature_read(csv_name):
    img_data = pd.read_csv('./'+csv_name+'.csv')
    img_data=np.array(img_data)
    img_data=img_data[:,1:]
    return img_data


# In[8]:


def learn_curve(x_train,y_train,x_test,y_test):
    l2_iter = []
    l2_iter_t = []
    iters = np.arange(5000,10000,500)
    for i in iters:
        lr2 = SVC(gamma='auto',max_iter=i)
        lr2 = lr2.fit(x_train,y_train)
        l2_iter.append(accuracy_score(lr2.predict(x_train),y_train))
        l2_iter_t.append(accuracy_score(lr2.predict(x_test),y_test))
    plt.plot(figsize=(20,6))
    plt.plot(iters,l2_iter,label='accuracy')
    plt.plot(iters,l2_iter_t,label='val_accuracy')
    plt.xticks(iters)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# In[17]:


get_ipython().run_cell_magic('time', '', '#Data without dimensionality reduction and feature extraction\n#get image data\nimg_data=load_image_to_vector(\'../datasets/celeba/img/\',5000)\n#get label\nlabel=get_label(\'../datasets/celeba/labels.csv\',\'gender\')\n#Standardize the data\nx_train, x_test, y_train, y_test = train_test_split(img_data, label,test_size=0.1,random_state=0)\ntransfer=StandardScaler()\nx_train = transfer.fit_transform(x_train)\nx_test = transfer.transform(x_test)\n#train the model and report accuracy\nmodel=SVM_model(x_train,y_train,x_test,y_test)\n#save model\npickle.dump(model,open("SVM_gender.dat","wb")) ')


# In[9]:


get_ipython().run_cell_magic('time', '', '#Data with dimensionality reduction by PCA\n#get image data\nimg_data=load_image_to_vector(\'../datasets/celeba/img/\',5000)\n#get label\nlabel=get_label(\'../datasets/celeba/labels.csv\',\'gender\')\n#pca for image data\nx_train, x_test, y_train, y_test = train_test_split(img_data, label,test_size=0.1,random_state=0)\nx_train=img_data_pca(x_train,100)\nx_test=img_data_pca(x_test,100)\n#Standardize the data\ntransfer=StandardScaler()\nx_train = transfer.fit_transform(x_train)\nx_test = transfer.transform(x_test)\n#train the model and report accuracy\nmodel=SVM_model(x_train,y_train,x_test,y_test)\n#save model\npickle.dump(model,open("SVM_gender_PCA.dat","wb")) ')


# In[12]:


get_ipython().run_cell_magic('time', '', '#Data with feature extraction\n#img_data=face_feature_read(\'face_feature_train\')\nimg_data=face_feature(\'celeba\',5000)\n#get label\nlabel=get_label(\'../datasets/celeba/labels.csv\',\'gender\')\n#Standardize the data\nx_train, x_test, y_train, y_test = train_test_split(img_data, label,test_size=0.1,random_state=0)\ntransfer=StandardScaler()\nx_train = transfer.fit_transform(x_train)\nx_test = transfer.transform(x_test)\n#train the model and report accuracy\nmodel=SVM_model(x_train,y_train,x_test,y_test)\n#save model\npickle.dump(model,open("SVM_gender_dlib.dat","wb")) ')


# In[13]:


#learning_curve for LR
learn_curve(x_train,y_train,x_test,y_test)


# In[ ]:


#The following is for testing 


# In[14]:


get_ipython().run_cell_magic('time', '', '##Data without dimensionality reduction and feature extraction\n#get and preprocess image data for testing\nimg_data_test=load_image_to_vector(\'../datasets/celeba_test/img/\',1000)\nimg_data_test = transfer.fit_transform(img_data_test)\n#get label_test\nlabel_test=get_label(\'../datasets/celeba_test/labels.csv\',\'gender\')\n#load model\nloaded_model = pickle.load(open("SVM_gender.dat","rb"))\nlabel_pred=loaded_model.predict(img_data_test)\nprint(\'Accuracy on test set: \'+str(accuracy_score(label_test,label_pred)))\nprint(classification_report(label_test,label_pred))')


# In[15]:


get_ipython().run_cell_magic('time', '', '#Data with dimensionality reduction by PCA\n#get and preprocess image data for testing\nimg_data_test=load_image_to_vector(\'../datasets/celeba_test/img/\',1000)\nimg_data_test = transfer.fit_transform(img_data_test )\n#get label\nlabel_test=get_label(\'../datasets/celeba_test/labels.csv\',\'gender\')\n#pca for image data\nimg_data_test=img_data_pca(img_data_test,100)\n#load model\nloaded_model = pickle.load(open("SVM_gender_PCA.dat","rb"))\nlabel_pred=loaded_model.predict(img_data_test)\nprint(\'Accuracy on test set: \'+str(accuracy_score(label_test,label_pred)))\n#print(classification_report(label_test,label_pred))')


# In[17]:


get_ipython().run_cell_magic('time', '', '#Data with feature extraction\n#get and preprocess image data for testing\n#img_data_test=face_feature_read(\'face_feature_test\')\nimg_data_test=face_feature(\'celeba_test\',1000)\nimg_data_test = transfer.fit_transform(img_data_test )\n#get label\nlabel_test=get_label(\'../datasets/celeba_test/labels.csv\',\'gender\')\n#load model\nloaded_model = pickle.load(open("SVM_gender_dlib.dat","rb"))\nlabel_pred=loaded_model.predict(img_data_test)\nprint(\'Accuracy on test set: \'+str(accuracy_score(label_test,label_pred)))\nprint(classification_report(label_test,label_pred))')


# In[ ]:




