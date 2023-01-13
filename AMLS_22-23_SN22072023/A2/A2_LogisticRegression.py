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
from sklearn.linear_model import LogisticRegression
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


# In[13]:


def LogisticRegression_model(x_train,y_train,x_test,y_test):
    clf = LogisticRegression(solver='newton-cg',fit_intercept=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Accuracy on train set:'+str(clf.score(x_train,y_train)))
    print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
    print(classification_report(y_test,y_pred))
    return clf


# In[6]:


def img_data_pca(img_data,dimention):
    pca = PCA(n_components = 100)
    pca.fit(img_data)
    pca_data=pca.transform(img_data)
    return pca_data


# In[5]:


def lip_feature(img_path,img_number):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    lip_data=[]
    nothing_number=[]
    for i in range (img_number):
        img = cv2.imread('../datasets/'+img_path+'/img/'+str(i)+'.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        positions_68_arr = []
        faces = detector(img_gray, 0)
        if len(faces) !=0:
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img, faces[0]).parts()])
            for idx, point in enumerate(landmarks):
                #coordinates of 68 points
                pos = (point[0, 0], point[0, 1])
                positions_68_arr.append(pos)
            positions_lip_arr = []
            for i in range(48, 68):
                positions_lip_arr.append(positions_68_arr[i][0])
                positions_lip_arr.append(positions_68_arr[i][1])
            lip_data.append(positions_lip_arr)
        else:
            nothing_number.append(i)
            continue
    return lip_data,nothing_number


# In[8]:


def lip_feature_read(csv_name):
    img_data = pd.read_csv('./'+csv_name+'.csv')
    img_data=np.array(img_data)
    img_data=img_data[:,1:]
    return img_data


# In[9]:


def learn_curve(x_train,y_train,x_test,y_test):
    l2_iter = []
    l2_iter_t = []
    iters = np.arange(600,1000,100)
    for i in iters:
        lr2 = LogisticRegression(penalty="l2",solver='sag',max_iter=i,random_state=0)
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


# In[9]:


get_ipython().run_cell_magic('time', '', '#Data without dimensionality reduction and feature extraction\n#get image data\nimg_data=load_image_to_vector(\'../datasets/celeba/img/\',5000)\n#get label\nlabel=get_label(\'../datasets/celeba/labels.csv\',\'smiling\')\n#Standardize the data\nx_train, x_test, y_train, y_test = train_test_split(img_data, label,test_size=0.1,random_state=0)\ntransfer=StandardScaler()\nx_train = transfer.fit_transform(x_train)\nx_test = transfer.transform(x_test)\n#train the model and report accuracy\nmodel=LogisticRegression_model(x_train,y_train,x_test,y_test)\n#save model\npickle.dump(model,open("LogisticRegression_smiling.dat","wb")) ')


# In[10]:


get_ipython().run_cell_magic('time', '', '#Data with dimensionality reduction by PCA\n#get image data\nimg_data=load_image_to_vector(\'../datasets/celeba/img/\',5000)\n#get label\nlabel=get_label(\'../datasets/celeba/labels.csv\',\'smiling\')\n#pca for image data\nx_train, x_test, y_train, y_test = train_test_split(img_data, label,test_size=0.1,random_state=0)\nx_train=img_data_pca(x_train,100)\nx_test=img_data_pca(x_test,100)\n#Standardize the data\ntransfer=StandardScaler()\nx_train = transfer.fit_transform(x_train)\nx_test = transfer.transform(x_test)\n#train the model and report accuracy\nmodel=LogisticRegression_model(x_train,y_train,x_test,y_test)\n#save model\npickle.dump(model,open("LogisticRegression_smiling_PCA.dat","wb")) ')


# In[14]:


get_ipython().run_cell_magic('time', '', '#Data with feature extraction\nimg_data,nothing_number=lip_feature(\'celeba\',5000)\n#get label\nlabel=get_label(\'../datasets/celeba/labels.csv\',\'smiling\')\nfor i in range(len(nothing_number)):\n    del label[nothing_number[i]]\n#Standardize the data\nx_train, x_test, y_train, y_test = train_test_split(img_data, label,test_size=0.1,random_state=0)\ntransfer=StandardScaler()\nx_train = transfer.fit_transform(x_train)\nx_test = transfer.transform(x_test)\n#train the model and report accuracy\nmodel=LogisticRegression_model(x_train,y_train,x_test,y_test)\n#save model\npickle.dump(model,open("LogisticRegression_smiling_dlib_lip.dat","wb")) ')


# In[12]:


#learning_curve for LR
learn_curve(x_train,y_train,x_test,y_test)


# In[ ]:


#The following is for testing 


# In[13]:


get_ipython().run_cell_magic('time', '', '##Data without dimensionality reduction and feature extraction\n#get and preprocess image data for testing\nimg_data_test=load_image_to_vector(\'../datasets/celeba_test/img/\',1000)\nimg_data_test = transfer.fit_transform(img_data_test)\n#get label_test\nlabel_test=get_label(\'../datasets/celeba_test/labels.csv\',\'smiling\')\n#load model\nloaded_model = pickle.load(open("LogisticRegression_smiling.dat","rb"))\nlabel_pred=loaded_model.predict(img_data_test)\nprint(\'Accuracy on test set: \'+str(accuracy_score(label_test,label_pred)))\nprint(classification_report(label_test,label_pred))')


# In[14]:


get_ipython().run_cell_magic('time', '', '#Data with dimensionality reduction by PCA\n#get and preprocess image data for testing\nimg_data_test=load_image_to_vector(\'../datasets/celeba_test/img/\',1000)\nimg_data_test = transfer.fit_transform(img_data_test )\n#get label\nlabel_test=get_label(\'../datasets/celeba_test/labels.csv\',\'smiling\')\n#pca for image data\nimg_data_test=img_data_pca(img_data_test,100)\n#load model\nloaded_model = pickle.load(open("LogisticRegression_smiling_PCA.dat","rb"))\nlabel_pred=loaded_model.predict(img_data_test)\nprint(\'Accuracy on test set: \'+str(accuracy_score(label_test,label_pred)))\nprint(classification_report(label_test,label_pred))')


# In[15]:


get_ipython().run_cell_magic('time', '', '#Data with feature extraction\n#get and preprocess image data for testing\nimg_data_test,nothing_number_test=lip_feature(\'celeba_test\',1000)\nimg_data_test = transfer.fit_transform(img_data_test )\n#get label\nlabel_test=get_label(\'../datasets/celeba_test/labels.csv\',\'smiling\')\nfor i in range(len(nothing_number_test)):\n    del label_test[nothing_number_test[i]]\n#load model\nloaded_model = pickle.load(open("LogisticRegression_smiling_dlib_lip.dat","rb"))\nlabel_pred=loaded_model.predict(img_data_test)\nprint(\'Accuracy on test set: \'+str(accuracy_score(label_test,label_pred)))\nprint(classification_report(label_test,label_pred))')


# In[ ]:




