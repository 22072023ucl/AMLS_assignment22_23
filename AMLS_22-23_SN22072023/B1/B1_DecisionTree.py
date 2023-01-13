#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import pickle
from sklearn import tree
from sklearn.model_selection import train_test_split
import tensorflow as tf


# In[2]:


#this function load the RGB image and change to gray inage, change the 2D image data to 1D and form a matrix
def load_image_to_vector(image_path,image_number):
    img_data=[]
    for i in range (image_number):
        img_color= mpimg.imread(image_path + str(i) + '.png')
        img=cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(200,200))
        img_vector=img.reshape(40000)
        img_data.append(img_vector)
    return img_data


# In[3]:


def get_label(label_path,label_name):
    label=pd.read_table(label_path)
    y=label[label_name]
    return y


# In[4]:


def DecisionTree_model(x_train,y_train,x_test,y_test):
    #tree_params={'criterion':'entropy'}
    clf = tree.DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=7 )
    clf.fit(x_train,y_train)
    y_pred =  clf.predict(x_test)
    print('Accuracy on train set:'+str(clf.score(x_train,y_train)))
    print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
    print(classification_report(y_test,y_pred))
    return clf


# In[5]:


get_ipython().run_cell_magic('time', '', '#get image data\nimg_data=load_image_to_vector(\'../datasets/cartoon_set/img/\',5000)\n#img_data_test=load_image_to_vector(\'../datasets/celeba_test/img/\',1000)\n#get label\nlabel=get_label(\'../datasets/cartoon_set/labels.csv\',\'face_shape\')\nlabel=label[0:5000]\n#y_test=get_label(\'../datasets/celeba_test/labels.csv\',\'gender\')\n#Standardize the data\nx_train, x_test, y_train, y_test = train_test_split(img_data, label,test_size=0.1,random_state=0)\ntransfer = StandardScaler()\nx_train = transfer.fit_transform(x_train)\nx_test = transfer.transform(x_test)\n#train the model and report accuracy\nmodel=DecisionTree_model(x_train,y_train,x_test,y_test)\nfig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)\ntree.plot_tree(model)\n#save model\npickle.dump(model,open("DecisionTree_shape.dat","wb")) ')


# In[6]:


get_ipython().run_cell_magic('time', '', '#get image data\nimg_data_test=load_image_to_vector(\'../datasets/cartoon_set_test/img/\',2500)\nimg_data_test= StandardScaler().fit_transform(img_data_test)\n#get label\nlabel_test=get_label(\'../datasets/cartoon_set_test/labels.csv\',\'face_shape\')\n#load model\nloaded_model = pickle.load(open("DecisionTree_shape.dat","rb"))\n\nlabel_pred=loaded_model.predict(img_data_test)\nprint(\'Accuracy on test set: \'+str(accuracy_score(label_test,label_pred)))\nprint(classification_report(label_test,label_pred))')


# In[7]:


fig.savefig('treestructure.png')


# In[ ]:




