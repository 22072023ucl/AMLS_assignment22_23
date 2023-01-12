#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


# In[2]:


def get_color_image(img_path,img_number):
    img_data=[]
    for i in range (img_number):
        img = mpimg.imread('../datasets/'+img_path+'/img/' + str(i) + '.png')
        img=cv2.resize(img,(200,200))
        img_data.append(img)
    img_data=np.array(img_data)
    return img_data


# In[3]:


def get_cartoon_label(label_path,label_name,label_number):
    label=pd.read_table('../datasets/'+label_path+'/labels.csv')
    label= label[label_name]
    label=label[:label_number]
    label= np.array(label)
    vector2arr = np.mat(label)
    label = vector2arr.A.T
    return label


# In[4]:


def train_model(x_train, x_test, y_train, y_test):
    model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200,200,4)),  # convolutional layer 1，convolution kernel 3*3
    layers.MaxPooling2D((2, 2)),  # pooling layer 1，2*2
    layers.Conv2D(64, (3, 3), activation='relu'),  # convolutional layer 2，convolution kernel 3*3
    layers.MaxPooling2D((2, 2)),  # pooling layer 2，2*2
    layers.Conv2D(64, (3, 3), activation='relu'),  # convolutional layer 3，convolution kernel 3*3
    layers.Dropout(.2),


    layers.Flatten(),  # Flatten
    layers.Dense(128, activation='relu'),  # FP layer
    layers.Dropout(.2),
    layers.Dense(64, activation='relu'),  # FP layer
    layers.Dense(10)  # output layer
    ])
    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    return model


# In[6]:


get_ipython().run_cell_magic('time', '', '#get color image data\nimg_data=get_color_image(\'cartoon_set\',5000)\n#get labels\nlabel=get_cartoon_label(\'cartoon_set\',\'face_shape\',5000)\n#train set and validation set\nx_train, x_test, y_train, y_test = train_test_split(img_data, label,test_size=0.1,random_state=0)\n#model\nmodel=train_model(x_train, x_test, y_train, y_test)\n#save model\npickle.dump(model,open("CNN_shape.dat","wb"))  ')


# In[10]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.4, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(x_test,y_test, verbose=2)
plt.savefig("./B1_accuracy.jpg") 


# In[11]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim([0, 1.7])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(x_test,y_test, verbose=2)
plt.savefig("./B1_loss.jpg") 


# In[8]:


get_ipython().run_cell_magic('time', '', 'img_data_test=get_color_image(\'cartoon_set_test\',2500)\nlabel_test=get_cartoon_label(\'cartoon_set_test\',\'face_shape\',2500)\nloaded_model = pickle.load(open("CNN_shape.dat","rb"))\nlabel_pred=loaded_model.evaluate(img_data_test,label_test)')


# In[ ]:




