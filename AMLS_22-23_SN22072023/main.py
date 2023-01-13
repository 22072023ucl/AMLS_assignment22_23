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
from sklearn.svm import SVC
from sklearn import tree


# In[2]:


#get image data, use this for task A1, A2, and decision tree for task B1
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


#get image label, use this for task A1, A2, and decision tree for task B1
def get_label(label_path,label_name):
    label=pd.read_table(label_path)
    y=label[label_name]
    return y


# In[4]:


# LR model, will not be used in this test code
def LogisticRegression_model(x_train,y_train,x_test,y_test):
    clf = LogisticRegression(solver='sag',fit_intercept=True,max_iter=1000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Accuracy on train set:'+str(clf.score(x_train,y_train)))
    print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
    print(classification_report(y_test,y_pred))
    return clf


# In[5]:


#SVM model, will not be used in this test code
def SVM_model(x_train,y_train,x_test,y_test):
    clf=SVC(gamma='auto')
    clf.fit(x_train,y_train)
    y_pred =  clf.predict(x_test)
    print('Accuracy on train set:'+str(clf.score(x_train,y_train)))
    print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
    print(classification_report(y_test,y_pred))
    return clf


# In[6]:


#Decision tree model, will not be used in this test code
def DecisionTree_model(x_train,y_train,x_test,y_test):
    #tree_params={'criterion':'entropy'}
    clf = tree.DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=7 )
    clf.fit(x_train,y_train)
    y_pred =  clf.predict(x_test)
    print('Accuracy on train set:'+str(clf.score(x_train,y_train)))
    print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
    print(classification_report(y_test,y_pred))
    return clf


# In[7]:


#CNN model, used for task B1 and B2
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


# In[8]:


#PCA for high dimention image data, use this for task A1, A2
def img_data_pca(img_data,dimention):
    pca = PCA(n_components = 100)
    pca.fit(img_data)
    pca_data=pca.transform(img_data)
    return pca_data


# In[9]:


#get face feature, use this for task A1
def face_feature(img_path,img_number):
    predictor_path = "./A1/shape_predictor_68_face_landmarks.dat"
    face_rec_model_path = "./A1/dlib_face_recognition_resnet_model_v1.dat"
    detector = dlib.get_frontal_face_detector() #a detector to find the faces
    sp = dlib.shape_predictor(predictor_path ) #shape predictor to find face landmarks
    facerec = dlib.face_recognition_model_v1(face_rec_model_path) #face recognition model
    img_data=np.zeros((img_number,128))
    for i in range (img_number):
        img = dlib.load_rgb_image('./datasets/'+img_path+'/img/' + str(i) + '.jpg')
        dets = detector(img, 1)  #Extract the face area in the picture
        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            img_data[i,:]=face_descriptor
    return img_data


# In[10]:


#get lip feature, use this for task A2
def lip_feature(img_path,img_number):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./A2/shape_predictor_68_face_landmarks.dat')
    lip_data=[]
    nothing_number=[]
    for i in range (img_number):
        img = cv2.imread('./datasets/'+img_path+'/img/'+str(i)+'.jpg')
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


# In[11]:


def get_color_image(img_path,img_number):
    img_data=[]
    for i in range (img_number):
        img = mpimg.imread('./datasets/'+img_path+'/img/' + str(i) + '.png')
        img=cv2.resize(img,(200,200))
        img_data.append(img)
    img_data=np.array(img_data)
    return img_data


# In[12]:


def get_cartoon_label(label_path,label_name,label_number):
    label=pd.read_table('./datasets/'+label_path+'/labels.csv')
    label= label[label_name]
    label=label[:label_number]
    label= np.array(label)
    vector2arr = np.mat(label)
    label = vector2arr.A.T
    return label


# In[ ]:


#The following is for testing 


# In[17]:


# A1 Logistic Regression raw data
##Data without dimensionality reduction and feature extraction
#get and preprocess image data for testing
img_data_test=load_image_to_vector('./datasets/celeba_test/img/',1000)
transfer=StandardScaler()
img_data_test= transfer.fit_transform(img_data_test)
#get label_test
label_test=get_label('./datasets/celeba_test/labels.csv','gender')
#load model
loaded_model = pickle.load(open("./A1/LogisticRegression_gender.dat","rb"))
label_pred=loaded_model.predict(img_data_test)
print('Accuracy on test set: '+str(accuracy_score(label_test,label_pred)))
print(classification_report(label_test,label_pred))


# In[14]:


#A1 Logistic Regression PCA data
#Data with dimensionality reduction by PCA
#get and preprocess image data for testing
img_data_test=load_image_to_vector('./datasets/celeba_test/img/',1000)
transfer=StandardScaler()
img_data_test = transfer.fit_transform(img_data_test )
#get label
label_test=get_label('./datasets/celeba_test/labels.csv','gender')
#pca for image data
img_data_test=img_data_pca(img_data_test,100)
#load model
loaded_model = pickle.load(open("./A1/LogisticRegression_gender_PCA.dat","rb"))
label_pred=loaded_model.predict(img_data_test)
print('Accuracy on test set: '+str(accuracy_score(label_test,label_pred)))
print(classification_report(label_test,label_pred))


# In[15]:


#A1 Logistic Regression feature data
#Data with feature extraction
#get and preprocess image data for testing
#img_data_test=face_feature_read('face_feature_test')
img_data_test=face_feature('celeba_test',1000)
transfer=StandardScaler()
img_data_test = transfer.fit_transform(img_data_test )
#get label
label_test=get_label('./datasets/celeba_test/labels.csv','gender')
#load model
loaded_model = pickle.load(open("./A1/LogisticRegression_gender_dlib.dat","rb"))
label_pred=loaded_model.predict(img_data_test)
print('Accuracy on test set: '+str(accuracy_score(label_test,label_pred)))
print(classification_report(label_test,label_pred))


# In[16]:


#A1 SVM raw data
##Data without dimensionality reduction and feature extraction
#get and preprocess image data for testing
img_data_test=load_image_to_vector('./datasets/celeba_test/img/',1000)
img_data_test = transfer.fit_transform(img_data_test)
#get label_test
label_test=get_label('./datasets/celeba_test/labels.csv','gender')
#load model
loaded_model = pickle.load(open("./A1/SVM_gender.dat","rb"))
label_pred=loaded_model.predict(img_data_test)
print('Accuracy on test set: '+str(accuracy_score(label_test,label_pred)))
print(classification_report(label_test,label_pred))


# In[17]:


#A1 SVM PCA data
#Data with dimensionality reduction by PCA
#get and preprocess image data for testing
img_data_test=load_image_to_vector('./datasets/celeba_test/img/',1000)
img_data_test = transfer.fit_transform(img_data_test )
#get label
label_test=get_label('./datasets/celeba_test/labels.csv','gender')
#pca for image data
img_data_test=img_data_pca(img_data_test,100)
#load model
loaded_model = pickle.load(open("./A1/SVM_gender_PCA.dat","rb"))
label_pred=loaded_model.predict(img_data_test)
print('Accuracy on test set: '+str(accuracy_score(label_test,label_pred)))
#print(classification_report(label_test,label_pred))


# In[18]:


#A1 SVM feature data
#Data with feature extraction
#get and preprocess image data for testing
#img_data_test=face_feature_read('face_feature_test')
img_data_test=face_feature('celeba_test',1000)
img_data_test = transfer.fit_transform(img_data_test )
#get label
label_test=get_label('./datasets/celeba_test/labels.csv','gender')
#load model
loaded_model = pickle.load(open("./A1/SVM_gender_dlib.dat","rb"))
label_pred=loaded_model.predict(img_data_test)
print('Accuracy on test set: '+str(accuracy_score(label_test,label_pred)))
print(classification_report(label_test,label_pred))


# In[20]:


# A2 Logistic Regression raw data
##Data without dimensionality reduction and feature extraction
#get and preprocess image data for testing
img_data_test=load_image_to_vector('./datasets/celeba_test/img/',1000)
transfer=StandardScaler()
img_data_test= transfer.fit_transform(img_data_test)
#get label_test
label_test=get_label('./datasets/celeba_test/labels.csv','smiling')
#load model
loaded_model = pickle.load(open("./A2/LogisticRegression_smiling.dat","rb"))
label_pred=loaded_model.predict(img_data_test)
print('Accuracy on test set: '+str(accuracy_score(label_test,label_pred)))
print(classification_report(label_test,label_pred))


# In[21]:


#A2 Logistic Regression PCA data
#Data with dimensionality reduction by PCA
#get and preprocess image data for testing
img_data_test=load_image_to_vector('./datasets/celeba_test/img/',1000)
transfer=StandardScaler()
img_data_test = transfer.fit_transform(img_data_test )
#get label
label_test=get_label('./datasets/celeba_test/labels.csv','smiling')
#pca for image data
img_data_test=img_data_pca(img_data_test,100)
#load model
loaded_model = pickle.load(open("./A2/LogisticRegression_smiling_PCA.dat","rb"))
label_pred=loaded_model.predict(img_data_test)
print('Accuracy on test set: '+str(accuracy_score(label_test,label_pred)))
print(classification_report(label_test,label_pred))


# In[24]:


#A2 Logistic Regression feature data
#Data with feature extraction
#get and preprocess image data for testing
img_data_test,nothing_number_test=lip_feature('celeba_test',1000)
transfer=StandardScaler()
img_data_test = transfer.fit_transform(img_data_test )
#get label
label_test=get_label('./datasets/celeba_test/labels.csv','smiling')
for i in range(len(nothing_number_test)):
    del label_test[nothing_number_test[i]]
#load model
loaded_model = pickle.load(open("./A2/LogisticRegression_smiling_dlib_lip.dat","rb"))
label_pred=loaded_model.predict(img_data_test)
print('Accuracy on test set: '+str(accuracy_score(label_test,label_pred)))
print(classification_report(label_test,label_pred))


# In[25]:


#A2 SVM raw data
##Data without dimensionality reduction and feature extraction
#get and preprocess image data for testing
img_data_test=load_image_to_vector('./datasets/celeba_test/img/',1000)
transfer=StandardScaler()
img_data_test = transfer.fit_transform(img_data_test)
#get label_test
label_test=get_label('./datasets/celeba_test/labels.csv','smiling')
#load model
loaded_model = pickle.load(open("./A2/SVM_smiling.dat","rb"))
label_pred=loaded_model.predict(img_data_test)
print('Accuracy on test set: '+str(accuracy_score(label_test,label_pred)))
print(classification_report(label_test,label_pred))


# In[26]:


#A2 SVM PCA data
#Data with dimensionality reduction by PCA
#get and preprocess image data for testing
img_data_test=load_image_to_vector('./datasets/celeba_test/img/',1000)
transfer=StandardScaler()
img_data_test = transfer.fit_transform(img_data_test )
#get label
label_test=get_label('./datasets/celeba_test/labels.csv','smiling')
#pca for image data
img_data_test=img_data_pca(img_data_test,100)
#load model
loaded_model = pickle.load(open("./A2/SVM_smiling_PCA.dat","rb"))
label_pred=loaded_model.predict(img_data_test)
print('Accuracy on test set: '+str(accuracy_score(label_test,label_pred)))
#print(classification_report(label_test,label_pred))


# In[27]:


#A2 SVM feature data
#Data with feature extraction
#get and preprocess image data for testing
img_data_test,nothing_number_test=lip_feature('celeba_test',1000)
transfer=StandardScaler()
img_data_test = transfer.fit_transform(img_data_test )
#get label
label_test=get_label('./datasets/celeba_test/labels.csv','smiling')
for i in range(len(nothing_number_test)):
    del label_test[nothing_number_test[i]]
#load model
loaded_model = pickle.load(open("./A2/SVM_smiling_dlib_lip.dat","rb"))
label_pred=loaded_model.predict(img_data_test)
print('Accuracy on test set: '+str(accuracy_score(label_test,label_pred)))
print(classification_report(label_test,label_pred))


# In[29]:


#get image data ,this is used for Decision Tree for B1
def load_image_to_vector_DT(image_path,image_number):
    img_data=[]
    for i in range (image_number):
        img_color= mpimg.imread(image_path + str(i) + '.png')
        img=cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(200,200))
        img_vector=img.reshape(40000)
        img_data.append(img_vector)
    return img_data


# In[31]:


#B1 decision tree 
#get image data
img_data_test=load_image_to_vector_DT('./datasets/cartoon_set_test/img/',2500)
transfer=StandardScaler()
img_data_test= StandardScaler().fit_transform(img_data_test)
#get label
label_test=get_label('./datasets/cartoon_set_test/labels.csv','face_shape')
#load model
loaded_model = pickle.load(open("./B1/DecisionTree_shape.dat","rb"))

label_pred=loaded_model.predict(img_data_test)
print('Accuracy on test set: '+str(accuracy_score(label_test,label_pred)))
print(classification_report(label_test,label_pred))


# In[33]:


#B1 CNN
#get image data
img_data_test=get_color_image('cartoon_set_test',2500)
#get label
label_test=get_cartoon_label('cartoon_set_test','face_shape',2500)
#load model
loaded_model = pickle.load(open("./B1/CNN_shape.dat","rb"))
label_pred=loaded_model.evaluate(img_data_test,label_test)


# In[34]:


#B2 CNN
#get image data
img_data_test=get_color_image('cartoon_set_test',2500)
#get label
label_test=get_cartoon_label('cartoon_set_test','eye_color',2500)
#load model
loaded_model = pickle.load(open("./B2/CNN_color.dat","rb"))
label_pred=loaded_model.evaluate(img_data_test,label_test)


# In[ ]:




