# AMLS_22-23_SN12345678
This repository contains the final code files for AMLS assignment.
## **Project Discription**
The project is named as **"AMLS_22-23_SN22072023"**. The main purpose of this project is to use multiple machine learning models to complete four classification tasks. The specific task description, datasets and applied models are as follows.
- A1
  - task name: gender detection
  - dataset for training and validation: celeba
  - dataset for testing: celeba_test
  - models: Logistic Regression, SVM
- A2
  - task name: emotion detection
  - dataset for training and validation: celeba
  - dataset for testing: celeba_test
  - models: Logistic Regression, SVM 
- B1
  - task name: face shape recognition
  - dataset for training and validation: cartoon_set
  - dataset for testing: cartoon_set_test
  - models: Decision Tree, CNN
- B2
  - task name: eye color recognition
  - dataset for training and validation: cartoon_set
  - dataset for testing: cartoon_set_test
  - models: CNN
  
## **Role of files**
### folders
There are four folders: A1, A2, A3 and A4 in this project. Each folder contains the code and models for the corresponding task. The code was written in Jupyter Notebook, which has a format of ".ipynb". **It is recommended to run the code in Jupyter Notebook.** However, the same code with a format of ".py" is also provided. Here are the details.
- A1
   - A1_LogisticRegression.ipynb or A1_LogisticRegression.py : Apply logistic Regression in task A1.
   - A1_SVM.ipynb or A1_SVM.py : Apply SVM in task A1.
   - LogisticRegression_gender.dat: a trained logistic regression model with raw image data for task A1
   - LogisticRegression_gender_PCA.dat: a trained logistic regression model with PCA data for task A1
   - LogisticRegression_gender_dlib.dat: a trained logistic regression model with feature data for task A1
   - SVM_gender.dat: a trained SVM model with raw image data for task A1
   - SVM_gender_PCA.dat: a trained SVM model with PCA data for task A1
   - SVM_gender_dlib.dat: a trained SVM model with feature data for task A1
   - dlib_face_recognition_resnet_model_v1.dat: a trained model downloaded from open source library to help extract facial features
   - shape_predictor_68_face_landmarks.dat: another trained model downloaded from open source library to help extract facial features
- A2
   - A2_LogisticRegression.ipynb or A2_LogisticRegression.py : Apply logistic Regression in task A2.
   - A2_SVM.ipynb or A2_SVM.py : Apply SVM in task A2.
   - LogisticRegression_smiling.dat: a trained logistic regression model with raw image data for task A2
   - LogisticRegression_smiling_PCA.dat: a trained logistic regression model with PCA data for task A2
   - LogisticRegression_smiling_dlib_lip: a trained logistic regression model with feature data for task A2
   - SVM_smiling.dat: a trained SVM model with raw image data for task A2
   - SVM_smiling_PCA.dat: a trained SVM model with PCA data for task A2
   - SVM_smiling_dlib.dat: a trained SVM model with feature data for task A2
   - dlib_face_recognition_resnet_model_v1.dat: a trained model downloaded from open source library to help extract facial features
   - shape_predictor_68_face_landmarks.dat: another trained model downloaded from open source library to help extract facial features
- B1
   - B1_DecisionTree.ipynb or B1_DecisionTree.py : Apply decision tree in task B1.
   - B1_CNN.ipynb or B1_CNN.py : Apply CNN in task B1.
   - DecisionTree_shape : a trained decision tree model for task B1
   - CNN_shape : a trained CNN model FOR task B1
- B2
   - B2_CNN.ipynb or B2_CNN.py : Apply CNN in task B2.
   - CNN_color.dat: a trained CNN model for task B2
   
### main.ipynb or main.py
This code file is to run the whole project. **It is also recommended to use "main.ipynb" and run it in Jupyter Notebook.** For some models or methods, running the entire code is very time-consuming, so here it will directly call the models in each folder and use the test set data to make predictions. If you want to view the details of model training, please go back to the code files in each folder.

### datasets
This file contains folders of "celeba, celeba_test, cartoon_set, cartoon_set_test" when the project develops. It is empty now. Anyone who want to run the project should Copy and paste the dataset into this folder. (You may need to create a new folder called "datasets" for this purpose.)

## Packages required
To run all the files successfully, the following packages are required.
Matplotlib,OpenCV2,Numpy,Pandas,Dlib,Sklearn,Pickle,Tensorflow

## Note
If any model is not found, please find it in the link below.

https://drive.google.com/drive/folders/1TRkpczhoyv6o_2fqD16E_05OBX1MrM0k?usp=share_link

Please add the model to the corresponding folder after downloading, so that the code can run successfully.
