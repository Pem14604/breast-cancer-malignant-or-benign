# breast-cancer-malignant-or-benign

This is the classification problem where we have to find breast cancer type malignant/benign

In this i used 3 algorithms to compare the predictions
        1. Logistic Regression 
        2. Random Fosrest Classifier
        3. Artificial Neural Networks 

Breif steps 

Import Library
Import Dataset using panda
create x matrix and y vector using iloc 
create train and test result using train_test_split from sklear preprocessing 
Perform feature scaling (standardization) using StadardScaler from sklear preprocessing
Apply logistic regression
Apply random forest classifier with criterion "entropy".
Apply ANN
Predict result in each case
create confusion martix 

Accuray for eaach model
 1. Logistic Regression         96.49
 2. Random Fosrest Classifier   97.36
 3. Artificial Neural Networks  95.61

