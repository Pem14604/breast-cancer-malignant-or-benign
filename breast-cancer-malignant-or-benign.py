import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 30].values



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Fitting Logistic Regression
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X_train,y_train)
y_predLR = LR.predict(X_test)


#Fitting Random Forest
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators = 100, criterion="entropy", random_state=0)
RF.fit(X_train, y_train)
y_predRF = RF.predict(X_test)
# Part 2 - Now let's make the ANN!


#Fitting ANN
import keras
from keras.models import Sequential
from keras.layers import Dense


ANN = Sequential()

ANN.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu', input_dim = 30))

ANN.add(Dense(output_dim = 14, init = 'uniform', activation = 'relu'))

ANN.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

ANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ANN.fit(X_train, y_train, batch_size = 10, nb_epoch = 20)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm1 = confusion_matrix(y_test, y_predLR)
cm2 = confusion_matrix(y_test, y_predRF)