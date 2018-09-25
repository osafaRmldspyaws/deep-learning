#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 08:35:59 2018

@author: osafakarim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load dataset
data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:,3:13].values
y = data.iloc[:,13].values

# preprocessing steps
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_X_1 = LabelEncoder()
X[:,1] = label_X_1.fit_transform(X[:,1])

label_X_2 = LabelEncoder()
X[:,2] = label_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,
                                                    random_state=123)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing libraries for ann
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#intializing the ann
classifier = Sequential()

#adding the input layer and first hidden layer with dropout
classifier.add(Dense(6,input_shape=(11,),
               kernel_initializer='uniform',
               activation='relu')
               )
classifier.add(Dropout(1))

#adding the second hidden layer
classifier.add(Dense(6,
               kernel_initializer='uniform',
               activation='relu')
               )
classifier.add(Dropout(1))

#adding the output layer

classifier.add(Dense(1,
               kernel_initializer='uniform',
               activation='sigmoid')
               )


#ccompile the ann
classifier.compile(optimizer='adam',loss='binary_crossentropy',
                   metrics=['accuracy'])

#fitting the ann to training dataset
classifier.fit(X_train,y_train,
               batch_size=10,
               epochs=100)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

# checking for metrics 
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test,y_pred)


# homework challenge

#prepare the data-set

#df = {'CreditScore':[600], 
#      'Geography':['France'],
#      'Gender':['Male'], 
#      'Age':[40], 
#      'Tenure':[3], 
#      'Balance':[60000], 
#      'NumOfProducts':[2],
#      'HasCrCard':[1],
#      'IsActiveMember':[1],
#      'EstimatedSalary':[50000]
#      }
#df_cust = pd.DataFrame(df)
#
#df_cust = pd.DataFrame(df,
#                       index = 1)

# model evaluation using k-fold cross validation

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dropout

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6,input_shape=(11,),
               kernel_initializer='uniform',
               activation='relu')
               )
    classifier.add(Dense(6,
                   kernel_initializer='uniform',
                   activation='relu')
                   )
    classifier.add(Dense(1,
               kernel_initializer='uniform',
               activation='sigmoid')
               )
    classifier.compile(optimizer='adam',loss='binary_crossentropy',
                   metrics=['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn=build_classifier,batch_size=10,
                             nb_epoch=100)
accuracies = cross_val_score(estimator = classifier , X = X_train , y = y_train,
                             cv = 10 , n_jobs = -1)

# Tuning the hyper parameters of a model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6,input_shape=(11,),
               kernel_initializer='uniform',
               activation='relu')
               )
    classifier.add(Dense(6,
                   kernel_initializer='uniform',
                   activation='relu')
                   )
    classifier.add(Dense(1,
               kernel_initializer='uniform',
               activation='sigmoid')
               )
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',
                   metrics=['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
    
parameters = {
            'batch_size':[25,32],
            'nb_epoch':[100,500],
            'optimizer':['adam','rmsprop']
            }

grid_serach = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring= 'accuracy',
                           cv = 10)

grid_serach = grid_serach.fit(X_train,y_train)

best_parameters =  grid_serach.best_params_
best_accuracy = grid_serach.best_score_


