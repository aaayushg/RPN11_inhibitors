#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing #to normalize the values
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

#Ask for file name and read the file
#file_name = raw_input("Name of file:")
#file_name = 'out_features_docking_pose'
file_name = 'out_features200'
#file_name = 'descriptors'
#data = pd.read_excel(file_name + '.xlsx', header=0)
data = pd.read_csv(file_name + '.csv', header=0)
#analysis_type = input("Analysis Type 'R' or 'C': ")

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

#Defining X
#X_raw = data.drop(['StayLeft','Lig'], axis=1)
X_raw = data.drop(['Zn','StayLeft','ligand','LigRMSD','halogen','Hydrogen','Hydrophobic','pi'], axis=1)
#X_raw = data.drop(['StayLeft','ligand','LigRMSD'], axis=1)
#Normalizing or not the data
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X_raw)
#X = X_raw
print(X)

#Defining Y variables depending on whether we have a regression or classification problem
Y = data.StayLeft

#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)

#Fit the neural network for Regression purposes (i.e., you expect a continuous variable out)
#Note that 'sgd' and 'adam' require a batch_size and the function is not as clear
acti = ['logistic', 'tanh', 'relu', 'identity']
algo = ['lbfgs', 'sgd', 'adam']
learn = ['constant', 'invscaling', 'adaptive']
neural = MLPClassifier(activation=acti[2],
                       solver=algo[1],
                       learning_rate = learn[2], 
                       hidden_layer_sizes=(20,),
                       random_state=1,
#                       learning_rate_init=0.0001,
#                       warm_start=True,
#                       alpha=0.001,
#                       beta_1=0.5,
#                       epsilon=1e-10,
#                       n_jobs=-1,
                       max_iter=100000) 

#Cross validation
neural_scores = cross_val_score(neural, X_train, Y_train, cv=5,n_jobs=-1)
scores = cross_val_score(neural, X, Y, cv=5,n_jobs=-1)
print(neural_scores)
print(scores)
print("Accuracy: {0} (+/- {1})".format(scores.mean().round(2), (scores.std() * 2).round(2)))
#Fitting final neural network
neural.fit(X_train, Y_train)
neural_score = neural.score(X_test, Y_test)
print(neural_score)
