#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn import preprocessing #to normalize the values
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
import random as rn
import os
from os import sys
np.set_printoptions(threshold=sys.maxsize)

#Random seeding
os.environ['PYTHONHASHSEED']='0'
np.random.seed(123)
rn.seed(123)
tf.random.set_seed(1)

#file_name = 'out_features_docking_pose'
file_name = 'all2'
data = pd.read_csv(file_name + '.csv', header=0)
data=data.dropna()

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

#Defining X
X_raw = data.drop(['StayLeft','ligand'], axis=1)

#Normalizing or not the data
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X_raw)

#Defining Y variables depending on whether we have a regression or classification problem
Y = data.StayLeft
X_lig = data.ligand

#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test, lig_train, lig_test = train_test_split(X, Y, X_lig, train_size = 0.60, random_state=1)
df_index= data[['StayLeft','ligand']]

model = Sequential([    
    Dense(1000, activation='relu', input_shape=(5284,), kernel_regularizer=regularizers.l2(0.01)), Dropout(0.3),
    Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.01)), Dropout(0.3),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)), Dropout(0.3),
    Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.01)), Dropout(0.3),
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),])

model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

hist = model.fit(X_train, Y_train,          
        batch_size=32, epochs=100,          
        validation_data=(X_test, Y_test))

pred=model.predict_classes(X_test)
prob=model.predict(X_test)
print(np.column_stack((lig_test,Y_test,pred,prob)))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

print("Training Accuracy: "   + str(model.evaluate(X_train,Y_train)[1]))
print("Validation Accuracy: " + str(model.evaluate(X_test,Y_test)[1]))


plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

