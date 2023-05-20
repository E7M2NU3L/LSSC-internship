# color recognizer

import pandas as pd
import numpy as np
import sklearn
import keras
import tensorflow as tf

from tkinter import *
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split


index = ["color", "color_name", "hex", "R", "G", "B"]
color = pd.read_csv('colors.csv',names = index)
print(color.head(5))

x = color.iloc[:,3:6]
print(x.head())

y = color.iloc[:,0]
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)
print(x_train.shape)
print(x_test.shape)

# creating the model

classifier = Sequential()

classifier.add(Dense(512,input_dim = 3,activation = "relu"))
classifier.add(Dense(256,activation = "relu"))

classifier.add(Dense(865,activation = "softmax"))
print(classifier.summary())

# preprocessing the data

from keras.optimizers import Adam

classifier.compile(loss = "categorical_crossentropy",optimizer = Adam(learning_rate = 0.01),metrics = ['accuracy'])
classifier.fit(x_train,y_train,epochs = 500,batch_size = 173)

# compiling the model

_,accuracy = classifier.evaluate(x_train,y_train)
print("accuracy of the model is {0}".format(accuracy*100))

