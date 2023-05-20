# leather detector

# leather holes recognizer

from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D , Flatten , Dense
import os
import keras
import tensorflow as tf
from tqdm import tqdm
import cv2

# classifier creation

classifier = Sequential()

# convolution and pooling

classifier.add(Conv2D(32,(3,3),input_shape = (64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

# neural network

classifier.add(Dense(512,activation = 'relu'))
classifier.add(Dense(128,activation = 'relu'))
classifier.add(Dense(2,activation = 'softmax'))

print(classifier.summary())

# compiling the network

classifier.compile(optimizer = "adam",loss = "binary_crossentropy",metrics = ["accuracy"])

# creating a dataset

# directory

DIR = "/home/emman/Desktop/leather projects/surace area calculator/leather_train_mini"
x =[]
y =[]

leather = "/home/emman/Desktop/leather projects/surace area calculator/leather_train_mini/leather"
not_a_leather = "/home/emman/Desktop/leather projects/surace area calculator/leather_train_mini/not_a_leather"

def assign_type(img ,color):
    return assign_type

def make_training_data(color , DIR):
    for img in tqdm(os.listdir(DIR)):
        path = os.path.join(DIR , img)
        label = assign_type(img , color)
        img = cv2.imread(path , cv2.IMREAD_COLOR)
        img = cv2.resize(img , (64,64))

        x.append(img)
        y.append(str(label))

# making the dataset

make_training_data("leather",leather)
print(len(x))

make_training_data("not a leather" ,not_a_leather)
print(len(x))

# label encoding

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y,2)

# making the image pixels between 0 and 1

import numpy as np

x = np.array(x)
x = x/255

import matplotlib.pyplot as plt

print(x[1].shape)
plt.imshow(x[1])
plt.colorbar()
plt.show()

# segregating the data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

print("shape info..!!!")
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)

# fitting the data

classifier.fit(x_train,y_train,epochs = 1,steps_per_epoch = 120,validation_data = (x_test,y_test))

# predictions

result = classifier.evaluate(x_test,y_test)
print("model's result: loss is {0}% and accuracy is {1}%".format(result[0]*100,result[1]*100))

# pred




