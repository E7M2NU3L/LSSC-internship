# color recognizer

# step-1 building the model

import keras as ke
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense , Conv2D , Flatten , MaxPooling2D

classifier = Sequential()

classifier.add(Conv2D(filters = 32,input_shape = (128,128,3),kernel_size = (3,3)))
classifier.add(MaxPooling2D(pool_size = (2,2),strides = (1,1)))

classifier.add(Conv2D(filters = 64,kernel_size = (3,3)))
classifier.add(MaxPooling2D(pool_size = (2,2),strides = (1,1)))

classifier.add(Conv2D(filters = 128,kernel_size = (3,3)))
classifier.add(MaxPooling2D(pool_size = (2,2),strides = (1,1)))

classifier.add(Conv2D(filters = 256,kernel_size = (3,3)))
classifier.add(MaxPooling2D(pool_size = (2,2),strides = (1,1)))

classifier.add(Flatten())

classifier.add(Dense(units = 512,activation = "relu"))
classifier.add(Dense(128 , activation = "relu"))
classifier.add(Dense(9,activation = "softmax"))

print(classifier.summary())

# step-2 creating a dataset

import cv2
import os
import tqdm
from tqdm import tqdm

DIR = "/home/emman/color_dataset"
x =[]
y =[]

black = "/home/emman/color_dataset/Black"
blue = "/home/emman/color_dataset/Blue"
green = "/home/emman/color_dataset/Green"
brown = "/home/emman/color_dataset/Brown"
violet = "/home/emman/color_dataset/Violet"
white = "/home/emman/color_dataset/White"
orange = "/home/emman/color_dataset/orange"
red = "/home/emman/color_dataset/red"
yellow = "/home/emman/color_dataset/yellow"


def assign_type(img ,color):
    return assign_type

def make_training_data(color , DIR):
    for img in tqdm(os.listdir(DIR)):
        path = os.path.join(DIR , img)
        label = assign_type(img , color)
        img = cv2.imread(path , cv2.IMREAD_COLOR)
        img = cv2.resize(img , (128,128))

        x.append(img)
        y.append(str(label))

# making the dataset

make_training_data("Black",black)
print(len(x))

make_training_data("Blue" ,blue)
print(len(x))

make_training_data("Green",green)
print(len(x))

make_training_data("Orange",orange)
print(len(x))

make_training_data("Brown",brown)
print(len(x))

make_training_data("Red",red)
print(len(x))

make_training_data("Violet",violet)
print(len(x))

make_training_data("White",white)
print(len(x))

make_training_data("Yellow",yellow)
print(len(x))

# Step-3 Preprocessing the data

from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.utils import to_categorical

x = np.array(x)
x = x/255
print(x[0])

le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y,9)

# Step-4 segregating the data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

print(x_train.shape)
print(y_train.shape)

# step-5 compiling evluating and fitting the model

batch_size = 64
from keras.optimizers import Adam

classifier.compile(optimizer = Adam(learning_rate = 0.001) ,loss = "categorical_crossentropy",metrics = ['accuracy'])
classifier.fit(x_train,y_train,epochs = 5,validation_data = (x_test,y_test),steps_per_epoch = 20)

result = classiier.evaluate(x_test,y_test)

print("model's quality: {0}".format(result))
