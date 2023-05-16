# color_classification_for_leathers_tains

# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from tqdm import tqdm
import pandas as pd
import tensorflow as tk
import keras as ke
import sklearn
import cv2
import PIL

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D , Flatten , AveragePooling2D ,Dense
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# importing the datasets

# directory

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
        img = cv2.resize(img , (300,150))

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


# storing the data

# categorizing the label

le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y,6)

# making the image pixels between 0 and 1

x = np.array(x)
x = x/255

# seperating the data

x_train,x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

print(x_train.shape,x_test.shape)

# model training

model = Sequential()

# first stage
model.add(Conv2D(filters = 32 , kernel_size = (3,3)))
model.add(MaxPooling2D(pool_size = (2,2) , strides = (1,1)))

# secoind stage
model.add(Conv2D(filters = 64 , kernel_size = (3,3)))
model.add(AveragePooling2D(pool_size = (2,2) , strides = (1,1)))

# third stage
model.add(Conv2D(filters = 64 , kernel_size = (3,3)))
model.add(MaxPooling2D(pool_size = (2,2) , strides = (1,1)))

# fourth stage
model.add(Conv2D(filters = 128 , kernel_size = (3,3)))
model.add(MaxPooling2D(pool_size = (2,2) , strides = (1,1)))

# flattening layer

model.add(Flatten())

# input layer

model.add(Dense(6,activation = "relu",input_shape = (300,150)))

# hidden layers

model.add(Dense(12,activation = "relu"))

# output layer

model.add(Dense(6 , activation = "softmax"))

history = model.compile(loss = "categorical_crossentropy" , optimizer = Adam(lr = 0.001) , metrics = ['accuracy'])

print(history)

from keras.callbacks import ReduceLROnPlateau
red_lr = ReduceLROnPlateau(monitor = 'val_acc',patience = 3,factor = 0.1,verbode = 1)

datagen = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 10,
    zoom_range = 0.1,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True)

batch_size = 128
epochs = 10

model.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = epochs,validation_data = (x_test,y_test),verbose = 1,
                    steps_per_epoch = x_train.shape[0] // batch_size)

result = model.evaluate(x_test,y_test,batch_size = 128)
print("loss and accuracy: ",result)

prediction = model.predict(x_test)
print(prediction[2])

plt.figure()
plt.imshow(x_test[2])
plt.colormap()
plt.show()
