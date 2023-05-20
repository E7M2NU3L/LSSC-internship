# final

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
from keras.models import model_from_json

json_file = open("classifier.json",'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights("colors.h5")
print("loaded model from the dir")

import cv2
image = cv2.imread(r'/home/emman/Desktop/leather projects/gradient_analyser/image_folder/Image.jpg')
image = cv2.resize(image,(16,16))

cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()


row,col,dim = image.shape
rgb_values = []

for y in range(col):
    for x in range(row):
        r,g,b = image[x,y]
        rgb_values.append((r,g,b))

print(rgb_values[1])

# importing colors dataset
colors = pd.read_csv('colors.csv')

resulting_colors_final = []

for rgb in rgb_values:
    data = pd.DataFrame(rgb)
    data = np.array(data).reshape(1,3)

    result = model.predict(data)

    resulting_color = np.argmax(result)
    color_name = colors.iloc[:,1]
    resulting_colors_final.append(color_name[resulting_color])

for i in resulting_colors_final:
    print(i)
