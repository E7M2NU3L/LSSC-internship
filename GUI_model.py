# creating the final leather gradient analyser

from tkinter import *
import cv2
from tkinter import messagebox

root = Tk()
root.geometry('700x700')
root.title('leather gradient analyser')
root.configure(bg = "#FF7F50")

def capture_image():
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Capture a frame from the camera
    ret, frame = cap.read()

    # Release the camera
    cap.release()

    # Save the captured frame to a specific directory
    save_path = "/home/emman/Desktop/leather projects/gradient_analyser/image_folder/Image.jpg"
    cv2.imwrite(save_path, frame)

    # Display a message box to indicate successful capture and save
    messagebox.showinfo("Capture Success", "Image captured and saved successfully!")

def color_finder():

    import pandas as pd
    import numpy as np
    from keras.models import model_from_json

    json_file = open("classifier.json",'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights("colors.h5")
    print("loaded model from the dir")

    image = cv2.imread(r'/home/emman/Desktop/leather projects/gradient_analyser/image_folder/Image.jpg')
    image = cv2.resize(image,(16,16))


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

# Create the capture button
capture_button = Button(root, text="Capture", command=capture_image)
capture_button.pack()

# getting the color
color_getter_button = Button(root,text = "get color",command = color_finder).pack()
root.mainloop()
