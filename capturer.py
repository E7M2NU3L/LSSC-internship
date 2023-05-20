# capturer

import tkinter as tk
import cv2
from PIL import Image, ImageTk
from tkinter import messagebox

from scipy.integrate import *
import numpy as np
from tkinter import *

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera App")

        # Create a label to display the camera feed
        self.label = tk.Label(root)
        self.label.pack()

        # Create a button to capture an image
        capture_btn = tk.Button(root, text="Capture Image", command=self.capture_image)
        capture_btn.pack()

        # Initialize the camera
        self.camera = cv2.VideoCapture(0)

        # Start capturing the video feed
        self.capture_video()

    def capture_video(self):
        # Read a frame from the camera
        _, frame = self.camera.read()

        # Convert the frame from OpenCV BGR format to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame to fit the label
        resized_frame = cv2.resize(rgb_frame, (640, 480))

        # Convert the frame to an ImageTk format
        image = Image.fromarray(resized_frame)
        image_tk = ImageTk.PhotoImage(image)

        # Update the label with the new frame
        self.label.config(image=image_tk)
        self.label.image = image_tk

        # Schedule the next frame capture
        self.root.after(10, self.capture_video)

    def capture_image(self):
        # Read a frame from the camera
        _, frame = self.camera.read()

        # Save the frame as an image file
        cv2.imwrite("captured_image.jpg", frame)

        # Display a message box to indicate successful capture
        tk.messagebox.showinfo("Capture Successful", "Image captured and saved!")
    
    def trap(self,x):
        
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to obtain a binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate the perimeter of the contour using arcLength
        perimeter = cv2.arcLength(largest_contour, True)

        # Get the x and y coordinates of the contour
        x_coords = largest_contour[:, 0, 0]
        y_coords = largest_contour[:, 0, 1]

        # Perform trapezoidal numerical integration to calculate the surface area
        self.surface_area = trapz(y_coords, x_coords)

        print("Surface Area:", surface_area)


    def simpsons(self,x):
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to obtain a binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Sort the contour points based on their x-coordinates
        contour_points = largest_contour.reshape(-1, 2)
        contour_points = contour_points[np.argsort(contour_points[:, 0])]

        # Get the x and y coordinates of the contour
        x_coords = contour_points[:, 0]
        y_coords = contour_points[:, 1]

        # Perform Simpson's 1/3 rule numerical integration to calculate the surface area
        self.surface_area = simps(y_coords, x_coords, even='first')

        print("Surface Area:", surface_area)

    def picks(self,x):

        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to obtain a binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Flatten the contour coordinates
        contour_points = largest_contour.reshape(-1, 2)

        # Compute the signed area using Pick's method
        signed_area = 0
        num_points = len(contour_points)

        for i in range(num_points):
            x1, y1 = contour_points[i]
            x2, y2 = contour_points[(i + 1) % num_points]
            signed_area += (x1 * y2) - (x2 * y1)

        # Calculate the absolute value of the area
        self.surface_area = abs(signed_area) / 2

        print("Surface Area:", surface_area)

    def boundary(self,x):

        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to obtain a binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour with a simpler polygon
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approximated_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Calculate the surface area of the approximated contour
        self.surface_area = cv2.contourArea(approximated_contour)

        print("Surface Area:", surface_area)

    self.simp = Button(root,text = "try simpson's rule",command = lambda:simpsons(img))
    self.bound = Button(root,text = "try boundary approximation",command = lambda:boundary(img))

    self.trap = Button(root,text = "try trapezoidal rule",command =trap(img))
    self.picks = Button(root,text = "try pick's method",command =picks(img))


    trap.pack()
    simp.pack()
    bound.pack()
    picks.pack()

if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()

    # Create an instance of the CameraApp class
    app = CameraApp(root)

    # Start the main event loop
    root.mainloop()
