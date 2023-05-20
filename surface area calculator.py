import tkinter as tk
import cv2
from PIL import Image, ImageTk

from scipy.integrate import *
import numpy as np
from tkinter import messagebox

class CameraApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Camera App")

        # Create a label to display the camera feed
        self.label = tk.Label(window)
        self.label.pack()

        # Create buttons for different methods
        trap_btn = tk.Button(window, text="Try Trapezoidal Rule", command=lambda: self.calculate_method("trapezoidal"))
        simp_btn = tk.Button(window, text="Try Simpson's Rule", command=lambda: self.calculate_method("simpson"))
        bound_btn = tk.Button(window, text="Try Boundary Approximation", command=lambda: self.calculate_method("boundary"))
        picks_btn = tk.Button(window, text="Try Pick's Method", command=lambda: self.calculate_method("picks"))

        trap_btn.pack()
        simp_btn.pack()
        bound_btn.pack()
        picks_btn.pack()

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
        self.window.after(10, self.capture_video)

    def calculate_method(self, method):
        # Read a frame from the camera
        _, frame = self.camera.read()

        # Process the frame using the selected method
        if method == "trapezoidal":
            # Perform trapezoidal rule calculation
            result = self.trapezoidal_rule(frame)
        elif method == "simpson":
            # Perform Simpson's rule calculation
            result = self.simpsons_rule(frame)
        elif method == "boundary":
            # Perform boundary approximation calculation
            result = self.boundary_approximation(frame)
        elif method == "picks":
            # Perform Pick's method calculation
            result = self.picks_method(frame)
        else:
            result = None

        # Display the result in a message box
        if result is not None:
            tk.messagebox.showinfo("Calculation Result", f"The result using {method} is: {result}")
        else:
            tk.messagebox.showwarning("Invalid Method", "Invalid method selected!")

    def trapezoidal_rule(self, frame):
        # Perform trapezoidal rule calculation here
        # Replace this placeholder code with your implementation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        surface_area = trapz(y_coords, x_coords)

        print("Surface Area:", surface_area)


    def simpsons_rule(self, frame):
        # Perform Simpson's rule calculation here
        # Replace this placeholder code with your implementation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        surface_area = simps(y_coords, x_coords, even='first')

        print("Surface Area:", surface_area)

    def boundary_approximation(self, frame):
        # Perform boundary approximation calculation here
        # Replace this placeholder code with your implementation

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        surface_area = abs(signed_area) / 2

        print("Surface Area:", surface_area)

    def picks_method(self, frame):
        # Perform Pick's method calculation here
        # Replace this placeholder code with your implementation

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        surface_area = cv2.contourArea(approximated_contour)

        print("Surface Area:", surface_area)

if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()

    # Create an instance of the CameraApp class
    app = CameraApp(root)

    # Start the main event loop
    root.mainloop()
