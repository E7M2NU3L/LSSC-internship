#  Spectrometer project

# step-1 importing the libraries

import cv2
import numpy as np

# step-2 importing the images

foreground_image = cv2.imread("image (1).jpg")
background_image = cv2.imread("image (2).jpg")
print(foreground_image.shape)
print(background_image.shape)

# step-3 preprocessing

fg_gray = cv2.cvtColor(foreground_image , cv2.COLOR_BGR2GRAY)
smooth_fg = cv2.blur(fg_gray , (2,2))
blurred_fg = cv2.GaussianBlur(smooth_fg, (0, 0), 3)
sharpened_fg = cv2.addWeighted(fg_gray, 1.5, blurred_fg, -0.5, 0)
denoised_fg = cv2.fastNlMeansDenoising(sharpened_fg, None, h=10, templateWindowSize=7, searchWindowSize=21)
denoised_bgr_fg = cv2.cvtColor(denoised_fg, cv2.COLOR_GRAY2BGR)

bg_gray = cv2.cvtColor(background_image , cv2.COLOR_BGR2GRAY)
smooth_bg = cv2.blur(bg_gray , (2,2))
blurred_bg = cv2.GaussianBlur(smooth_bg, (0, 0), 3)
sharpened_bg = cv2.addWeighted(bg_gray, 1.5, blurred_bg, -0.5, 0)
denoised_bg = cv2.fastNlMeansDenoising(sharpened_bg, None, h=10, templateWindowSize=7, searchWindowSize=21)
denoised_bgr_bg = cv2.cvtColor(denoised_bg, cv2.COLOR_GRAY2BGR)

# Step-4 determining the ROI

Roi = denoised_bgr_fg - denoised_bgr_bg

# Step-5 spectral analysis

# Convert ROI to grayscale
roi_gray = cv2.cvtColor(Roi, cv2.COLOR_BGR2GRAY)

# Calculate the average spectrum across the ROI
spectrum = np.mean(roi_gray, axis=0)

# Plot the spectrum
import matplotlib.pyplot as plt

wavelengths = np.arange(len(spectrum))
plt.plot(wavelengths, spectrum)
plt.xlabel("Wavelength")
plt.ylabel("Intensity")
plt.title("Electromagnetic Spectrum")
plt.show()

