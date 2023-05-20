import cv2
import numpy as np

# Step 1: Load the positive and negative images
positive_images = '/home/emman/Desktop/leather projects/stitches and holes counter/stitches_dataset'
negative_images = '/home/emman/Desktop/leather projects/stitches and holes counter/stitches_dataset'

# Step 2: Create the positive samples description file
positive_samples_file = 'positive_samples.txt'
with open(positive_samples_file, 'w') as f:
    for image_path in positive_images:
        f.write(f'{image_path} 1 0 0 100 100\n')

# Step 3: Create the negative samples description file
negative_samples_file = 'negative_samples.txt'
with open(negative_samples_file, 'w') as f:
    for image_path in negative_images:
        f.write(f'{image_path}\n')

# Step 4: Create the positive samples vector file
positive_samples_vector = 'positive_samples.vec'
positive_samples_cmd = f'opencv_createsamples -info {positive_samples_file} -vec {positive_samples_vector} -w 100 -h 100'

import os
os.system(positive_samples_cmd)

# Step 5: Train the cascade classifier
cascade_output = '/home/emman/Desktop/leather projects/stitches and holes counter'
cascade_xml_file = 'stitches_cascade.xml'
cascade_cmd = f'opencv_traincascade -data {cascade_output} -vec {positive_samples_vector} -bg {negative_samples_file} -numPos 2000 -numNeg 1000 -numStages 20 -w 100 -h 100 -featureType HAAR -mode ALL -precalcValBufSize 1024 -precalcIdxBufSize 1024 -minHitRate 0.995 -maxFalseAlarmRate 0.5 -numThreads 4 -acceptanceRatioBreakValue 10e-5 -bt GAB -maxDepth 1'
os.system(cascade_cmd)

# Step 6: Load and test the custom cascade classifier
cascade = cv2.CascadeClassifier(cascade_xml_file)

test_image = cv2.imread('/home/emman/Desktop/leather projects/stitches and holes counter/stitches_dataset/image_101d.jpg')
gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in objects:
    cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Object Detection', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
