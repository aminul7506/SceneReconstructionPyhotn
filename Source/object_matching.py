import numpy as np
import cv2

# We point OpenCV's CascadeClassifier function to where our
# classifier (XML file format) is stored, remember to keep the code and classifier in the same folder
face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load our image then convert it to grayscale
image = cv2.imread('../InputData/original_image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Our classifier returns the ROI of the detected face as a tuple
# It stores the top left coordinate and the bottom right coordinates
# it returns the list of lists, which are the location of different faces detected.
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# When no faces detected, face_classifier returns and empty tuple
if faces is ():
    print("No faces found")

# We iterate through our faces array and draw a rectangle
# over each face in faces

for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2)
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()