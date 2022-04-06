import os
from PIL import Image
import numpy as np
import cv2
def generate_dataset(img, img_id):
    # write image in data dir
    cv2.imwrite("data_to_crop/img_" +str(img_id)+".jpg", img)

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    gray_img = np.array(gray_img, dtype='uint8')
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        coords = [x, y, w, h]
    return coords


# Method to detect the features
def detect_noses(img, faceCascade, img_id):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    # If feature is detected, the draw_boundary method will return the x,y coordinates and width and height of rectangle else the length of coords will be 0
    if len(coords)==4:
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        # Assign unique id to each user
        generate_dataset(roi_img, img_id)


# Loading classifiers
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('Nariz.xml')
mouthCascade = cv2.CascadeClassifier('Mouth.xml')

# Method to train custom classifier to recognize face
def feature_crop(data_dir):
    # Read all the images in custom data-set
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    id = 0

    # Store images in a numpy format and ids of the user on the same index in imageNp and id lists
    for image in path:
        img = Image.open(image)
        imageNp = np.array(img, 'uint8')
        detect_noses(imageNp, eyesCascade, id)
        id += 1

feature_crop("data")