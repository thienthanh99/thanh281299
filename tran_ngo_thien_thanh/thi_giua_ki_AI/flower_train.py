#-----------------------------------
# GLOBAL FEATURE EXTRACTION
#-----------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import mahotas
import cv2
import os
import glob
from sklearn.svm import SVC
import matplotlib.pyplot as plt
# import h5py

#--------------------
# tunable-parameters
#--------------------
images_per_class = 9
fixed_size       = tuple((500, 500))
train_path       = "dataset/train"
bins             = 8


# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()

# empty lists to hold feature vectors and labels
global_features = []
labels          = []

def train_file():
# loop over the training data sub-folders
    i = 1
    for training_name in train_labels:
        # join the training data path and each species training folder
        dir = os.path.join(train_path, training_name)

        # get the current training label
        current_label = i

    # loop over the images in each sub-folder
        for file in glob.glob(os.path.join(dir, '*.jpg')):
            # get the image file name
            # file = dir + "/image_000" + str(x) + ".jpg"
            print(file)

            # read the image and resize it to a fixed-size
            image = cv2.imread(file)
            image = cv2.resize(image, fixed_size)

            ####################################
            # Global Feature extraction
            ####################################
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick   = fd_haralick(image)
            fv_histogram  = fd_histogram(image)

            ###################################
            # Concatenate global features
            ###################################
            global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

            # update the list of labels and feature vectors
            labels.append(current_label)
            global_features.append(global_feature)

            print(labels)
        i = i + 1
    np.savetxt('feature-datas.txt', global_features)
    np.savetxt('label-datas.txt', labels)
    print("[STATUS] processed folder: {}".format(current_label))

train_file()