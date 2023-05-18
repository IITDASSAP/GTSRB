import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator

signalsBasePath = "TrafficSignals"
labelsPath = 'labels.csv'
signalsTrainingPath = 'Train'
testPath = 'Test'
IMAGE_HEIGHT = 38
IMAGE_WIDTH = 38
CHANNELS = 3  # Simply means the colors are of 3 channels Red,Blue and  Green
batchingSize = 50
stepsPerEpochValue = 2000
epochValue = 5
imageDimensions = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
testRatio = .2
validationRatio = .2

allImages = []
classNumber = []
print(os.listdir())
classList = os.listdir(signalsBasePath + "/" + signalsTrainingPath)
numberOfClasses = len(classList)
print("The Number Of training classes detected are:", numberOfClasses)

for trainingDataFolderIndex in range(0, len(classList)):
    imagesInEachClass = os.listdir(
        signalsBasePath + "/" + signalsTrainingPath + "/" + str(trainingDataFolderIndex))
    for eachImage in imagesInEachClass:
        # print(signalsBasePath + "/" + signalsTrainingPath + "/" + str(trainingDataFolderIndex) + "/" + eachImage)
        currentImage = cv2.imread(
            signalsBasePath + "/" + signalsTrainingPath + "/" + str(trainingDataFolderIndex) + "/" + eachImage)
        imageFromArray = Image.fromarray(currentImage, "RGB")
        resizedImage = imageFromArray.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
        allImages.append(np.array(resizedImage))
        classNumber.append(trainingDataFolderIndex)
# print(len(allImages))
allImages = np.array(allImages)
classNumber = np.array(classNumber)

rearrangeIndies = np.arange(allImages.shape[0])
np.random.shuffle(rearrangeIndies)
allImages = allImages[rearrangeIndies]
classNumber = classNumber[rearrangeIndies]
X_train_data, X_test_data, Y_train_data, Y_test_data = train_test_split(allImages, classNumber, test_size=testRatio)
X_train_data, X_validation_data, Y_train_data, Y_validation_data = train_test_split(X_train_data, Y_train_data,
                                                                                    test_size=validationRatio)
## Graph related data still in progress
