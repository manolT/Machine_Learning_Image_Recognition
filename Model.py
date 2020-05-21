import numpy as np
import matplotlib.pyplot as plt
import skimage.feature
from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.python.client import device_lib
#loading data
trainImages = np.load('trnImage.npy')
trainLabels = np.load('trnLabel.npy')
testImages = np.load('tstImage.npy')
testLabels = np.load('tstLabel.npy')
def computeFeatures(image):
    # This function computes the HOG features with the parsed hyperparameters and returns the features as hog_feature. 
    # By setting visualize=True we obtain an image, hog_as_image, which can be plotted for insight into extracted HOG features.
    hog_feature, hog_as_image = skimage.feature.hog(image, visualize=True, block_norm='L2-Hys')
    return hog_feature, hog_as_image
	
#extracting features

features,wild_boar = computeFeatures(trainImages[:,:,:,0])

#extracting features from training set
trainData = np.zeros((trainImages.shape[3], features.shape[0]))
for i in range(trainImages.shape[3]):
    trainData[i,:], hog_image = computeFeatures(trainImages[:,:,:,i])

#extracting features from testing set
testData = np.zeros((testImages.shape[3], features.shape[0]))
for i in range(testImages.shape[3]):
    testData[i,:], hog_image = computeFeatures(testImages[:,:,:,i])
	
# Normalising the data based on the training set
normaliser = StandardScaler().fit(trainData)
trainData = normaliser.transform(trainData)
testData = normaliser.transform(testData)

#predicting the labels with the SVM
predictedLabels = model.predict(testData)


accuracy = metrics.accuracy_score(testLabels, predictedLabels)*100
print('Percentage accuracy on testing set is:', accuracy,'%')
print("Precision report:")
print(metrics.classification_report(testLabels, predictedLabels))
print(metrics.confusion_matrix(testLabels, predictedLabels))

