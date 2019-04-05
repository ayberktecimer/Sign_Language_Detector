import os

import cv2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
import pickle

TRAIN_SET_DIRECTORY = "../samples/train/"


def reduceWithPCA(features, n):
	pca = PCA(n)
	return pca.fit_transform(features)


def trainLDA():
	"""
	Train an LDA model with images in "../samples" folder
	Finally, save the model to file
	:return:
	"""
	# Create training features and labels
	trainFeatures = []
	trainLabels = []
	for imageName in os.listdir(TRAIN_SET_DIRECTORY):
		imageLabel = imageName[0]
		trainFeatures.append(cv2.imread(TRAIN_SET_DIRECTORY + imageName, 0).ravel())
		trainLabels.append(imageLabel)

	# trainFeatures = reduceWithPCA(trainFeatures, 25) # TODO: to enable PCA, uncomment this line

	# Initialize LDA model and train
	model = LinearDiscriminantAnalysis()
	model.fit(trainFeatures, trainLabels)

	# Save model to file
	with open('modelLDA.obj', 'wb') as fp:
		pickle.dump(model, fp)

	print("LDA training completed and saved to file.")


def trainSVN():
	"""
	Train an SVN model with images in "../samples" folder
	Finally, save the model to file
	:return:
	"""
	trainFeatures = []
	trainLabels = []
	for imageName in os.listdir(TRAIN_SET_DIRECTORY):
		imageLabel = imageName[0]
		trainFeatures.append(cv2.imread(TRAIN_SET_DIRECTORY + imageName, 0).ravel())
		trainLabels.append(imageLabel)

	# trainFeatures = reduceWithPCA(trainFeatures, 25) # TODO: to enable PCA, uncomment this line

	# Initialize SVN model and train
	model = svm.SVC(gamma='scale')
	model.fit(trainFeatures, trainLabels)

	# Save model to file
	with open('modelSVN.obj', 'wb') as fp:
		pickle.dump(model, fp)

	print("SVN training completed and saved to file.")
