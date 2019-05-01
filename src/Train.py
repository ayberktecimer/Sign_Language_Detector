import os

import cv2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
import pickle
import matplotlib.pyplot as plt
import numpy as np

TRAIN_SET_DIRECTORY = "../samples/train/"


def reduceWithPCA(features, n):
	pca = PCA(n)
	pca.fit(features)
	# Save fitted PCA model to a file
	with open('../generatedModels/PCA.obj', 'wb') as fp:
		pickle.dump(pca, fp)

	# Plot something I don't know
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.xlabel("Number of components")
	plt.ylabel("Cumulative explained variance")

	# Transform features
	return pca.transform(features)


def trainLDA(parameters):
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

	trainFeatures = reduceWithPCA(trainFeatures, 50)

	# Initialize LDA model and train
	model = LinearDiscriminantAnalysis(**parameters)
	model.fit(trainFeatures, trainLabels)

	# Save model to file
	with open('../generatedModels/modelLDA.obj', 'wb') as fp:
		pickle.dump(model, fp)

	print("LDA training completed and saved to file.")


def trainSVM():
	"""
	Train an SVM model with images in "../samples" folder
	Finally, save the model to file
	:return:
	"""
	trainFeatures = []
	trainLabels = []
	for imageName in os.listdir(TRAIN_SET_DIRECTORY):
		imageLabel = imageName[0]
		trainFeatures.append(cv2.imread(TRAIN_SET_DIRECTORY + imageName, 0).ravel())
		trainLabels.append(imageLabel)

	trainFeatures = reduceWithPCA(trainFeatures, 50)  # TODO: to enable PCA, uncomment this line

	# Initialize SVM model and train
	model = svm.SVC(gamma='scale')
	model.fit(trainFeatures, trainLabels)

	# Save model to file
	with open('../generatedModels/modelSVM.obj', 'wb') as fp:
		pickle.dump(model, fp)

	print("SVM training completed and saved to file.")
