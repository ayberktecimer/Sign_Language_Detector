import os

import cv2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle


def reduceWithPCA(features, n):
	pca = PCA(n)
	return pca.fit_transform(features)


def trainImages():
	"""
	Train an LDA model with images in "../samples" folder
	Finally, save the model to file
	:return:
	"""
	# Create training features and labels
	trainFeatures = []
	trainLabels = []
	for imageName in os.listdir("../samples/train"):
		imageLabel = imageName[0]
		trainFeatures.append(cv2.imread("../samples/train/" + imageName, 0).ravel())
		trainLabels.append(imageLabel)
		print(imageName, imageLabel)

	# trainFeatures = reduceWithPCA(trainFeatures, 25) # TODO: to enable PCA, uncomment this line

	# Initialize LDA model and train
	model = LinearDiscriminantAnalysis()
	model.fit(trainFeatures, trainLabels)
	# LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False,
	# 						   tol=0.0001)

	with open('model.obj', 'wb') as fp:
		pickle.dump(model, fp)
