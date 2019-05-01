import os
import pickle

import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load model from saved file
with open('../generatedModels/modelLDA.obj', 'rb') as fp:
	model = pickle.load(fp)
with open('../generatedModels/modelSVM.obj', 'rb') as fp:
	modelSVM = pickle.load(fp)
with open('../generatedModels/PCA.obj', 'rb') as fp:
	pca = pickle.load(fp)


def reduceWithPCA(features):
	return pca.transform(features)


def predict(features, algorithm):
	reducedFeatures = reduceWithPCA(features)
	if algorithm == "SVM":
		return modelSVM.predict(reducedFeatures)
	elif algorithm == "LDA":
		return model.predict(reducedFeatures)
	else:
		raise Exception("Algorithm name is wrong!")


def testLDA():
	# Create test features and labels
	testFeatures = []
	testLabels = []
	for imageName in os.listdir("../samples/test"):
		imageLabel = imageName[0]
		testFeatures.append(cv2.imread("../samples/test/" + imageName, 0).ravel())
		testLabels.append(imageLabel)

	testFeatures = reduceWithPCA(testFeatures)  # TODO: to enable PCA, uncomment this line

	# Test
	predictedLabels = model.predict(testFeatures)

	return model, testLabels, predictedLabels


def testSVM():
	# Create test features and labels
	testFeatures = []
	testLabels = []
	for imageName in os.listdir("../samples/test"):
		imageLabel = imageName[0]
		testFeatures.append(cv2.imread("../samples/test/" + imageName, 0).ravel())
		testLabels.append(imageLabel)

	testFeatures = reduceWithPCA(testFeatures)  # TODO: to enable PCA, uncomment this line

	# Test
	predictedLabels = modelSVM.predict(testFeatures)

	return modelSVM, testLabels, predictedLabels
