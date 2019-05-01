import os
import pickle

import cv2
from sklearn.metrics import accuracy_score

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

	# Test and calculate accuracy
	predictedLabels = model.predict(testFeatures)
	print("LDA Accuracy: ", accuracy_score(testLabels, predictedLabels))

	return testLabels, predictedLabels


def testSVM():
	# Create test features and labels
	testFeatures = []
	testLabels = []
	for imageName in os.listdir("../samples/test"):
		imageLabel = imageName[0]
		testFeatures.append(cv2.imread("../samples/test/" + imageName, 0).ravel())
		testLabels.append(imageLabel)

	testFeatures = reduceWithPCA(testFeatures)  # TODO: to enable PCA, uncomment this line

	# Test and calculate accuracy
	predictedLabels = modelSVM.predict(testFeatures)
	print("SVM Accuracy: ", accuracy_score(testLabels, predictedLabels))

	return testLabels, predictedLabels
