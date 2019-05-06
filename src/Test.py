import os
import pickle

import cv2

# Load model from saved file
with open('../generatedModels/PCA.obj', 'rb') as fp:
	pca = pickle.load(fp)


def reduceWithPCA(features):
	return pca.transform(features)


def predict(features, algorithm):
	reducedFeatures = reduceWithPCA(features)
	if algorithm == "SVM":
		with open('../generatedModels/modelSVM.obj', 'rb') as fp:
			modelSVM = pickle.load(fp)
		return modelSVM.predict(reducedFeatures)
	elif algorithm == "LDA":
		with open('../generatedModels/modelLDA.obj', 'rb') as fp:
			model = pickle.load(fp)
		return model.predict(reducedFeatures)
	else:
		raise Exception("Algorithm name is wrong!")


def testLDA():
	with open('../generatedModels/modelLDA.obj', 'rb') as fp:
		model = pickle.load(fp)
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
	with open('../generatedModels/modelSVM.obj', 'rb') as fp:
		modelSVM = pickle.load(fp)

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
