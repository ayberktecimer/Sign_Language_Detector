import os
import pickle

import cv2

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
	countTrue = 0
	countFalse = 0
	predictedLabels = model.predict(testFeatures)
	for i in range(len(testLabels)):
		if testLabels[i] == predictedLabels[i]:
			# print("TRUE", "Predicted:", predictedLabels[i], "Actual:", testLabels[i])
			countTrue += 1
		else:
			# print("FALSE", "Predicted:", predictedLabels[i], "Actual:", testLabels[i])
			countFalse += 1

	accuracy = countTrue / (countTrue + countFalse)
	print("LDA Accuracy", accuracy)
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
	countTrue = 0
	countFalse = 0
	predictedLabels = modelSVM.predict(testFeatures)
	for i in range(len(testLabels)):
		if testLabels[i] == predictedLabels[i]:
			# print("TRUE", "Predicted:", predictedLabels[i], "Actual:", testLabels[i])
			countTrue += 1
		else:
			# print("FALSE", "Predicted:", predictedLabels[i], "Actual:", testLabels[i])
			countFalse += 1

	accuracy = countTrue / (countTrue + countFalse)

	print("SVM Accuracy", accuracy)
	return testLabels, predictedLabels
