import os
import cv2
import pickle

# Load model from saved file
with open('modelLDA.obj', 'rb') as fp:
	model = pickle.load(fp)
with open('modelSVN.obj', 'rb') as fp:
	modelSVN = pickle.load(fp)


def predict(features, algorithm):
	if algorithm == "SVN":
		return modelSVN.predict(features)
	elif algorithm == "LDA":
		return model.predict(features)
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

	# testFeatures = reduceWithPCA(testFeatures, 25) # TODO: to enable PCA, uncomment this line

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


def testSVN():
	# Create test features and labels
	testFeatures = []
	testLabels = []
	for imageName in os.listdir("../samples/test"):
		imageLabel = imageName[0]
		testFeatures.append(cv2.imread("../samples/test/" + imageName, 0).ravel())
		testLabels.append(imageLabel)

	# testFeatures = reduceWithPCA(testFeatures, 25) # TODO: to enable PCA, uncomment this line

	# Test and calculate accuracy
	countTrue = 0
	countFalse = 0
	predictedLabels = modelSVN.predict(testFeatures)
	for i in range(len(testLabels)):
		if testLabels[i] == predictedLabels[i]:
			# print("TRUE", "Predicted:", predictedLabels[i], "Actual:", testLabels[i])
			countTrue += 1
		else:
			# print("FALSE", "Predicted:", predictedLabels[i], "Actual:", testLabels[i])
			countFalse += 1

	accuracy = countTrue / (countTrue + countFalse)
	print("SVN Accuracy", accuracy)
