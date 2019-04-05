import os
import cv2
import pickle

# Load model from saved file
with open('model.obj', 'rb') as fp:
	model = pickle.load(fp)


def predict(features):
	return model.predict(features)


def testImages():
	# Create test features and labels
	testFeatures = []
	testLabels = []
	for imageName in os.listdir("../samples/test"):
		imageLabel = imageName[0]
		testFeatures.append(cv2.imread("../samples/test/" + imageName, 0).ravel())
		testLabels.append(imageLabel)
		print(imageName, imageLabel)

	# testFeatures = reduceWithPCA(testFeatures, 25) # TODO: to enable PCA, uncomment this line

	# Test and calculate accuracy
	countTrue = 0
	countFalse = 0
	predictedLabels = predict(testFeatures)
	for i in range(len(testLabels)):
		if testLabels[i] == predictedLabels[i]:
			print("TRUE", "Predicted:", predictedLabels[i], "Actual:", testLabels[i])
			countTrue += 1
		else:
			print("FALSE", "Predicted:", predictedLabels[i], "Actual:", testLabels[i])
			countFalse += 1

	accuracy = countTrue / (countTrue + countFalse)
	print("Accuracy", accuracy)
