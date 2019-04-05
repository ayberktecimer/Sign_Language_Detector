import os
import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Create training features and labels
trainFeatures = []
trainLabels = []
for imageName in os.listdir("../samples/train"):
	imageLabel = imageName[0]
	trainFeatures.append(cv2.imread("../samples/train/" + imageName, 0).ravel())
	trainLabels.append(imageLabel)
	print(imageName, imageLabel)

pca = PCA(25)
reducedTrainFeatures = pca.fit_transform(trainFeatures)

# Initialize LDA model and train
model = LinearDiscriminantAnalysis()
model.fit(reducedTrainFeatures, trainLabels)
LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False,
						   tol=0.0001)

# Create test features and labels
testFeatures = []
testLabels = []
for imageName in os.listdir("../samples/test"):
	imageLabel = imageName[0]
	testFeatures.append(cv2.imread("../samples/test/" + imageName, 0).ravel())
	testLabels.append(imageLabel)
	print(imageName, imageLabel)

reducedTestFeatures = pca.fit_transform(testFeatures)

# Test and calculate accuracy
countTrue = 0
countFalse = 0
predictedLabels = model.predict(reducedTestFeatures)
for i in range(len(testLabels)):
	if testLabels[i] == predictedLabels[i]:
		print("TRUE", "Predicted:", predictedLabels[i], "Actual:", testLabels[i])
		countTrue += 1
	else:
		print("FALSE", "Predicted:", predictedLabels[i], "Actual:", testLabels[i])
		countFalse += 1

accuracy = countTrue / (countTrue + countFalse)
print("Accuracy", accuracy)
