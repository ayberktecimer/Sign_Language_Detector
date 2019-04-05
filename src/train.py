import os

import cv2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def reduceWithPCA(features, n):
	pca = PCA(n)
	return pca.fit_transform(features)


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

# testFeatures = reduceWithPCA(testFeatures, 25) # TODO: to enable PCA, uncomment this line

# Test and calculate accuracy
countTrue = 0
countFalse = 0
predictedLabels = model.predict(testFeatures)
for i in range(len(testLabels)):
	if testLabels[i] == predictedLabels[i]:
		print("TRUE", "Predicted:", predictedLabels[i], "Actual:", testLabels[i])
		countTrue += 1
	else:
		print("FALSE", "Predicted:", predictedLabels[i], "Actual:", testLabels[i])
		countFalse += 1

accuracy = countTrue / (countTrue + countFalse)
print("Accuracy", accuracy)
