import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# Load model from saved file
with open('../generatedModels/modelLDA.obj', 'rb') as fp:
	model = pickle.load(fp)
with open('../generatedModels/modelSVM.obj', 'rb') as fp:
	modelSVM = pickle.load(fp)


def reduceWithPCA(features, n):
	pca = PCA(n)
	return pca.fit_transform(features)


def predict(features, algorithm):
	if algorithm == "SVM":
		return modelSVM.predict(features)
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

	testFeatures = reduceWithPCA(testFeatures, 25)  # TODO: to enable PCA, uncomment this line

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


testLabels, predictedLabels = testLDA()
class_names = ['A', 'B', 'C', 'D', 'E']


def plot_confusion_matrix(y_true, y_pred, classes,
						  normalize=False,
						  title=None,
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	# classes = classes[unique_labels(y_true, y_pred)]
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   # ... and label them with the respective list entries
		   xticklabels=classes, yticklabels=classes,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(testLabels, predictedLabels, classes=class_names,
					  title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(testLabels, predictedLabels, classes=class_names, normalize=True,
					  title='Normalized confusion matrix')

plt.show()


def testSVM():
	# Create test features and labels
	testFeatures = []
	testLabels = []
	for imageName in os.listdir("../samples/test"):
		imageLabel = imageName[0]
		testFeatures.append(cv2.imread("../samples/test/" + imageName, 0).ravel())
		testLabels.append(imageLabel)

	# testFeatures = reduceWithPCA(testFeatures, 25)  # TODO: to enable PCA, uncomment this line

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
