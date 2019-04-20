"""
File: Visualization
Author: Emre Sülün
Date: 20.04.2019
Project: Sign_Language_Detector
Description:
"""
import string

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plotConfusionMatrix(actualLabels, predictedLabels):
	np.set_printoptions(precision=2)
	class_names = string.ascii_uppercase
	class_names.replace("J", "")  # TODO: We don't have J letter, do we?

	# Plot non-normalized confusion matrix
	plot_confusion_matrix(actualLabels, predictedLabels, classes=class_names,
						  title='Confusion matrix, without normalization')

	# Plot normalized confusion matrix
	plot_confusion_matrix(actualLabels, predictedLabels, classes=class_names, normalize=True,
						  title='Normalized confusion matrix')

	plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
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
