import os
import sys

import numpy
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import cv2

features = []
labels = []
for imageName in os.listdir("../samples"):
	imageLabel = imageName[0]
	features.append(cv2.imread("../samples/" + imageName, 0).ravel())
	labels.append(imageLabel)
	print(imageName, imageLabel)

# features = features.ravel()
# labels = np.array(labels)
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# y = np.array([1, 1, 1, 2, 2, 2])
clf = LinearDiscriminantAnalysis()
clf.fit(features, labels)
LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
						   solver='svd', store_covariance=False, tol=0.0001)

toBePredicted = [cv2.imread("../samples/C-1554389496.JPG", 0).ravel()]
print(clf.predict(toBePredicted))
