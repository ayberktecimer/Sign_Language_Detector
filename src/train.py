import os
import sys

import numpy
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import cv2

im = cv2.imread("../samples/A/A-1554389273.JPG", 0)
numpy.set_printoptions(threshold=sys.maxsize)
print(im)

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
						   solver='svd', store_covariance=False, tol=0.0001)
print(clf.predict([[-0.8, -1]]))
