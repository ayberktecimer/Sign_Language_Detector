import cv2
import numpy as np

'''
	Getting hand posture using color filtering and thresholding (adaptive)

	@param img: the part from the frame
'''
def get_hand_posture(img):

	# Converting to HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Color for skin: Hue Saturation Value
	lower_color = np.array([0, 48, 80])
	upper_color = np.array([20, 255, 255])

	# Creating a mask with the given color ranges
	mask = cv2.inRange(hsv, lower_color, upper_color)

	# Applying erosion and dilation to the mask with ellipsoid
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	# mask = cv2.erode(mask, kernel)
	mask = cv2.dilate(mask, kernel)

	# Finally applying the mask
	res = cv2.bitwise_and(img, img, mask = mask)

	# Converting color to Gray scale
	res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

	# Applying Gaussian Blur
	res = cv2.GaussianBlur(res, (11, 11), 0)

	# Applying Threshold
	res = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)

	# Returning the result image (posture)
	return res
