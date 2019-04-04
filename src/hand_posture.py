import cv2
import numpy as np

'''
	Image Preprocessing: Getting hand posture using color filtering

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

	# Applying dilation to the mask with ellipsoid to fill empty spots
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	mask = cv2.dilate(mask, kernel, iterations = 4)

	# Applying Gaussian Blur
	mask = cv2.GaussianBlur(mask, (5, 5), 100)

	# Returning the result image (posture)
	return mask
