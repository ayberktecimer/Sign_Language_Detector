import cv2
import numpy as np

# Background: currently none
background = None
no_frames = 0

'''
    Removing background and getting hand segmentation
'''
# Removes the background by averaging and segmenting
def remove_bg(img):
	global no_frames

	weight = 0.5

	# 'Detecting' background
	if no_frames < 60:
		bg_avg(img, weight)
	else:
		# Ready
		# Getting hand segmentation
		hand = segment(img)
		if hand is not None:
			(thresholded, segmented) = hand
			return thresholded, (0, 255, 0)
	# Increasing number of frames
	no_frames += 1


'''
    Background averaging
'''
def bg_avg(img, weight):
    global background

    if background is None:
        background = img.copy().astype("float")
        return

    cv2.accumulateWeighted(img, background, weight)


'''
    Getting hand segmentation from the given image
'''
def segment(image, threshold = 25):
    global background

    # find the absolute difference between background and current frame
    diff = cv2.absdiff(background.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
	cnts, hiearchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key = cv2.contourArea)
        return (thresholded, segmented)


'''
    Resetting background
'''
def reset_bg():
    global background
    global no_frames

    background = None
    no_frames = 0
