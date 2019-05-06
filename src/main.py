import os
import time

import cv2
import numpy as np

from src.image_processing.bg_extraction import remove_bg, reset_bg
from src.Test import predict

# Initializing Video capture
cap = cv2.VideoCapture(0)

# Box parameters
BOX_SIZE = 300
BOX_X, BOX_Y = int(cap.get(3) / 8), int(cap.get(4) / 2 - BOX_SIZE // 2)
box_color = (0, 0, 255)  # red

FILENAME = 'E'


'''
	Processing the frame
'''


def process_image():
	# Reading frames from Video
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)

	# Converting frame to gray scale
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Getting inside of the box and converting it to gray scale
	cropped = frame[BOX_Y:BOX_Y + BOX_SIZE, BOX_X:BOX_X + BOX_SIZE]
	# cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

	# ------------------------------------------------------------------------------
	# Removing the background and segmenting the hand
	img = None
	ret = remove_bg(cropped)
	if ret is not None:
		img, box_color = remove_bg(cropped)

	if img is not None:
		frame[BOX_Y:BOX_Y + BOX_SIZE, BOX_X:BOX_X + BOX_SIZE] = img

	return frame, img


'''
	Key Handling
'''


def key_handler(img, word):
	# Keyboard events
	keyCode = cv2.waitKey(50)

	# Reset Background
	if keyCode == ord('r'):
		img = None
		reset_bg()

	# Save
	if keyCode == 115:
		if img is not None:
			print("s pressed. Saving picture...")
			# predicted = predict([img.ravel()], "SVM")
			# print(predicted)
			word += "A"

			if not os.path.exists("../samples/{}".format(FILENAME)):
				os.makedirs("../samples/{}".format(FILENAME))

			OUTPUT_FILE_NAME = "../samples/{}/{}-{}.JPG".format(FILENAME, FILENAME, str(int(time.time())))
			cv2.imwrite(OUTPUT_FILE_NAME, img)

	return keyCode, word


'''
	Drawing the Prediction window
'''


def draw_predicted_window(word):
	blackboard = np.zeros((300, 700), dtype=np.uint8)
	cv2.putText(blackboard, str(word), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.imshow('Predictions', blackboard)


'''
	Drawing the Region Of Interest
'''


def draw_roi(frame):
	# Drawing Box in the given coordinates
	cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_SIZE, BOX_Y + BOX_SIZE), box_color, 2)

	# Showing the frame
	cv2.imshow('Sign Language', frame)


'''
	Main Loop
'''


def main_loop():
	word = ""

	while True:
		# Processing the frame
		frame, img = process_image()

		# Drawing
		draw_roi(frame)
		draw_predicted_window(word)

		# Key code handling
		keyCode, word = key_handler(img, word)

		# Exit
		if keyCode == 27:
			print("ESC pressed. Closing...")
			break


main_loop()

# Releasing Video capture
cap.release()
cv2.destroyAllWindows()
