import os
import time
import datetime
import cv2
import numpy as np

from bg_extraction import remove_bg, reset_bg

# Initializing Video capture
cap = cv2.VideoCapture(0)

# Box parameters
BOX_SIZE = 300
BOX_X, BOX_Y = int(cap.get(3) / 8), int(cap.get(4) / 2 - BOX_SIZE // 2)
box_color = (0, 0, 255)	# red

FILENAME = 'E'

# Main Loop
while True:
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

	# ------------------------------------------------------------------------------

	# Drawing Box in the given coordinates
	cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_SIZE, BOX_Y + BOX_SIZE), box_color, 2)

	# Showing the frame
	cv2.imshow('Sign Language', frame)

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

			if not os.path.exists("../samples/{}".format(FILENAME)):
				os.makedirs("../samples/{}".format(FILENAME))

			OUTPUT_FILE_NAME = "../samples/{}/{}-{}.JPG".format(FILENAME, FILENAME, str(int(time.time())))
			cv2.imwrite(OUTPUT_FILE_NAME, img)

	# Exit
	if keyCode == 27:
		print("ESC pressed. Closing...")
		break

# Releasing Video capture
cap.release()
cv2.destroyAllWindows()
