import time
import datetime
import cv2
import numpy as np

from hand_posture import get_hand_posture



# Initializing Video capture
cap = cv2.VideoCapture(0)

# Box parameters
BOX_SIZE = 400
BOX_X, BOX_Y = int(cap.get(3) / 8), int(cap.get(4) / 2 - BOX_SIZE // 2)

# Main Loop
while True:
	# Reading frames from Video
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)

	# Getting inside of the box
	cropped = frame[BOX_Y:BOX_Y + BOX_SIZE, BOX_X:BOX_X + BOX_SIZE]

	# Getting hand posture
	img = get_hand_posture(cropped)

	# TODO: remove it when you are done
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame[BOX_Y:BOX_Y + BOX_SIZE, BOX_X:BOX_X + BOX_SIZE] = img

	# Drawing Box in the given coordinates
	cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_SIZE, BOX_Y + BOX_SIZE), (255, 0, 0), 2)

	# Showing the frame
	cv2.imshow('Sign Language', frame)

	# Keyboard events
	keyCode = cv2.waitKey(50)

	# Save
	# if keyCode == 115:
	# time.sleep(0.5)
	print("s pressed. Saving picture...")

	OUTPUT_FILE_NAME = "../samples/test-" + str(datetime.datetime.now()) + ".JPG"
	cv2.imwrite(OUTPUT_FILE_NAME, img)

	# Exit
	if keyCode == 27:
		print("ESC pressed. Closing...")
		break

# Releasing Video capture
cap.release()
cv2.destroyAllWindows()
