import cv2

# Initializing Video capture
cap = cv2.VideoCapture(0)

BOX_SIZE = 300
BOX_X, BOX_Y = int(cap.get(3) / 8), int(cap.get(4) / 2 - BOX_SIZE // 2)

while True:
    # Reading frames from Video
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Converting frame into gray scale
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Cropping the image
    cropped_img = gray_scale[BOX_Y:BOX_Y + BOX_SIZE, BOX_X:BOX_X + BOX_SIZE]

    # Applying Gaussian Blur
    cropped_img = cv2.GaussianBlur(cropped_img, (21, 21), 0)

    # Applying Adaptive Threshold
    cropped_img = cv2.adaptiveThreshold(cropped_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    # Drawing Box in the given coordinates
    cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_SIZE, BOX_Y + BOX_SIZE), (255, 0, 0), 2)

    # Showing the frame
    cv2.imshow('Sign Language', frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('test.JPG', cropped_img)
        break

# Releasing Video capture
cap.release()
cv2.destroyAllWindows()