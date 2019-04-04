import cv2

from bg_extraction import remove_bg


img = cv2.imread('../samples/A/A-1554389273.JPG', 0)


img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 121, 21)

cv2.imshow('abc', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
