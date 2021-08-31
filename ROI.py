import cv2
import numpy as np

im = cv2.imread("51442_door.png")
# r = cv2.selectROI(frame)

# frame_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# frame_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)
# edged = cv2.Canny(frame_blur, 30, 50)

# (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(edged, cnts, -1, (0, 255, 0), 3)
r = cv2.selectROI(im) # (689, 100, 239, 346)
print(r)
imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
cv2.imshow("Image", imCrop)
cv2.waitKey(0)

