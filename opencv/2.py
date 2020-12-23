import numpy as np
import cv2 as cv2
import matplotlib
import numpy as np

image = cv2.imread('img1.png')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
sharpen_kernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blur, 30,80)
cv2.imshow("edged",edged)
cv2.waitKey(0)
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
print("I count {} cells in this image".format(len(cnts)))
cells = image.copy()
cv2.drawContours(cells, cnts, -1, (0, 255, 0), 1)
cv2.imshow("Cells", cells)
cv2.waitKey(0)