import argparse
import cv2
import numpy as np
import imutils
import mahotas

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
help = "Path to the image")
args = vars(ap.parse_args()) #parses from command line

image = cv2.imread(args["image"])  #loads image from command line
cv2.imshow("ORIGINAL", image) 
cv2.waitKey(0)
# python opencv.py --image  image/coins.jpg
cv2.destroyAllWindows()

##### Resizing using imutils ######


r = 250.0 / image.shape[1]   ## calculates aspect ratio
dim = (250, int(image.shape[0] * r))   ##calculates new dimension of image

##r = 50.0 / image.shape[0]
##dim = (int(image.shape[1] * r), 50)

image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)  ## first argument is image, second argument is new dimension
cv2.imshow("Resized (Width)", image)  ## Shows resized image
cv2.destroyAllWindows()


###### Flipping  #######

flipped = cv2.flip(image, 1)
cv2.imshow("Flipped Horizontally", flipped)

flipped = cv2.flip(image, 0)
cv2.imshow("Flipped Vertically", flipped)

flipped = cv2.flip(image, -1)
cv2.imshow("Flipped Horizontally & Vertically", flipped)

cv2.waitKey(0)

cv2.destroyAllWindows()

######### Cropping ########

cropped = image[80:150 , 240:335]
cv2.imshow("Cropped image", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

############# Bit wise #############

rectangle = np.zeros((300, 300), dtype = "uint8")
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
cv2.imshow("Rectangle", rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

circle = np.zeros((300, 300), dtype = "uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow("Circle", circle)
cv2.waitKey(0)
cv2.destroyAllWindows()


bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND", bitwiseAnd)    ########  A bitwise AND is true if and only if both pixels are greater than zero
cv2.waitKey(0)
cv2.destroyAllWindows()

bitwiseOr = cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR", bitwiseOr)    ######## A bitwise OR is true if either of the two pixels are greater than zero.
cv2.waitKey(0)
cv2.destroyAllWindows()

bitwiseXor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR", bitwiseXor)  ######  A bitwise XOR is true if and only if either of the two pixels are greater than zero, but not both.
cv2.waitKey(0)
cv2.destroyAllWindows()


bitwiseNot = cv2.bitwise_not(circle)
cv2.imshow("NOT", bitwiseNot)  #####  A bitwise NOT inverts the “on” and “off” pixels in an image
cv2.waitKey(0)
cv2.destroyAllWindows()


################### MASKING ####################

mask = np.zeros(image.shape[:2], dtype = "uint8")
(cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)  ## //means integer division 
cv2.rectangle(mask, (cX - 75, cY - 75), (cX + 75 , cY + 75), 255,-1)  ###
cv2.imshow("Mask", mask)

masked = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)

cv2.destroyAllWindows()

mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.circle(mask, (cX, cY), 100, 255, -1)
masked = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Mask", mask)

cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)

cv2.destroyAllWindows()

############## noise reduction functions  ###############

#### Blurring ####

## Averaged Blurring ##

Ablurred = np.hstack([
cv2.blur(image, (3, 3)),
cv2.blur(image, (5, 5)),
cv2.blur(image, (7, 7))])
cv2.imshow("Averaged", Ablurred)
cv2.waitKey(0)

cv2.destroyAllWindows()


## Gaussian Blurring ##

Gblurred = np.hstack([
cv2.GaussianBlur(image, (3, 3), 0),
cv2.GaussianBlur(image, (5, 5), 0),
cv2.GaussianBlur(image, (7, 7), 0)])

cv2.imshow("Gaussian", Gblurred)

cv2.waitKey(0)

cv2.destroyAllWindows()

## Median Blur ##

Mblurred = np.hstack([
cv2.medianBlur(image, 3),
cv2.medianBlur(image, 5),
cv2.medianBlur(image, 7)])

cv2.imshow("Median", Mblurred)

cv2.waitKey(0)

cv2.destroyAllWindows()

## Bilateral Blur ##

Bblurred = np.hstack([
cv2.bilateralFilter(image, 5, 21, 21),
cv2.bilateralFilter(image, 7, 31, 31),
cv2.bilateralFilter(image, 9, 41, 41)])

cv2.imshow("Bilateral", Bblurred)

cv2.waitKey(0)

cv2.destroyAllWindows()

################# Thresholding  #####################


## we will use gaussian blurred image for Thresholding to eliminate high frequencyedges in the image
## image is converted to gray scale to eliminate rgb and alpha components

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", image)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)

## Firstargument- Simple Thresholding ##
## Second Argument- grayscale image to be thresholded
## Third Argument- manual threshold value
## Fourth Argument- maximum value applied during thresholding
## Fifth Argument- Thresholding method
(T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Binary", thresh)

(T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.
THRESH_BINARY_INV)

cv2.imshow("Threshold Binary Inverse", threshInv)

cv2.imshow("with mask", cv2.bitwise_and(image, image, mask =
threshInv)) ## Threshold Binary Inverse mask applied to gray scale

cv2.waitKey(0)

cv2.destroyAllWindows()


#Adaptive Thresholding ##

### first parameter- image to threshold, 
###second- argument- maximum value =255, 
###third argument- method for computing threshold, in this case mean of the suurounding pixels.
###fourth argument- method of thresholding
### fifth argument- neighbourhood size

thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
cv2.imshow("Mean Thresh", thresh)

thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
cv2.imshow("Gaussian Thresh", thresh)

cv2.waitKey(0)

cv2.destroyAllWindows()

## Otsu and Riddler-Calvard ##
T = mahotas.thresholding.otsu(blurred)
print("Otsu’s threshold: {}".format(T))

## Otsu’s threshold: 169


thresh = blurred.copy()
thresh[thresh > T] = 255


thresh[thresh < 255] = 0
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Otsu", thresh)

T = mahotas.thresholding.rc(blurred)
print("Riddler-Calvard: {}".format(T))

## Riddler-Calvard: 169.18437675287691

thresh = blurred.copy()

thresh[thresh > T] = 255
thresh[thresh < 255] = 0
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Riddler-Calvard", thresh)

cv2.waitKey(0)

cv2.destroyAllWindows()

################ Gradients and Edge detection #################


## Laplacin and Sobel ##


lap = cv2.Laplacian(image, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))

cv2.imshow("Laplacian", lap)
cv2.waitKey(0)

sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)

cv2.imshow("Sobel X", sobelX)
cv2.imshow("Sobel Y", sobelY)

cv2.imshow("Sobel Combined", sobelCombined)
cv2.waitKey(0)

cv2.destroyAllWindows()

########## Canny Edge Detector #############

canny = cv2.Canny(blurred, 30, 150)

## first argument- blurred gray scale image
## second argument- threshold_1 and threshold_2
### threshold_1 value= any value below this is considered to not be an edge
### threshhold_2 value= any value above ths is considered to be an edge


cv2.imshow("Canny", canny)

cv2.waitKey(0)
cv2.destroyAllWindows()

############## Contours ############

edged = cv2.Canny(blurred, 30, 150)
cv2.imshow("edged",edged)

#find contour are destructive methods hence we mke a copy

(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

### first argument- edged image
### second arguent- type of contour required
### third argument- approximate of contour

cube = edged.copy()

cv2.drawContours(cube, cnts, -1, ( 0, 0, 255), 1)
## argument 1- image to be drawn on
## argument 2- list of contours
## argument 3- contour index
## argument 4- color of line
## argument 5- thickness of line

cv2.imshow("Cube", cube)

cv2.waitKey(0)
cv2.destroyAllWindows()


#################### Important pointS ######################

########  A bitwise AND is true if and only if both pixels are greater than zero.
####### A bitwise OR is true if either of the two pixels are greater than zero.
######  A bitwise XOR is true if and only if either of the two pixels are greater than zero, but not both.
#####  A bitwise NOT inverts the “on” and “off” pixels in an image.
#### cv2.imshow("Original", image) #loads image
##cv2.waitKey(0) ## waits for user to press any key
#cv2.destroyAllWindows() ## destroys all hig gui windows opened