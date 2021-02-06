import cv2
import numpy as np
import math
import random

def randomSample(img, startX,startY,endX,endY,pixels):
    for i in range(pixels):
        rX = random.randint(startX, endX)
        rY = random.randint(startY, endY)

        img[rY][rX] = 255
    return img

#Output from model
ROI = np.array([[0, 27, 576, 391], [587, 172, 270, 90]])

#Set Total Pixels
pix = 10000
roiPor = 80
backPor = 100 - roiPor

#RGB Width and Height
imW = 1024
imH = 436

#blank_image = np.zeros((h,w), np.uint8)

img = np.zeros((imH, imW), np.uint8)
RS = np.zeros((imH, imW), np.uint8)

#========Linear Random Sample Background=========
rows, cols = RS.shape
for i in range(0, pix):
    rX = random.randint(0, cols-1)
    rY = random.randint(0, rows-1)

    RS[rY][rX] = 255
#================================================


#========Random Adaptive Sampling=========
rows, cols  = img.shape

#Round up ROI pixels
roiTotPix = math.ceil(pix * (roiPor / 100))

#Round Down background pixels
backPix = math.floor(pix * (backPor/100))

#Random Sample Background
#img = randomSample(img, 0, 0, cols-1, rows-1, backPix)


for i in range(0, backPix):
    rX = random.randint(0, cols-1)
    rY = random.randint(0, rows-1)

    img[rY][rX] = 255

#========ROI's Random Sampling=========

#Calculate total number of original ROI pixels

pixSumROI = 0
for i in ROI:
    x, y, w, h = i
    pixSumROI += w * h


#====================REMOVE===========================
#ONLY USED TO SHOW THE REGION OF INTEREST
for i in ROI:
    x, y, w, h = i

    nPixels = round(roiTotPix * (w * h)/(pixSumROI))
    for j in range(0, nPixels):
        rX = random.randint(x, x + w - 1)
        rY = random.randint(y, y + h - 1)

        img[rY][rX] = 255
#=====================================================

#Show the Image
cv2.imshow("Adaptive Sampling",img)
cv2.imshow("Random Sampling",RS)
# Keep Image Open
cv2.waitKey(0)

# Close Windows
cv2.destroyAllWindows()