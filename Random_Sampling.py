import cv2
import numpy as np
import math
import random

#https://stackoverflow.com/questions/17197492/is-there-a-library-function-for-root-mean-square-error-rmse-in-python
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def randomSample(img, startX,startY,endX,endY,pixels):
    for i in range(pixels):
        rX = random.randint(startX, endX)
        rY = random.randint(startY, endY)

        img[rY][rX] = 255
    return img

#Output from model
#Frame 30
#ROI = np.array([[0, 27, 576, 391], [587, 172, 270, 90]])
#Frame 05 cave_2
ROI = np.array([[418, 67, 211, 310]])

depth = cv2.imread("Depth.png")
depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)

#Set Total Pixels
pix = 50000
roiPor = 80
backPor = 100 - roiPor

#RGB Width and Height
imH, imW = depth.shape

#blank_image = np.zeros((h,w), np.uint8)

AS = np.zeros((imH, imW), np.uint8)
RS = np.zeros((imH, imW), np.uint8)

#========Linear Random Sample Background=========
rows, cols = RS.shape
for i in range(0, pix):
    rX = random.randint(0, cols-1)
    rY = random.randint(0, rows-1)

    RS[rY][rX] = depth[rY][rX]
    #RS[rY][rX] = 255
#================================================


#========Random Adaptive Sampling=========
rows, cols  = AS.shape

#Round up ROI pixels
roiTotPix = math.ceil(pix * (roiPor / 100))

#Round Down background pixels
backPix = math.floor(pix * (backPor/100))

#Random Sample Background
#img = randomSample(img, 0, 0, cols-1, rows-1, backPix)


for i in range(0, backPix):
    rX = random.randint(0, cols-1)
    rY = random.randint(0, rows-1)

    AS[rY][rX] = depth[rY][rX]
    #img[rY][rX] = 255

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
    ROIPort = (w * h)/(pixSumROI)
    nPixels = round(roiTotPix * ROIPort)
    for j in range(0, nPixels):
        rX = random.randint(x, x + w - 1)
        rY = random.randint(y, y + h - 1)

        AS[rY][rX] = depth[rY][rX]
        #img[rY][rX] = 255
#=====================================================

#RMSE of total image
rmseN = rmse(depth, depth)
rmseAS = rmse(AS, depth)
rmseRS = rmse(RS, depth)

print("Normal RMSE: ", rmseN)
print("Random RMSE: ", rmseRS)
print("AS RMSE: ", rmseAS)

#=============RMSE of ROI=================
print("RMSE of ROI")
for i in ROI:
    x,y,w,h = i

    cropAS = AS[y:y+h, x:x+w]
    cropRS = RS[y:y+h, x:x+w]
    cropDepth = depth[y:y+h, x:x+w]

    rmseAS = rmse(cropAS, cropDepth)
    rmseRS = rmse(cropRS, cropDepth)

    print("ROI Random RMSE: ", rmseRS)
    print("ROI AS RMSE: ", rmseAS)
#=========================================

#Show the Image
cv2.imshow("Adaptive Sampling", AS)
cv2.imshow("Random Sampling", RS)

#Show Cropped Image
cv2.imshow("ROI AS", cropAS)
cv2.imshow("ROI RS", cropRS)

#Save Image


# Keep Image Open
cv2.waitKey(0)

# Close Windows
cv2.destroyAllWindows()

cv2.imwrite('AS.png', AS)
cv2.imwrite("RS.png",RS)



