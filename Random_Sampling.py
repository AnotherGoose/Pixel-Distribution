import cv2
import numpy as np
import math
import random

#https://stackoverflow.com/questions/17197492/is-there-a-library-function-for-root-mean-square-error-rmse-in-python
def rmse(predictions, targets):
    #return np.sqrt(((predictions - targets) **
    MSE = np.square(np.subtract(targets, predictions)).mean()
    RMSE = math.sqrt(MSE)
    return RMSE

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
pix = 400000
roiPor = 80
backPor = 100 - roiPor

#RGB Width and Height
imH, imW = depth.shape

#blank_image = np.zeros((h,w), np.uint8)

AS = np.zeros((imH, imW), np.uint8)
RS = np.zeros((imH, imW), np.uint8)

#========Linear Random Sample Background=========
rows, cols = RS.shape

if ((rows * cols) < pix):
    pix = rows * cols
    print("Error! allocated more pixels than are available")

for i in range(0, pix):
    unused = False

    while(unused == False):
        rX = random.randint(0, cols - 1)
        rY = random.randint(0, rows - 1)

        if RS[rY][rX] == 0:
            RS[rY][rX] = depth[rY][rX]
            # RS[rY][rX] = 255
            unused = True
        else:
            unused = False


#================================================


#========Random Adaptive Sampling=========
rows, cols  = AS.shape

#Round up ROI pixels
roiTotPix = math.ceil(pix * (roiPor / 100))

#Round Down background pixels
backPix = math.floor(pix * (backPor/100))

#Calculate total number of original ROI pixels

pixSumROI = 0
for i in ROI:
    x, y, w, h = i
    pixSumROI += w * h


#====================REMOVE===========================
#ONLY USED TO SHOW THE REGION OF INTEREST
pCount = 0
for i in ROI:
    x, y, w, h = i
    ROIPort = (w * h)/(pixSumROI)
    nPixels = round(roiTotPix * ROIPort)

    #Check to make sure doesnt indefinatley loop
    if((w * h) < nPixels):
        backPix = backPix + (nPixels - (w * h))
        nPixels = (w * h)

    for j in range(0, nPixels):
        unused = False
        while (unused == False):
            rX = random.randint(x, x + w - 1)
            rY = random.randint(y, y + h - 1)

            if AS[rY][rX] == 0:
                AS[rY][rX] = depth[rY][rX]
                pCount += 1
                # AS[rY][rX] = 255
                unused = True
            else:
                #Can indefinetly loop if total number of pixels is surpassed
                unused = False
        #img[rY][rX] = 255
#=====================================================

#Random Sample Background
#img = randomSample(img, 0, 0, cols-1, rows-1, backPix)

if backPix > ((imW * imH) - pCount):
    #Make sure not over allocating pixels
    backPix = (imW * imH) - pCount

for i in range(0, backPix):
    unused = False
    while (unused == False):
        rX = random.randint(0, cols - 1)
        rY = random.randint(0, rows - 1)

        if AS[rY][rX] == 0:
            AS[rY][rX] = depth[rY][rX]
            #AS[rY][rX] = 255
            unused = True
        else:
            unused = False
    #img[rY][rX] = 255

#========ROI's Random Sampling=========

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



