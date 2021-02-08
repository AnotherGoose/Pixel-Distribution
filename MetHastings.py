import cv2
import numpy as np
import math
import random

def rmse(predictions, targets):
    #return np.sqrt(((predictions - targets) **
    MSE = np.square(np.subtract(targets, predictions)).mean()
    RMSE = math.sqrt(MSE)
    return RMSE


pix = 10000

#Define constants for feature map
backConst = 1
roiConst = 10

depth = cv2.imread("Depth.png")
depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)

ROI = np.array([[418, 67, 211, 310]])

#RGB Width and Height
imH, imW = depth.shape

fMap = np.zeros((imH, imW))
AS = np.zeros((imH, imW), np.uint8)

#=====Make Feature Map=======

#Define all points by background constant
for i in range(imH):
    for j in range(imW):
        fMap[i][j] = depth[i][j] * backConst


#Boost values within ROI
for r in ROI:
    x, y, w, h = r
    for i in range(y, y + h + 1):
        for j in range(x, x + w + 1):
            fMap[i][j] = (depth[i][j]) * roiConst

#Pixel Count
pCount = 0

#Make Initial Position n
rX = random.randint(0, imW)
rY = random.randint(0, imH)
nX = rX
nY = rY
n = depth[rY][rX]
AS[rY][rX] = n
pCount +=1


#=========Preform Met Hastings=============
#for t in range(0, pix - 1):
while pCount < pix:
    #Random x and y Values
    rX = random.randint(0, imW - 1)
    rY = random.randint(0, imH - 1)
    #Ratio of new point compared to previous on feature map
    α = (fMap[rY][rX])/(fMap[nY][nX])
    #Random int between 1 and 0
    r = random.uniform(0, 1)
    if r < α:
        #Check if pixel is used
        if AS[rY][rX] == 0:
            nX = rX
            nY = rY
            n = depth[rY][rX]
            AS[rY][rX] = n
            pCount += 1

print(pCount)


#RMSE of total image
rmseN = rmse(depth, depth)
rmseAS = rmse(AS, depth)

print("Normal RMSE: ", rmseN)
print("AS RMSE: ", rmseAS)

#=============RMSE of ROI=================
print("RMSE of ROI")
for i in ROI:
    x,y,w,h = i

    cropAS = AS[y:y+h, x:x+w]
    cropDepth = depth[y:y+h, x:x+w]

    rmseAS = rmse(cropAS, cropDepth)

    print("ROI AS RMSE: ", rmseAS)
#=========================================

cv2.imshow("Met Hastings AS", AS)

# Keep Image Open
cv2.waitKey(0)

# Close Windows
cv2.destroyAllWindows()

