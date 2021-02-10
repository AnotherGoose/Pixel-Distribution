import cv2
import numpy as np
import math
import random
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def rmse(predictions, targets):
    #return np.sqrt(((predictions - targets) **
    MSE = np.square(np.subtract(targets, predictions)).mean()
    RMSE = math.sqrt(MSE)
    return RMSE


pix = 1000

#Define constants for feature map
backConst = 1
roiConst = 100

depth = cv2.imread("Depth.png")
depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)

ROI = np.array([[418, 67, 211, 310]])

#RGB Width and Height
imH, imW = depth.shape

fMap = np.zeros((imH, imW))
AS = np.empty((imH, imW))
AS[:] = np.nan

if pix > imH*imW:
    print("Error: Pixel allocation is too large")
    pix = imH*imW

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
pCount += 1


#=========Preform Met Hastings=============
while pCount < pix:
    #Random x and y Values
    rX = random.randint(0, imW - 1)
    rY = random.randint(0, imH - 1)
    #Ratio of new point compared to previous on feature map
    α = min((fMap[rY][rX])/(fMap[nY][nX]), 1)
    #Random int between 1 and 0
    r = random.uniform(0, 1)
    if r < α:
        #Check if pixel is used
        if math.isnan(AS[rY][rX]):
            nX = rX
            nY = rY
            n = depth[rY][rX]
            AS[rY][rX] = n
            pCount += 1

print(pCount)


#=========PREFORM INTERPOLATION BEFORE RMSE==================
#Grid to interpolate over
grid_y, grid_x = np.mgrid[0:imH, 0:imW]

#Values for non NaN points in the AS array
values = np.empty(pix)
#X and Y coordinates of values
points = np.empty((pix, 2))

#Counter to assure values and coordinates are linked
c = 0
for i in range(imH):
    for j in range(imW):
        if not math.isnan(AS[i][j]):
            values[c] = AS[i][j]
            points[c] = (i, j)
            c += 1


gridNearest = griddata(points, values, (grid_y, grid_x), method='nearest')

plt.subplot(121)
plt.imshow(gridNearest.T)
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(depth.T)
plt.xticks([])
plt.yticks([])
plt.show()

#============================================================

#RMSE of total image
rmseN = rmse(depth, depth)
rmseAS = rmse(gridNearest, depth)

print("Normal RMSE: ", rmseN)
print("AS RMSE: ", rmseAS)

#=============RMSE of ROI=================
print("RMSE of ROI")
for i in ROI:
    x,y,w,h = i

    cropAS = gridNearest[y:y+h, x:x+w]
    cropDepth = depth[y:y+h, x:x+w]

    rmseAS = rmse(cropAS, cropDepth)

    print("ROI AS RMSE: ", rmseAS)
#=========================================

#cv2.imshow("Met Hastings AS", AS)

# Keep Image Open
cv2.waitKey(0)

# Close Windows
cv2.destroyAllWindows()

