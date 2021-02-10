import cv2
import numpy as np
import math
import random
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

#https://stackoverflow.com/questions/17197492/is-there-a-library-function-for-root-mean-square-error-rmse-in-python
def rmse(predictions, targets):
    #return np.sqrt(((predictions - targets) **
    MSE = np.square(np.subtract(targets, predictions)).mean()
    RMSE = math.sqrt(MSE)
    return RMSE


def nInterp2D(pixels, array):
    # Given a specific number of non-NaN pixels
    # interpolate to the grid of the 2D array
    c = 0
    h, w = array.shape

    # Grid to interpolate over
    grid_y, grid_x = np.mgrid[0:h, 0:w]

    # Values for non NaN points in the array
    values = np.empty(pixels)
    # X and Y coordinates of values
    points = np.empty((pixels, 2))

    for i in range(h):
        for j in range(w):
            if not math.isnan(array[i][j]):
                values[c] = array[i][j]
                points[c] = (i, j)
                c += 1
    Nearest = griddata(points, values, (grid_y, grid_x), method='nearest')
    return Nearest


#Output from model
#Frame 30
#ROI = np.array([[0, 27, 576, 391], [587, 172, 270, 90]])
#Frame 05 cave_2
ROI = np.array([[418, 67, 211, 310]])

depth = cv2.imread("Depth.png")
depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)

#Set Total Pixels
pix = 10000
roiPor = 80
backPor = 100 - roiPor

#RGB Width and Height
imH, imW = depth.shape

#blank_image = np.zeros((h,w), np.uint8)

AS = np.empty((imH, imW))
AS[:] = np.nan
RS = np.empty((imH, imW))
RS[:] = np.nan

#========Linear Random Sample Background=========
rows, cols = RS.shape

if ((rows * cols) < pix):
    pix = rows * cols
    print("Error! allocated more pixels than are available")

pCount = 0
while pCount < pix:
    rX = random.randint(0, cols - 1)
    rY = random.randint(0, rows - 1)

    if math.isnan(RS[rY][rX]):
        RS[rY][rX] = depth[rY][rX]
        pCount += 1
        # RS[rY][rX] = 255

print("Linear RS Pixels Used: ", pCount)

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


#========ROI's Random Sampling=========
pCount = 0
for i in ROI:
    x, y, w, h = i
    ROIPort = (w * h)/(pixSumROI)
    nPixels = round(roiTotPix * ROIPort)

    #Check to make sure doesnt indefinatley loop
    if((w * h) < nPixels):
        backPix = backPix + (nPixels - (w * h))
        nPixels = (w * h)

    while(pCount < nPixels):
        rX = random.randint(x, x + w - 1)
        rY = random.randint(y, y + h - 1)

        if math.isnan(AS[rY][rX]):
            AS[rY][rX] = depth[rY][rX]
            pCount += 1
            # AS[rY][rX] = 255
            #Can indefinetly loop if total number of pixels is surpassed
        #img[rY][rX] = 255
#=====================================================

#==============Random Sample Background================

if backPix > ((imW * imH) - pCount):
    #Make sure not over allocating pixels
    backPix = (imW * imH) - pCount

bCount = 0
while(bCount < backPix):
    rX = random.randint(0, cols - 1)
    rY = random.randint(0, rows - 1)

    if math.isnan(AS[rY][rX]):
        AS[rY][rX] = depth[rY][rX]
        bCount += 1
        #AS[rY][rX] = 255

#=========PREFORM INTERPOLATION BEFORE RMSE==================

ASNearest = nInterp2D(pix, AS)
RSNearest = nInterp2D(pix, RS)

#============================================================





#RMSE of total image
rmseN = rmse(depth, depth)
rmseAS = rmse(ASNearest, depth)
rmseRS = rmse(RSNearest, depth)

print("Normal RMSE: ", rmseN)
print("Random RMSE: ", rmseRS)
print("AS RMSE: ", rmseAS)

#=============RMSE of ROI=================
print("RMSE of ROI")
for i in ROI:
    x,y,w,h = i

    cropAS = ASNearest[y:y+h, x:x+w]
    cropRS = RSNearest[y:y+h, x:x+w]
    cropDepth = depth[y:y+h, x:x+w]

    rmseAS = rmse(cropAS, cropDepth)
    rmseRS = rmse(cropRS, cropDepth)

    print("ROI Random RMSE: ", rmseRS)
    print("ROI AS RMSE: ", rmseAS)
#=========================================

plt.subplot(231)
plt.imshow(ASNearest.T)
plt.xticks([])
plt.yticks([])
plt.subplot(232)
plt.imshow(RSNearest.T)
plt.xticks([])
plt.yticks([])
plt.subplot(233)
plt.imshow(depth.T)
plt.xticks([])
plt.yticks([])

plt.subplot(234)
plt.imshow(cropAS.T)
plt.xticks([])
plt.yticks([])
plt.subplot(235)
plt.imshow(cropRS.T)
plt.xticks([])
plt.yticks([])
plt.subplot(236)
plt.imshow(cropDepth.T)
plt.xticks([])
plt.yticks([])
plt.show()


'''
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
'''


