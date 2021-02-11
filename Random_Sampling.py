import cv2
import numpy as np
import math
import random
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def rmse(predictions, targets):
    RMSE = mean_squared_error(targets, predictions, squared=False)
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
    Nearest = Nearest.astype(np.uint8)
    return Nearest

def randomAS(img, ROI, pixels, roiPort):
    imH, imW = img.shape

    AS = np.empty((imH, imW))
    AS[:] = np.nan

    if pixels > imH * imW:
        print("Error: Pixel allocation is too large")
        pixels = imH * imW

    # Round ROI pixels
    newROIP = round(pixels * (roiPort / 100))

    totROI = 0

    #Total ROI pixels
    for r in ROI:
        x, y, w, h = r
        totROI += w * h

    if newROIP > totROI:
        newROIP = totROI

    pCount = 0
    for r in ROI:
        x, y, w, h = r
        #Portion of total ROI
        ROIPort = (w * h) / (totROI)
        nPixels = newROIP * ROIPort
        while pCount < nPixels:
            rX = random.randint(x, x + w - 1)
            rY = random.randint(y, y + h - 1)

            if math.isnan(AS[rY][rX]):
                AS[rY][rX] = img[rY][rX]
                pCount += 1

    while pCount < pixels:
        rX = random.randint(0, imW - 1)
        rY = random.randint(0, imH - 1)

        if math.isnan(AS[rY][rX]):
            AS[rY][rX] = img[rY][rX]
            pCount += 1

    nearestAS = nInterp2D(pixels, AS)
    print("AS pixels used: ", pCount)
    return nearestAS

def randomS(img, pixels):
    imH, imW = img.shape

    RS = np.empty((imH, imW))
    RS[:] = np.nan

    if pixels > imH * imW:
        print("Error: Pixel allocation is too large")
        pixels = imH * imW

    pCount = 0
    while pCount < pixels:
        rX = random.randint(0, imW - 1)
        rY = random.randint(0, imH - 1)

        if math.isnan(RS[rY][rX]):
            RS[rY][rX] = img[rY][rX]
            pCount += 1

    nearestRS = nInterp2D(pixels, RS)
    print("RS pixels used: ", pCount)
    return nearestRS

#Output from model
#Frame 30
#ROI = np.array([[0, 27, 576, 391], [587, 172, 270, 90]])
#Frame 05 cave_2
ROI = np.array([[418, 67, 211, 310]])

depth = cv2.imread("Depth.png")
depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)

#Set Total Pixels
pix = 10000

ASNearest = randomAS(depth, ROI, pix, 40)
RSNearest = randomS(depth, pix)

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

'''
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
'''



#Show the Image
cv2.imshow("Adaptive Sampling", ASNearest)
cv2.imshow("Random Sampling", RSNearest)

#Show Cropped Image
cv2.imshow("ROI AS", cropAS)
cv2.imshow("ROI RS", cropRS)

#plt.show()

# Keep Image Open
cv2.waitKey(0)

# Close Windows
cv2.destroyAllWindows()

#Save Image
cv2.imwrite('AS.png', ASNearest)
cv2.imwrite("RS.png", RSNearest)



