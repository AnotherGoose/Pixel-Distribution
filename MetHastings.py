import cv2
import numpy as np
import math
import random
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def rmse(predictions, targets):
    #return np.sqrt(((predictions - targets) **
    #MSE = np.square(np.subtract(targets, predictions)).mean()
    #RMSE = math.sqrt(MSE)
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

def MetHastings(img, ROI, pixels, bConst, roiConst):
    imH, imW = img.shape

    #Define AS value array and Feature Map
    fMap = np.zeros((imH, imW))
    AS = np.empty((imH, imW))
    AS[:] = np.nan

    if pixels > imH * imW:
        print("Error: Pixel allocation is too large")
        pixels = imH * imW

    #Make Feature Map
    #Background
    for i in range(imH):
        for j in range(imW):
            fMap[i][j] = img[i][j] * bConst

    #ROI
    for r in ROI:
        x, y, w, h = r
        for i in range(y, y + h + 1):
            for j in range(x, x + w + 1):
                fMap[i][j] = (img[i][j]) * roiConst

    #Pixels sampled
    pCount = 0

    #Set initial Met Hastings position
    rX = random.randint(0, imW)
    rY = random.randint(0, imH)
    nX = rX
    nY = rY
    n = img[rY][rX]
    AS[rY][rX] = n
    pCount += 1

    #Loop through other pixels
    while pCount < pix:
        # Random x and y Values
        rX = random.randint(0, imW - 1)
        rY = random.randint(0, imH - 1)
        # Ratio of new point compared to previous on feature map
        α = min((fMap[rY][rX]) / (fMap[nY][nX]), 1)
        # Random int between 1 and 0
        r = random.uniform(0, 1)
        if r < α:
            # Check if pixel is used
            if math.isnan(AS[rY][rX]):
                nX = rX
                nY = rY
                n = img[rY][rX]
                AS[rY][rX] = n
                pCount += 1

    nearestAS = nInterp2D(pix, AS)

    return nearestAS

depth = cv2.imread("Depth.png")
depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)

ROI = np.array([[418, 67, 211, 310]])

pix = 10000

gridNearest = MetHastings(depth, ROI, pix, 1, 10)

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

plt.subplot(221)
plt.imshow(gridNearest.T)
plt.xticks([])
plt.yticks([])
plt.subplot(222)
plt.imshow(depth.T)
plt.xticks([])
plt.yticks([])

plt.subplot(223)
plt.imshow(cropAS.T)
plt.xticks([])
plt.yticks([])
plt.subplot(224)
plt.imshow(cropDepth.T)
plt.xticks([])
plt.yticks([])

cv2.imshow("Met Hastings AS", gridNearest)

plt.show()
# Keep Image Open
cv2.waitKey(0)

# Close Windows
cv2.destroyAllWindows()

