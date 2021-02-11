import numpy as np
import math
import random
from scipy.interpolate import griddata
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
    while pCount < pixels:
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

    nearestAS = nInterp2D(pixels, AS)
    return nearestAS
