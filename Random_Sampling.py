import numpy as np
import math
import random
from scipy.interpolate import griddata

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
    return nearestRS