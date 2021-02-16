import numpy as np
import math
import cv2
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

def getNewDimensions(nPixels, oWidth, oHeight):
    #Calculate Aspect Ratio
    aspectRatio = oWidth/oHeight

    #Calculate new width and height
    nWidth = math.sqrt(nPixels * aspectRatio)
    nHeight = nPixels/ nWidth

    return (nWidth, nHeight)


def checkRounding(limit, w, h):
    tVal = round(w) * round(h)
    if round(w) * round(h) > limit:
        if w * round(h) > limit:
            h = math.floor(h)
        if round(w) * h > limit:
            w = math.floor(w)

    w = round(w)
    h = round(h)
    return w, h

def uniformS(img, nPixels):
    imH, imW = img.shape

    US = np.empty((imH, imW))
    US[:] = np.nan

    nW, nH = getNewDimensions(nPixels, imW, imH)
    nW, nH = checkRounding(nPixels, nW, nH)

    stepW = np.linspace(0, imW - 1, nW)
    stepH = np.linspace(0, imH - 1, nH)

    counter = 0
    for i in stepH:
        i = round(i)
        for j in stepW:
            j = round(j)
            US[i][j] = img[i][j]
            counter += 1

    uniformNearest = nInterp2D(nPixels, US)
    return uniformNearest