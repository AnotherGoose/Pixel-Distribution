import numpy as np
import math
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
    if round(w) * round(h) > limit:
        if w * round(h) > limit:
            h = math.floor(h)
        if round(w) * h > limit:
            w = math.floor(w)

    w = round(w)
    h = round(h)
    return w, h

def uniformSpread(oArray, nPoints, nArray):
    oH, oW = oArray.shape

    if nPoints > oH * oW:
        nPoints = oH * oW

    nW, nH = getNewDimensions(nPoints, oW, oH)
    nW, nH = checkRounding(nPoints, nW, nH)

    stepW = np.linspace(0, oW - 1, nW)
    stepH = np.linspace(0, oH - 1, nH)

    counter = 0
    for i in stepH:
        i = round(i)
        for j in stepW:
            j = round(j)
            nArray[i][j] = oArray[i][j]
            counter += 1

    return nArray

def uniformS(img, nPixels):
    imH, imW = img.shape

    if(nPixels > imH * imW):
        nPixels = imH * imW

    US = np.empty((imH, imW))
    US[:] = np.nan

    US = uniformSpread(img, nPixels, US)

    uniformNearest = nInterp2D(nPixels, US)
    return uniformNearest

def nonNan(array):
    h, w = array.shape
    counter = 0
    for i in range(h):
        for j in range(w):
            if not math.isnan(array[i][j]):
                counter += 1
    return counter

def uniformAS(img, ROI, nPixels, rPort):
    imH, imW = img.shape

    if(nPixels > imH * imW):
        nPixels = imH * imW

    AS = np.empty((imH, imW))
    AS[:] = np.nan

    #Can be an input
    bPort = 1 - rPort

    rPixels = round(nPixels * rPort)
    bPixels = round(nPixels * bPort)

    #Pixels Remaining
    pRem = 0

    #Total pixels in ROI
    roiSum = 0
    for r in ROI:
        x, y, w, h = r
        roiSum += ((w + 1) * (h + 1))

    for r in ROI:
        x, y, w, h = r
        roiPort = ((w + 1) * (h + 1))/roiSum
        roiPixels = round(roiPort * rPixels)

        #Add on the remainding pixels
        roiPixels += pRem
        AS[y:y+h, x:x+w] = uniformSpread(img[y:y+h, x:x+w], roiPixels, AS[y:y+h, x:x+w])

        pRem = roiPixels - nonNan(AS[y:y+h, x:x+w])

    bPixels += pRem
    AS = uniformSpread(img, bPixels, AS)
    pUsed = nonNan(AS)
    uniformNearest = nInterp2D(pUsed, AS)
    return uniformNearest
