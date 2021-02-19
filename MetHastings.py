import numpy as np
import math
import random
import cv2
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error

def rmse(predictions, targets):
    RMSE = mean_squared_error(targets, predictions, squared=False)
    return RMSE

def seperateArray(array, pixels):
    h, w = array.shape

    # Values for non NaN points in the array
    values = np.empty(pixels)
    # X and Y coordinates of values
    points = np.empty((pixels, 2))

    c = 0
    for i in range(h):
        for j in range(w):
            if not math.isnan(array[i][j]):
                values[c] = array[i][j]
                points[c] = (i, j)
                c += 1
    return values, points

def nInterp2D(pixels, array):
    # Given a specific number of non-NaN pixels
    # interpolate to the grid of the 2D array
    c = 0
    h, w = array.shape

    # Grid to interpolate over
    grid_y, grid_x = np.mgrid[0:h, 0:w]

    values, points = seperateArray(array, pixels)

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

def nonNan(array):
    h, w = array.shape
    counter = 0
    for i in range(h):
        for j in range(w):
            if not math.isnan(array[i][j]):
                counter += 1
    return counter

def invertGrayscale(img):
    img = cv2.bitwise_not(img)
    return img

def createFeatureMap(img, ROI, bConst, rConst):
    H, W = img.shape
    fMap = np.zeros((H, W))

    #invert so closest pixels have higher weightings
    imgInvert = invertGrayscale(img)

    #Background
    for i in range(H):
        for j in range(W):
            #fMap[i][j] = bConst
            fMap[i][j] = imgInvert[i][j] * bConst
            if fMap[i][j] <= 0:
                fMap[i][j] = bConst

    #ROI
    for r in ROI:
        x, y, w, h = r
        for i in range(y, y + h):
            for j in range(x, x + w):
                #fMap[i][j] = rConst
                fMap[i][j] = imgInvert[i][j] * rConst
                if fMap[i][j] <= 0:
                    fMap[i][j] = rConst
    return fMap

def walkIndex(prevI, max, sigma):
    a = -1
    b = 1
    if prevI - sigma < 0:
        #Can exceed the minimum
        a = -(prevI) / sigma
    if prevI + sigma > max:
        #Index can exceed maximum
        b = (max-prevI)/sigma
    propI = int(prevI + round(sigma * random.uniform(a, b)))
    return propI


def MetHastings(img, ROI, pixels, bConst, roiConst, N):
    imH, imW = img.shape

    #Define AS value array and Feature Map
    #fMap = np.zeros((imH, imW))
    AS = np.empty((imH, imW))
    AS[:] = np.nan

    if pixels > imH * imW:
        print("Error: Pixel allocation is too large")
        pixels = imH * imW

    #Make Feature Map
    fMap = createFeatureMap(img, ROI, bConst, roiConst)

    #Pixels sampled
    pCount = 0

    #Set initial Met Hastings position
    rX = random.randint(0, imW - 1)
    rY = random.randint(0, imH - 1)
    nX = rX
    nY = rY
    n = img[rY][rX]
    AS[rY][rX] = n
    pCount += 1

    #Determine if the iteration succeeded in finding a good candidate
    accept = False

    #Loop through other pixels
    while pCount < pixels:
        accept = False
        for i in range(N):
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
                    accept = True

        if accept:
            # Check if pixel is used
            if math.isnan(AS[nY][nY]):
                AS[nY][nX] = n
                pCount += 1

    nearestAS = nInterp2D(pixels, AS)
    #nearestAS = AS
    return nearestAS

def RandomWalkMetHastings(img, ROI, pixels, bConst, roiConst, sigma, N):
    imH, imW = img.shape

    AS = np.empty((imH, imW))
    AS[:] = np.nan

    AS = uniformSpread(img, pixels, AS)
    fMap = createFeatureMap(img, ROI, bConst, roiConst)

    #Split array to quickly go through pixels
    pUsed = nonNan(AS)
    values, points = seperateArray(AS, pUsed)

    for i in range(values.size):
        y, x = points[i]
        yPrev = y = int(y)
        xPrev = x = int(x)
        for j in range(N):
            xProp = walkIndex(xPrev, imW-1, sigma)
            yProp = walkIndex(yPrev, imH-1, sigma)

            # Ratio of new point compared to previous on feature map
            α = min((fMap[yProp][xProp]) / (fMap[yPrev][xPrev]), 1)

            # Random int between 1 and 0
            r = random.uniform(0, 1)
            # Check proposal
            if r < α:
                # Check if point is used
                if math.isnan(AS[yProp][xProp]):
                    yPrev = yProp
                    xPrev = xProp
        AS[y][x] = np.nan
        AS[yPrev][xPrev] = img[yPrev][xPrev]
    nearestAS = nInterp2D(pixels, AS)
    nearestAS = AS
    return nearestAS

#depth = cv2.imread("depth.png")
depth = cv2.imread("Mannequin.png")
depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
ROI = np.array([[0, 0, 142, 142]])
#ROI = np.array([[0, 27, 576, 391], [587, 172, 270, 90]])
RWHH = RandomWalkMetHastings(depth, ROI, 10000, 1, 10, 25, 5)
#RWHH = MetHastings(depth, ROI, 10000, 1, 10, 5)
cv2.imshow("MH", RWHH)
cv2.waitKey(0)
cv2.destroyAllWindows()

