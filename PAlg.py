import cv2
import numpy as np
import math

def getNewDimensions(nPixels, oWidth, oHeight):
    #Calculate Aspect Ratio
    aspectRatio = oWidth/oHeight
    #Calculate new width and height
    nWidth = math.sqrt(nPixels * aspectRatio)
    nHeight = nPixels/ nWidth

    #MAKE A CHECK FOR IF THE PIXELS ARE OVER
    if nPixels < round(nWidth) * round(nHeight):
        print("======================")
        print("ERROR IN NEW DIMENSIONS \nPixels: ", nWidth * nHeight)
        print("======================")

        #CLEAN UP THIS IS MESSY AS
        if nPixels > (round(nWidth) * math.floor(nHeight)):
            nWidth = round(nWidth)
            nHeight = math.floor(nHeight)
        elif nPixels > (math.floor(nWidth) * round(nHeight)):
            nWidth = math.floor(nWidth)
            nHeight = round(nHeight)
        elif nPixels > (math.floor(nWidth) * math.floor(nHeight)):
            nWidth = math.floor(nWidth)
            nHeight = math.floor(nHeight)
    else:
        nWidth = round(nWidth)
        nHeight = round(nHeight)

    #nW, nH, uStepW, uStepH = getNewDimensions(pix, uWid, uHei)
    uStepW = round(oWidth / nWidth)
    uStepH = round(oHeight / nHeight)

    return (oHeight, nHeight, uStepW, uStepH)

def nearestInterpolation():
    x = 0
    return x

#Output from model
ROI = np.array([[0, 27, 576, 391], [587, 172, 270, 90]])

#Set Total Pixels
pix = 2500
roiPor = 80
backPor = 100 - roiPor

#RGB Width and Height
imW = 1024
imH = 436

#blank_image = np.zeros((h,w), np.uint8)

img = np.ones((imH, imW), np.uint8)
uniform = np.ones((imH, imW), np.uint8)

#========UNIFORM SAMPLING=========
uHei, uWid  = uniform.shape
uAspRot = uWid/uHei
nW, nH, uStepW, uStepH = getNewDimensions(pix, uWid, uHei)
#uStepW = round(uWid/nW)
#uStepH = round(uHei/nH)

c = 0
for i in range(0, uHei):
    uY = (i * uStepH)
    if uY < imH:
        for j in range(0, uWid):
            uX = (j * uStepW)
            if uX < imW:
                #WHY IS THIS FAILING
                if c < pix:
                    uniform[uY][uX] = 255
                    c += 1
print("Pixels used: ", c)
#=================================


#Constant ROI Limit
#SET ROI LIMIT?

#Round up ROI pixels
roiTotPix = round(pix * (roiPor / 100))

#Round Down background pixels
backPix = round(pix * (backPor/100))

#Calculate total number of original ROI pixels

pixSumROI = 0
for i in ROI:
    x, y, w, h = i
    pixSumROI += w * h


#Calculate individual ROI pixels
newAspect = np.zeros((ROI.shape[0],4), np.uint8)
pixies = np.zeros((ROI.shape[0],1), np.uint8)

for i in range(0, ROI.shape[0]):
    x, y, w, h = ROI[i]
    oldRes = (w * h)

    roiRot = oldRes/pixSumROI
    nPixels = round(roiTotPix * roiRot)
    newAspect[i] = getNewDimensions(nPixels, w, h)
    #CHANGE ACCORDINGLY
    (newW, newH, s, p) = newAspect[i]

    print("===========================")
    print("Aspect Ratio", (w/h))
    print("New Pixels: ", nPixels)
    print("Orig W: ", w)
    print("Orig H: ", h)
    print("New W: ", newW)
    print("New H: ", newH)
    print("===========================")

#Make half the image white and the other black
counter = 0

for i in range(0, ROI.shape[0]):
    x, y, w, h = ROI[i]
    aW, aH, stepW, stepH = newAspect[i]

    #====================REMOVE===========================
    #ONLY USED TO SHOW THE REGION OF INTEREST
    for j in range(y, y + h):
        for k in range(x, x + w):
            img[j][k] = 20
    #=====================================================

    #Calculate the step value for the pixels
    #stepW = round(w / aW)
    #stepH = round(h / aH)

    #STOP WHEN TOTAL PIXELS ARE REACHED
    #Loop by each step which has been defined

    count = 0
    for j in range(aH):
        yN = y + (j * stepH)
        #SHOULDNT HAPPEN
        if yN < imH:
            for k in range(aW):
                xN = x + (k * stepW)
                #THIS SHOULDNT EVER FUCKING HAPPEN
                if xN < imW:
                    img[yN][xN] = 255
                    counter += 1

print("Total number of iterations for this loop is: ", counter)

#Show the Image
cv2.imshow("Adaptive Sampling",img)
cv2.imshow("Uniform",uniform)
# Keep Image Open
cv2.waitKey(0)

# Close Windows
cv2.destroyAllWindows()