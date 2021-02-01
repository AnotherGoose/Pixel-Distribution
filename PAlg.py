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

    tStepW = oWidth/ nWidth
    tStepH = oHeight/ nHeight

    (nWidth, nHeight) = checkRoundingDim(nPixels, nWidth, nHeight)
    #(tStepW, tStepH) = checkRoundingStep(nPixels, tStepW, tStepH, oWidth, oHeight)

    #CHECK ROUNDING
    tStepH = math.ceil(tStepH)
    tStepW = math.ceil(tStepW)
    #nW, nH, uStepW, uStepH = getNewDimensions(pix, uWid, uHei)

    uStepW = tStepW
    uStepH = tStepH

    return (nWidth, nHeight, uStepW, uStepH)


def checkRoundingDim(limit, w, h):
    #Check if the rounding for the width and height surpasses the pixel limit
    if limit < round(w) * round(h):
        #CLEAN UP THIS IS MESSY AS
        if limit > (round(w) * math.floor(h)):
            w = round(w)
            h = math.floor(h)

        elif limit > (math.floor(w) * round(h)):
            w = math.floor(w)
            h = round(h)

        elif limit > (math.floor(w) * math.floor(h)):
            w = math.floor(w)
            h = math.floor(h)
    else:
        w = round(w)
        h = round(h)
    return(w, h)
'''
def checkRoundingStep(limit, stepW, stepH, oWidth, oHeight):
    #FIX FUNCTION
    w = oWidth / stepW
    h = oHeight / stepH
    if h > nH:
        stepH
    if limit < round(w) * round(h):
        if limit > (round(w) * math.floor(h)):
            #w = round(w)
            stepH = math.ceil(stepH)

        elif limit > (math.floor(w) * round(h)):
            stepW = math.ceil(stepW)

        elif limit > (math.floor(w) * math.floor(h)):
            stepW = math.ceil(stepW)
            stepH = math.ceil(stepH)
    else:
        stepW = round(stepW)
        stepH = round(stepH)
    return(stepW, stepH)
'''

def nearestInterpolation():
    x = 0
    return x

#Output from model
ROI = np.array([[0, 27, 576, 391], [587, 172, 270, 90]])

#Set Total Pixels
pix = 1716
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
print("Total Pixels: ", pix)
print("Uniform pixels used: ", c)
#=================================


#Constant ROI Limit
#SET ROI LIMIT?

#Round up ROI pixels
roiTotPix = (pix * (roiPor / 100))

#Round Down background pixels
backPix = (pix * (backPor/100))

#Calculate total number of original ROI pixels

pixSumROI = 0
for i in ROI:
    x, y, w, h = i
    pixSumROI += w * h


#Calculate individual ROI pixels
newAspect = np.zeros((ROI.shape[0],4), np.uint8)
pixies = np.zeros((ROI.shape[0],1), np.uint8)

remPixels = 0
for i in range(0, ROI.shape[0]):
    x, y, w, h = ROI[i]
    oldRes = (w * h)

    roiRot = oldRes/pixSumROI
    nPixels = (roiTotPix * roiRot)
    newAspect[i] = getNewDimensions(nPixels + remPixels, w, h)
    (nW, nH, nSW, nSH) = newAspect[i]
    #If there are pixels remaining pass it on to the next ROI
    #UNSURE IF THIS WORKS FOR CONTINUING ROI MIGHT ONLY WORK FOR PASSING IT ONTO THE NEXT ROI
    #FIGURE OUT HOW LOG LEFT OVER PIXELS FOR THE ENTIRE ROIs
    #WHY IS THIS FUCKED
    #nP = nW * nH
    #remPixels = nPixels - nP

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

print("Adaptive pixels used: ", counter)

#Show the Image
cv2.imshow("Adaptive Sampling",img)
cv2.imshow("Uniform",uniform)
# Keep Image Open
cv2.waitKey(0)

# Close Windows
cv2.destroyAllWindows()