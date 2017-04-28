#!/usr/bin/env python
"The start of an automated malaria diagnosis system"
import sys, cv2, numpy
import pylab, argparse
import uninfected_extractor as extractor
import entropyTest as ent
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
 

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("image", help="file path to image for processing")
parser.add_argument("-sh","--show", help="show image processing pipeline",
 action="store_true")
parser.add_argument("-t","--test",type=int, help="select different tests for image")
parser.add_argument("-g","--graph", help="output graphs from processing stages", action="store_true")
args = parser.parse_args()

# Check file
try:
    open(args.image, 'r')
except IOError:
    print("IOError: No such file or directory: '" + args.image + "'")
    sys.exit()

if(args.test == 1):
    import segTest
    segTest.getCells(args.image)
if(args.test == None or args.test == 4):
    im2 = cv2.imread(args.image)
    finIm = cv2.imread(args.image)
    import segTest4
    conts, image = segTest4.getContours(args.image)

    segImages = segTest4.getCells(conts,image)

    kms = segTest4.kTrain(segImages, args.graph)

    for x in range(len(segImages)):
        
        entropy1 = ent.entropy(segImages[x][0],False)
        y1, x1 = segImages[x][0].shape
        XW = segImages[x][4]
        YH = segImages[x][2]
        Y = segImages[x][1]
        X = segImages[x][3]

        res = segTest4.predict(kms,y1,x1)

        if(res == 0):
            cv2.rectangle(im2,(X,Y),(XW,YH),(0,255,0,2))
        if(res == 1):
            cv2.rectangle(im2,(X,Y),(XW,YH),(255,0,0,2))
        if(res == 2):
            cv2.rectangle(im2,(X,Y),(XW,YH),(0,0,255,2))
        if(extractor.idCell(finIm[Y:YH, X:XW])):
            cv2.rectangle(finIm,(X,Y),(XW,YH),(0,0,0,2))
            print(entropy1)
        if(entropy1 <= -100000):
            cv2.rectangle(finIm,(X,Y),(XW,YH),(0,255,0,2))
    
    if(args.show == True):
        cv2.imshow("Test 4 Segmentation", im2)
        cv2.waitKey(0)
    cv2.imshow("Cells Identified as Infected", finIm)
    cv2.waitKey(0)


	
	
#-------------------------------------------------------------------------------
# End of Malarian.
#-------------------------------------------------------------------------------
