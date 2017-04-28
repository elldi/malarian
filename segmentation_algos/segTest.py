#!/usr/bin/env python
"Test 1 of Segmentation - Segmenting red blood cells from an image"
import sys, cv2, os
import numpy as np

def getCells(fn):
	im = cv2.imread (fn,0)
	im3 = cv2.imread (fn)

	blur = cv2.GaussianBlur(im,(5,5),0)

	ret, th1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	im2,contours,hierarchy = cv2.findContours(th1, 1, 2)

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		output = im[y:y+h, x:x+w]

		ySize, xSize= output.shape

		if(ySize < 100 and ySize > 50):
			var = "   " + str(y) + " " + str(y+h) + " " + str(x) + " " + str(x+w)
			print(var)
			cv2.rectangle(im3,(x,y),(x+w,y+h),(0,255,0),2)

	# cv2.imwrite("segtest1.png", im3)
	# cv2.imwrite("contours.png", im2)
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

getCells(sys.argv[1])

