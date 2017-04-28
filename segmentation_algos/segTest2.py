#!/usr/bin/env python
"Test 2 of Segmentation - Segmenting red blood cells from an image"
import sys, cv2, os
import numpy as np

def getCells(fn):

	im = cv2.imread (fn,0) # Gray
	im2 = cv2.imread (fn)  # Colour

	# Reduce noise in image
	blur = cv2.GaussianBlur(im,(5,5),0)

	# Generate kernel for morphological functions. 
	kernel = np.ones((5,5), np.uint8)

	# Opening the image, erosion followed by dilation. 
	opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)

	# Get difference between erosion and dilation of image. 
	# Gradient will generate outline of image.
	gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)

	# Threshold image to prepare for input into the contour function. 
	__, th1 = cv2.threshold(gradient,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# cv2.imshow("Alright Trigg.", th1)
	# cv2.waitKey(0)
	# cv2.imwrite("gradient_report.png", th1)


	__,contours,__ = cv2.findContours(th1, 1, 2)

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		output = im[y:y+h, x:x+w]

		ySize, xSize= output.shape
		cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,0),2)

		# if(ySize < 100 and ySize > 50):
			# cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,0),2)
		var = "   " + str(y) + " " + str(y+h) + " " + str(x) + " " + str(x+w)
		print(var)


	# cv2.imshow("Testing", im2)
	# cv2.resizeWindow("Testing", 100,100)
	# cv2.waitKey(0)
	# cv2.imwrite("morph_show.png", im2)

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

getCells(sys.argv[1])