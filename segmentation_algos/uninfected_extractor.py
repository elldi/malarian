#!/usr/bin/env python
"Test 1 of Classification - Basic image threshold to extract infected cells."
"RGB values."
import sys, cv2, os
import numpy as np

def getCells(fn):
	im = cv2.imread (fn,0)
	segColor = cv2.imread (fn)

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

	im2,contours,hierarchy = cv2.findContours(th1, 1, 2)

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		output = im[y:y+h, x:x+w]

		ySize, xSize= output.shape

		if(ySize < 100 and ySize > 50):
			checkPurple(segColor[y:y+h, x:x+w], y,y+h,x,x+w)




def checkPurple(image, y1,x1,y2,x2):
	## y,x,c = im.shape

	lower_gray = np.array([90,80,80])
	upper_gray = np.array([100,90,90])

	mask_gray = cv2.inRange(image,lower_gray,upper_gray)

	lower_purp = np.array([220,150,150])
	upper_purp = np.array([230,160,160])

	mask_purp = cv2.inRange(image,lower_purp,upper_purp)

	countG = cv2.countNonZero(mask_gray)
	countP = cv2.countNonZero(mask_purp)

	if(countG > 1 and countP > 1):
		## cv2.imshow("Hopeful",image)
		## cv2.waitKey(0)
		var = "   " + str(y1) + " " + str(x1) + " " + str(y2) + " " + str(x2)
		print(var)

		
def idCell(image):
	lower_gray = np.array([90,80,80])
	upper_gray = np.array([100,90,90])

	mask_gray = cv2.inRange(image,lower_gray,upper_gray)

	lower_purp = np.array([220,150,150])
	upper_purp = np.array([230,160,160])

	mask_purp = cv2.inRange(image,lower_purp,upper_purp)

	countG = cv2.countNonZero(mask_gray)
	countP = cv2.countNonZero(mask_purp)

	if(countG > 1 or countP >1):
		## cv2.imshow("Hopeful",image)
		## cv2.waitKey(0)
		return True
	else:
		return False




#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
##getCells(sys.argv[1])


