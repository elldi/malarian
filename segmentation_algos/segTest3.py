#!/usr/bin/env python
"Test 3 of Segmentation - Segmenting red blood cells from an image"
import sys, cv2, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def getContours(fn):
	im = cv2.imread (fn,0)

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

	__,contours,__ = cv2.findContours(th1, 1, 2)

	return contours, im

def kTrain():
	finList = []
	finY = []
	finX = []

	loc = "../hate_images/"
	files = os.listdir(loc)
	for file1 in files:
		if(("jpg" in file1) and ("ese" not in file1)):
			contours,im = getContours(loc+file1)
			outImages = getCells(contours,im)
			for x in range(len(outImages)):
				yS, xS = outImages[x][0].shape
				finList.append([yS,xS])
				finY.append(yS)
				finX.append(xS)



	# y_pred = KMeans(n_clusters=3).fit_predict(finList)
	kmeans = KMeans(n_clusters=3,random_state = 0).fit(finList)
	# plt.scatter (finY,finX,c=y_pred)
	# plt.title("Classification of Segments")
	# plt.show()

	return kmeans



def getCells(contours,im):

	outImages = []

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		output = im[y:y+h, x:x+w]
		outImages.append([output,y,y+h,x,x+w])

	return outImages


	# y_pred = KMeans(n_clusters=3).fit_predict(varList)
	# kmeans = KMeans(n_clusters=3).fit(varList)

	# for cnt in contours:
	# 	x,y,w,h = cv2.boundingRect(cnt)
	# 	res = kmeans.predict([[abs(y-(y+h)), abs(x-(x+w))]])
	# 	if(res == 1):
	# 		var = "   " + str(y) + " " + str(y+h) + " " + str(x) + " " + str(x+w)
	# 		print(var)


	# plt.scatter (xList,yList,c=y_pred)
	# plt.title("Classification of Segments")
	# plt.show()

def predict(kms, y,x):
	return kms.predict([[y,x]])


#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

# getCells(sys.argv[1])

kms = kTrain()

conts, image = getContours(sys.argv[1])

segImages = getCells(conts,image)

for x in range(len(segImages)):
	y, x = segImages[x][0].shape
	if(predict(kms,y,x) == 1):
		var = "   " + str(segImages[x][1]) + " " + str(segImages[x][2]) + " " + str(segImages[x][3]) + " " + str(segImages[x][4])
		print(var)








