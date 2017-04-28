#!/usr/bin/env python
"Test 4 of Segmentation - Segmenting red blood cells from an image"
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

def kTrain(images, graph):
	finList = []
	finY = []
	finX = []

	for x in range(len(images)):
		yS, xS = images[x][0].shape
		finList.append([yS,xS])
		finY.append(yS)
		finX.append(xS)


	if(graph == True):
		y_pred = KMeans(n_clusters=3).fit_predict(finList)
		plt.scatter (finY,finX,c=y_pred)
		plt.title("Classification of Segments")
		plt.show()

	kmeans = KMeans(n_clusters=3).fit(finList)

	return kmeans



def getCells(contours,im):

	outImages = []

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		output = im[y:y+h, x:x+w]
		outImages.append([output,y,y+h,x,x+w])

	return outImages

def predict(kms, y,x):
	return kms.predict([[y,x]])


#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

# 1 - recall 0.876
# 
np.random.seed(1)

conts, image = getContours(sys.argv[1])

im2 = cv2.imread(sys.argv[1])

segImages = getCells(conts,image)

kms = kTrain(segImages, True)

for x1 in range(len(segImages)):
	y, x = segImages[x1][0].shape
	if(predict(kms,y,x) == 1):
		var = "   " + str(segImages[x1][1]) + " " + str(segImages[x1][2]) + " " + str(segImages[x1][3]) + " " + str(segImages[x1][4])
		print(var)
		cv2.rectangle(im2,(segImages[x1][3],segImages[x1][1]),(segImages[x1][4],segImages[x1][2]),(255,0,0),1)
	if(predict(kms,y,x) == 0):
		cv2.rectangle(im2,(segImages[x1][3],segImages[x1][1]),(segImages[x1][4],segImages[x1][2]),(0,255,0),1)
	if(predict(kms,y,x) == 2):
		cv2.rectangle(im2,(segImages[x1][3],segImages[x1][1]),(segImages[x1][4],segImages[x1][2]),(0,0,255),1)

cv2.imshow("test", im2)
cv2.waitKey(0)
cv2.imwrite("k_mean_report.png", im2)






