#!/usr/bin/env python
import sys,cv2
import numpy as np
from random import randint
# import segTest4 as seglib
import masks

globY = 0
globX = 0
thres = None
colorThres = None
sys.setrecursionlimit(30000)
green = 0
red = 0
blue = 0

numOfPix = 0

biggestY = [0,None]
biggestX = [None,0]
lowestY = [256,None]
lowestX = [None,256]

def clearArrays():
	global biggestY
	global biggestX
	global lowestY
	global lowestX
	biggestY = [0,None]
	biggestX = [None,0]
	lowestY = [globY,None]
	lowestX = [None,globX]

def flood(y,x, oldC, newC):
	if(y >= globY or x >= globX):
		return
	if(y < 0 or x < 0):
		return
	if(thres[y,x] != oldC):
		return

	thres[y,x] = newC
	colorThres[y,x,0] = blue
	colorThres[y,x,1] = green
	colorThres[y,x,2] = red

	if(y > biggestY[0]):
		biggestY[0] = y
		biggestY[1] = x
	if(x > biggestX[1]):
		biggestX[0] = y
		biggestX[1] = x
	if(y < lowestY[0]):
		lowestY[0] = y
		lowestY[1] = x
	if(x < lowestX[1]):
		lowestX[0] = y
		lowestX[1] = x

	flood(y+1, x, oldC, newC)
	flood(y-1, x, oldC, newC)
	flood(y, x+1, oldC, newC)
	flood(y, x-1, oldC, newC)
	global numOfPix
	numOfPix += 1

def idCell(image):

	lower_gray = np.array([90,80,80])
	upper_gray = np.array([100,90,90])

	mask_gray = cv2.inRange(image,lower_gray,upper_gray)

	lower_purp = np.array([220,150,150])
	upper_purp = np.array([230,160,160])

	mask_purp = cv2.inRange(image,lower_purp,upper_purp)

	countG = cv2.countNonZero(mask_gray)
	countP = cv2.countNonZero(mask_purp)


	if(countG >1):
		## cv2.imshow("Hopeful",image)
		## cv2.waitKey(0)
		return True
	else:
		return False



fn = sys.argv[1]
im = cv2.imread(fn,0)
colorThres = cv2.imread(fn)
colorThres2 = cv2.imread(fn)


# Reduce noise in image
blur = cv2.GaussianBlur(im,(5,5),0)

# Generate kernel for morphological functions. 
kernel = np.ones((5,5), np.uint8)

dilation = cv2.dilate(blur,kernel,iterations = 1)

# cv2.imshow("Done", dilation)
# cv2.waitKey(0)	

# Opening the image, erosion followed by dilation. 
opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)

# Get difference between erosion and dilation of image. 
# Gradient will generate outline of image.
gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
# cv2.imshow("Done", gradient)
# cv2.waitKey(0)	

__, th1 = cv2.threshold(gradient,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow("Done", th1)
# cv2.waitKey(0)	

# global thres
thres = th1

globY, globX = im.shape
# print(globY)
# print(globX)

ny ,nx = im.shape
globY, globX = im.shape
counter = 0

cutImages = []

for y in range(ny):
	for x in range(nx):
		if(thres[y,x] == 255):

			blue = randint(0,255)
			green = randint(0,255)
			red = randint(0,255)
			counter += 1
			flood(y,x,255,155)
			# print(biggestY) ## South
			# print(biggestX) ## East
			# print(lowestY) ## North
			# print(lowestX) ## West
			# cv2.imshow("Done", colorThres)
			# cv2.waitKey(1)	

			# if(biggestX[0] != None):
			# 	cv2.circle(colorThres, (biggestX[1], biggestX[0]), 3,(255,0,0), -1)
			# if(lowestX[0] != None):
			# 	cv2.circle(colorThres, (lowestX[1], lowestX[0]), 3, (255,0,0), -1)
			# if(biggestY[1] != None):
			# 	cv2.circle(colorThres, (biggestY[1], biggestY[0]), 3,(255,0,0), -1)
			# if(lowestY[1] != None):
			# 	cv2.circle(colorThres, (lowestY[1], lowestY[0]), 3, (255,0,0), -1)

			# if(numOfPix > 2000):
			# 	cv2.rectangle(colorThres, (lowestX[1],lowestY[0]), (biggestX[1],biggestY[0]), (255,0,0))
			# elif(numOfPix < 900):
			# 	cv2.rectangle(colorThres, (lowestX[1],lowestY[0]), (biggestX[1],biggestY[0]), (0,0,255))
			# else:
			# 	cv2.rectangle(colorThres, (lowestX[1],lowestY[0]), (biggestX[1],biggestY[0]), (0,255,0))

			width = abs(lowestX[1] - biggestX[1])
			height = abs(lowestY[0] - biggestY[0])
			cutImage = im[lowestY[0]: lowestY[0] + height, lowestX[1]: lowestX[1] + width]
			cutImage2 = colorThres2[lowestY[0]: lowestY[0] + height, lowestX[1]: lowestX[1] + width]
			cutImages.append([cutImage,lowestY[0], lowestY[0]+height, lowestX[1], lowestX[1]+width])

			if(numOfPix > 200):
				full_mask = masks.get_all_masks(cutImage2)
				cv2.imshow("wow", cutImage2)
				cv2.waitKey(0)
				var = "   " + str(lowestX[1]) + " " + str(lowestY[0]) + " " + str(biggestX[1]) + " " + str(biggestY[0])
				print(var)
			# clearArrays()
			# cv2.imshow("Done", colorThres)
			# cv2.waitKey(1)
			## print(numOfPix)
			numOfPix = 0

# cv2.imshow("Done", colorThres)
# cv2.waitKey(0)	

# print(counter)

# print(biggestY) ## South
# print(biggestX) ## East
# print(lowestY) ## North
# print(lowestX) ## West



# avgH = 0
# avgW = 0
# avgCount = 0

# kms = seglib.kTrain(cutImages,False)
# for element in cutImages:
# 	XW = element[4]
# 	YH = element[2]
# 	Y = element[1]
# 	X = element[3]

# 	y1, x1 = element[0].shape

# 	res = seglib.predict(kms, y1, x1)

# 	if(y1 in range(45,105) or x1 in range(45,105)):
# 		# cv2.rectangle(colorThres2,(X,Y),(XW,YH),(0,255,0,3))
# 		# var = "   " + str(element[1]) + " " + str(element[2]) + " " + str(element[3]) + " " + str(element[4])
# 		# print(var)
# 		if(idCell(colorThres2[Y:Y+y1, X:X+x1])):
# 			var = "   " + str(element[1]) + " " + str(element[2]) + " " + str(element[3]) + " " + str(element[4])
# 			print(var)
# 			cv2.rectangle(colorThres2,(X,Y),(XW,YH),(0,255,0,3))

# 	# # if(res == 0):
# 	# 	# cv2.rectangle(colorThres2,(X,Y),(XW,YH),(0,255,0,3))
# 	# if(res == 1):
# 	# 	# cv2.rectangle(colorThres2,(X,Y),(XW,YH),(255,0,0,3))
# 	# 	# var = "   " + str(element[1]) + " " + str(element[2]) + " " + str(element[3]) + " " + str(element[4])
# 	# 	# print(var)
# 	# 	if(idCell(colorThres2[Y:Y+y1, X:X+x1])):
# 	# 		var = "   " + str(element[1]) + " " + str(element[2]) + " " + str(element[3]) + " " + str(element[4])
# 	# 		print(var)
# 	# 		cv2.rectangle(colorThres2,(X,Y),(XW,YH),(0,0,0))
# 	# if(res == 2):
# 	# 	# cv2.rectangle(colorThres2,(X,Y),(XW,YH),(0,0,255,3))
# 	# 	# var = "   " + str(element[1]) + " " + str(element[2]) + " " + str(element[3]) + " " + str(element[4])
# 	# 	# print(var)
# 	# 	if(idCell(colorThres2[Y:Y+y1, X:X+x1])):
# 	# 		var = "   " + str(element[1]) + " " + str(element[2]) + " " + str(element[3]) + " " + str(element[4])
# 	# 		print(var)
# 	# 		cv2.rectangle(colorThres2,(X,Y),(XW,YH),(0,0,0,3))
# 	# if(res == 3):
# 	# 	# cv2.rectangle(colorThres2,(X,Y),(XW,YH),(0,0,0,3))
# 	# 	# var = "   " + str(element[1]) + " " + str(element[2]) + " " + str(element[3]) + " " + str(element[4])
# 	# 	# print(var)
# 	# 	if(idCell(colorThres2[Y:Y+y1, X:X+x1])):
# 	# 		var = "   " + str(element[1]) + " " + str(element[2]) + " " + str(element[3]) + " " + str(element[4])
# 	# 		print(var)
# 	# 		cv2.rectangle(colorThres2,(X,Y),(XW,YH),(0,0,0))
# cv2.imshow("Done", colorThres2)
# cv2.waitKey(0)

# print(avgH / avgCount)
# print(avgW / avgCount)















