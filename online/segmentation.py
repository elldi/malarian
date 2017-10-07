
#!/usr/bin/env python
import sys,cv2, os
import numpy as np
from random import randint

globY = 0
globX = 0
thres = None
colorThres = None
sys.setrecursionlimit(40000)
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



def init(imagePath, time):

	counter = 0 
	im = cv2.imread(imagePath,0)
	global colorThres
	colorThres = cv2.imread(imagePath)
	final = cv2.imread(imagePath)
	finalShow = cv2.imread(imagePath)


	# Reduce noise in image
	blur = cv2.GaussianBlur(im,(5,5),0)

	# Generate kernel for morphological functions. 
	kernel = np.ones((5,5), np.uint8)

	dilation = cv2.dilate(blur,kernel,iterations = 1)	

	# Opening the image, erosion followed by dilation. 
	opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)

	# Get difference between erosion and dilation of image. 
	# Gradient will generate outline of image.
	gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
	# cv2.imshow("Done", gradient)
	# cv2.waitKey(0)	

	__, th1 = cv2.threshold(gradient,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	global thres
	thres = th1

	global globY
	global globX
	ny ,nx = im.shape
	globY, globX = im.shape

	# myModel = cc.load_pretrained_model("./models/basic_cnn_malaria_10_epochs.h5")


	for y in range(ny):
		for x in range(nx):
			if(thres[y,x] == 255):

				global blue
				global green
				global red

				blue = randint(0,255)
				green = randint(0,255)
				red = randint(0,255)
				counter += 1
				flood(y,x,255,155)


				cv2.rectangle(colorThres, (lowestX[1],lowestY[0]), (biggestX[1],biggestY[0]), (0,255,0))
				cv2.rectangle(finalShow, (lowestX[1],lowestY[0]), (biggestX[1],biggestY[0]), (0,255,0))
				width = abs(lowestX[1] - biggestX[1])
				height = abs(lowestY[0] - biggestY[0])
				cutImage = final[lowestY[0]: lowestY[0] + height, lowestX[1]: lowestX[1] + width]

				
		
				
				clearArrays()
				# cv2.imshow("Done", colorThres)
				# cv2.waitKey(1)

	cv2.imwrite("./output/" + time + ".jpg", finalShow)




