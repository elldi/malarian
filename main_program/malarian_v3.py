#!/usr/bin/env python
import os
import sys
f = open("../tmp/dump", 'w')
sys.stdout = f
sys.stderr = f

import cv2
import numpy as np
from random import randint
import cell_classifier as cc
import argparse

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

#------------------------------------------------------------------------------
# Global Variables
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
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

def otsu_gradient(im):
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

	return th1

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("image", help="file path to image for processing")
parser.add_argument("-sv","--save", help="show image processing pipeline",
 action="store_true")
parser.add_argument("-t","--test",type=int, help="select different tests for image")
parser.add_argument("-dir","--directory", help="add option for directories of smears",
 action="store_true")
args = parser.parse_args()

# Check file
try:
    open(args.image, 'r')
except IOError:
    print("IOError: No such file or directory: '" + args.image + "'")
    sys.exit()


im = cv2.imread(sys.argv[1],0)
colorThres = cv2.imread(sys.argv[1])
final = cv2.imread(sys.argv[1])
finalShow = cv2.imread(sys.argv[1])

thres = otsu_gradient(im)


ny ,nx = im.shape
globY, globX = im.shape

myModel = cc.load_pretrained_model(cc.save_loc)

counter_cell = 0.0 
counter_in_cell = 0.0

for y in range(ny):
	for x in range(nx):
		if(thres[y,x] == 255):

			blue = randint(0,255)
			green = randint(0,255)
			red = randint(0,255)
			flood(y,x,255,155)

			if(numOfPix > 2000):
				cv2.rectangle(colorThres, (lowestX[1],lowestY[0]), (biggestX[1],biggestY[0]), (255,0,0))
			elif(numOfPix < 900):
				cv2.rectangle(colorThres, (lowestX[1],lowestY[0]), (biggestX[1],biggestY[0]), (0,0,255))
			else:
				counter_cell += 1
				cv2.rectangle(colorThres, (lowestX[1],lowestY[0]), (biggestX[1],biggestY[0]), (0,255,0))
				width = abs(lowestX[1] - biggestX[1])
				height = abs(lowestY[0] - biggestY[0])
				cutImage = final[lowestY[0]: lowestY[0] + height, lowestX[1]: lowestX[1] + width]

				cutImage = cv2.resize(cutImage, (cc.img_width, cc.img_height))
				result = cc.predict_me(myModel, cutImage)

				if(result[0][0] == 0):
					cv2.rectangle(finalShow, (lowestX[1],lowestY[0]), (biggestX[1],biggestY[0]), (0,0,0))
					var = "   " + str(lowestX[1]) + " " + str(lowestY[0]) + " " + str(biggestX[1]) + " " + str(biggestY[0])
					print(var)
					counter_in_cell += 1
						
			clearArrays()
			numOfPix = 0


if(args.save):
	cv2.imwrite("malaria_output.png", finalShow)
	cv2.imwrite("seg_output.png", colorThres)

inf_rate = counter_in_cell / counter_cell * 100

print(str(inf_rate) + "% infection rate.")

if(inf_rate > 2.0):
	print("High possibility of infection with malaria. Patient must seek further" +
		" medical attention immediately.")
elif(inf_rate > 0.0):
	print("Small number of infected cells detected, retake blood or produce new smear, then conduct" +
		"computer test again. ")
if(inf_rate == 0.0):
	print("Smear is clear of parasites associated with malaria.")
