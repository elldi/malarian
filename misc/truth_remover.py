#!/usr/bin/env python
import sys, cv2, os
import numpy as np
"""
Program to extract infected blood cells from ground truth
"""

image = None 

def black_out(list1):
	ny, nx = image.shape

	global image

	for y in range(list1[1], list1[3]):
		for x in range(list1[0], list1[2]):
			image[y,x] = 0



def getValidFiles():
	items = os.listdir(sys.argv[1])
	check = 0
	fileList = []
	for fileName in items:
		if(fileName[len(fileName)-3:] == ".gt"):
			check+=1
			fileList.append(fileName)
	if(check == 0):
		print "No ground truth files were found in the directory ",sys.argv[1] 
	return fileList

def fileReader(theList):
	counter = 0
	for goodFile in theList:

		groundTruth = sys.argv[1] + goodFile
		cleanImage = sys.argv[2] + goodFile[:len(goodFile)-3]
		print cleanImage

		im = cv2.imread(cleanImage,0)
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
		global image
		image = th1

		myFile = open(groundTruth, "r")
		for line in myFile:
			line = line[:len(line)-1] ## Remove EOL terminator
			lineList = line.split(" ")

			lineList = map(int, lineList)
			print lineList 

			black_out(lineList)


		cv2.imwrite("./removed/0"+str(counter+52)+".jpg", image)
		counter +=1




if(len(sys.argv) != 3):
	print "2 arguments are needed for the program!"
	print len(sys.argv)-1, "have been given."
	print "Usage:", sys.argv[0], " <GROUND TRUTH DIRECTORY> <DIRECTORY CONTAINING CLEAN RBC IMAGES>" 
else:
	fileReader(getValidFiles())
