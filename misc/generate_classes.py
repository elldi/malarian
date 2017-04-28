#!/usr/bin/env python
"generate file classes for inception."
import sys, cv2, numpy
import pylab, argparse
import uninfected_extractor as extractor
import segTest4

counter =0 
for x in range(50,101):
	file1 = "../hate_images/"
	if(x== 100):
		file1 += str(x) + ".jpg"
	else:
		file1 += "0" + str(x) + ".jpg" 

	print(file1)

	im2 = cv2.imread(file1)
	finIm = cv2.imread(file1)

	conts, image = segTest4.getContours(file1)

	segImages = segTest4.getCells(conts,image)

	kms = segTest4.kTrain(segImages, False)

	for x in range(len(segImages)):
		counter +=1
		y1, x1 = segImages[x][0].shape
		XW = segImages[x][4]
		YH = segImages[x][2]
		Y = segImages[x][1]
		X = segImages[x][3]

		res = segTest4.predict(kms,y1,x1)

		if(res == 0):
			cv2.imwrite("./class1/"+str(counter)+".jpg",finIm[Y:YH, X:XW])
		if(res == 1):
			cv2.imwrite("./class2/"+str(counter)+".jpg",finIm[Y:YH, X:XW])
		if(res == 2):
			cv2.imwrite("./class3/"+str(counter)+".jpg",finIm[Y:YH, X:XW])
