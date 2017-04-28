#!/usr/bin/env python
import sys, cv2, os
"""
Program to extract infected blood cells from ground truth
"""


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

		myFile = open(groundTruth, "r")
		for line in myFile:
			line = line[:len(line)-1] ## Remove EOL terminator
			lineList = line.split(" ")
			im = cv2.imread(cleanImage)
			lineList = map(int, lineList)
			print lineList 

			output = im[lineList[1]:(lineList[1]+(lineList[3] - lineList[1])), lineList[0]:(lineList[0]+(lineList[2] - lineList[0]))]
			cv2.imshow("Hopeful", output)
			cv2.waitKey(1)

			cv2.imwrite("./infected/"+str(counter)+".jpg", output)
			counter +=1




if(len(sys.argv) != 3):
	print "2 arguments are needed for the program!"
	print len(sys.argv)-1, "have been given."
	print "Usage:", sys.argv[0], " <GROUND TRUTH DIRECTORY> <DIRECTORY CONTAINING CLEAN RBC IMAGES>" 
else:
	fileReader(getValidFiles())
