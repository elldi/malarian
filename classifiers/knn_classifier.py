#!/usr/bin/env python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from imutils import paths
import sys,cv2, os
import numpy as np

def image_to_feature_vector(image, size=(32,32)):
	return cv2.resize(image,size).flatten()

def extract_color_histogram(image, bins=(8,8,8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0,180,0,256,0,256])

	cv2.normalize(hist,hist)

	return hist.flatten()


imagePaths = list(paths.list_images(sys.argv[1]))

rawImages = []
features = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	# Path format /path/to/dataset/{class}.{image_num}.jpg
	label = imagePath.split(".")[1]
	label = label.split("/")[3]

	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)

	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)

	if( i > 0 and i % 1000 == 0):
		print("INFO processed {}/{}".format(i, len(imagePaths)))

rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)

# Training Data
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.75, random_state=42)
# Test data
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=42)

print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=29, n_jobs=-1, metric = 'cityblock')
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))


