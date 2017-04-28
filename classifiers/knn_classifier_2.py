#!/usr/bin/env python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import sys

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


(trainData, testData, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=42)

# construct the set of hyperparameters to tune
params = {"n_neighbors": np.arange(1, 31, 2),
	"metric": ["euclidean", "cityblock"]}

# tune the hyperparameters via a cross-validated grid search
print("[INFO] tuning hyperparameters via grid search")
model = KNeighborsClassifier(n_jobs=-1)
grid = GridSearchCV(model, params)
start = time.time()
grid.fit(trainData, trainLabels)
 
# evaluate the best grid searched model on the testing data
print("[INFO] grid search took {:.2f} seconds".format(
	time.time() - start))
acc = grid.score(testData, testLabels)
print("[INFO] grid search accuracy: {:.2f}%".format(acc * 100))
print("[INFO] grid search best parameters: {}".format(
	grid.best_params_))

# tune the hyperparameters via a randomized search
grid = RandomizedSearchCV(model, params)
start = time.time()
grid.fit(trainData, trainLabels)
 
# evaluate the best randomized searched model on the testing
# data
print("[INFO] randomized search took {:.2f} seconds".format(
	time.time() - start))
acc = grid.score(testData, testLabels)
print("[INFO] grid search accuracy: {:.2f}%".format(acc * 100))
print("[INFO] randomized search best parameters: {}".format(
	grid.best_params_))


