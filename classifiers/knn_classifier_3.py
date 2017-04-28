#!/usr/bin/env python

from sklearn.neighbors import KNeighborsClassifier
# from sklearn.grid_search import RandomizedSearchCV
# from sklearn.grid_search import GridSearchCV
# from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import sys



rawImages = []
features = []
labels = []
theCount = []
counter = 0

def image_to_feature_vector(image, size=(32,32)):
	return cv2.resize(image,size).flatten()

def extract_gray_histogram(image, size=(32,32)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return cv2.resize(hsv,size).flatten()

def build_vectors(argument1, type1):
	imagePaths = list(paths.list_images(argument1))

	global rawImages
	global features
	global labels
	global theCount
	global counter

	for (i, imagePath) in enumerate(imagePaths):
		image = cv2.imread(imagePath)

		pixels = image_to_feature_vector(image)
		hist = extract_gray_histogram(image)

		rawImages.append(pixels)
		features.append(hist)
		labels.append(type1)
		theCount.append(counter)
		counter+=1

def predict(image, model):

	testData1 = [extract_gray_histogram(image)]

	return model.predict_proba(testData1)

def train(infectedpath, uninfectedpath):
	build_vectors(infectedpath, "infected")
	build_vectors(uninfectedpath, "uninfected")

	global rawImages
	global features
	global labels
	global theCount
	global counter	
	rawImages = np.array(rawImages)
	features = np.array(features)
	labels = np.array(labels)
	theCount = np.array(theCount)

	# (trainData, testData, trainLabels, testLabels) = train_test_split(
	# 	features, labels, test_size=0.25, random_state=42)

	# imagePaths = list(paths.list_images(sys.argv[3]))

	# kmeans = KMeans(n_clusters=2).fit(features,labels)
	# params = {"n_neighbors": np.arange(1, 31, 2),
	# 	"metric": ["euclidean", "cityblock"]}

	model = KNeighborsClassifier(n_jobs=-1, n_neighbors = 3, metric = "euclidean")
	# grid = GridSearchCV(model, params)
	start = time.time()

	model.fit(features, labels)

	# print("SCORING")
	# acc = grid.score(testData, testLabels)
	# print(acc*100)
	# print("[INFO] time taken {:.2f} seconds".format(
	# 	time.time() - start))

	return model
	# print("[INFO] grid search best parameters: {}".format(
	# 	grid.best_params_))

	# model2 = KNeighborsClassifier(n_neighbors=2000, n_jobs=-1,metric = 'euclidean')
	# model2.fit(features, labels)


	# for (i, imagePath) in enumerate(imagePaths):
	# 	image2 = cv2.imread(imagePath)

	# 	pixels2 = image_to_feature_vector(image2)
	# 	hist2 = extract_color_histogram(image2)

	# 	acc = model.predict_proba([pixels2])
	# 	print(acc)

	# 	acc = model2.predict([hist2])
	# 	print(acc)


# train("../programs/just_cells/","../gt_extractor/infected/")	