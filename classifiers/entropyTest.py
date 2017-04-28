#!/usr/bin/env python
import sys, cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def entropy(im, showgraph):
	hist, bins = np.histogram(im.ravel(),256,[0,256])

	if(showgraph):
		plt.hist(hist)
		plt.show()

	ent = 0
	for i in hist:
		ent -= i * math.log(i if i>0 else 1)
	return ent

