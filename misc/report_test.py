#!/usr/bin/env python

import cv2, sys
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(sys.argv[1], 0)
equ = cv2.equalizeHist(img)

hist,bins = np.histogram(equ.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'g')
plt.hist(equ.flatten(),256,[0,256], color = 'b')
plt.xlim([0,256])
plt.show()

# equ = cv2.equalizeHist(img)

cv2.imwrite("h_e.png", equ)
# cv2.waitKey(0)

# cv2.imwrite("grayscale.png", img)

# blur = cv2.GaussianBlur(equ,(5,5),0)
# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# cv2.imwrite("otsus.png", th3)