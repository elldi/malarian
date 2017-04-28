#!/usr/bin/env python
import sys,cv2
import numpy as np

def dark_purple(im):
	lower_dpurp = np.array([90,20,35])
	upper_dpurp = np.array([100,30,45])
	mask_purp = cv2.inRange(im,lower_dpurp,upper_dpurp)
	countP = cv2.countNonZero(mask_purp)
	return countP, mask_purp

def light_purple(im):
	lower_dpurp = np.array([128,52,90])
	upper_dpurp = np.array([140,65,110])
	mask_purp = cv2.inRange(im,lower_dpurp,upper_dpurp)
	countP = cv2.countNonZero(mask_purp)
	return countP, mask_purp

def mid_purple(im):
	lower_dpurp = np.array([117,48,60])
	upper_dpurp = np.array([128,58,75])
	mask_purp = cv2.inRange(im,lower_dpurp,upper_dpurp)
	countP = cv2.countNonZero(mask_purp)
	return countP, mask_purp

def mist_purple(im):
	lower_dpurp = np.array([115,70,112])
	upper_dpurp = np.array([130,82,92])
	mask_purp = cv2.inRange(im,lower_dpurp,upper_dpurp)
	countP = cv2.countNonZero(mask_purp)
	return countP, mask_purp
def black(im):
	lower_dpurp = np.array([10,10,10])
	upper_dpurp = np.array([25,25,25])
	mask_purp = cv2.inRange(im,lower_dpurp,upper_dpurp)
	countP = cv2.countNonZero(mask_purp)
	return countP, mask_purp
def off_white(im):
	lower_dpurp = np.array([160,160,165])
	upper_dpurp = np.array([172,172,177])
	mask_purp = cv2.inRange(im,lower_dpurp,upper_dpurp)
	countP = cv2.countNonZero(mask_purp)
	return countP, mask_purp

def get_all_masks(im):

	# im = cv2.imread(sys.argv[1])

	c1, mask1 = dark_purple(im)
	c2, mask2 = light_purple(im)
	c3, mask3 = mid_purple(im)
	c4, mask4 = mist_purple(im)
	c5, mask5 = black(im)
	c6, mask6 = off_white(im)

	mask7 = mask1 | mask2 | mask3 | mask4 | mask5 | mask6

	return c1+c2+c3+c4+c5+c6

# cv2.imshow("Batmans Rubber Mask", mask7)
# cv2.waitKey(0)

