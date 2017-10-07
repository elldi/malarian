#!/usr/bin/env python
import sys,cv2, os
import numpy as np
import time
import io
import zipfile
import segmentation as seg
from random import randint
from flask import Flask
from flask import request
from flask import send_file 
from flask import send_from_directory
app = Flask(__name__)




@app.route('/malarian',methods=['POST'])
def malarian():
	if(request.method == 'POST'):
		f = request.files['image']

		myTime = str(time.time())
		fileLoc = "./transf/"+ myTime +".jpg"
		f.save(fileLoc)

		seg.init(fileLoc, myTime)

		return "/malarian/" + myTime + ".jpg"
	else:
		return "Only accepts post"

@app.route('/malarian/<string:filename1>',methods=['GET'])
def getimage(filename1):
	return send_from_directory(directory='output', filename=filename1, as_attachment=True)
