#!/usr/bin/env python
"The start of an automated malaria diagnosis system"
import sys, cv2, os
import numpy as np


#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
# If an argument hasn't been entered into the program
if len (sys.argv) < 2:
    print >>sys.stderr, "Usage:", sys.argv[0], "<image>..."
    sys.exit (1)

# Process the files given on the command line.
for fn in sys.argv[1:]:
    # Read in the image and print out its dimensions.
    im = cv2.imread (fn)

    blur = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(blur,(5,5),0)
    ret, th1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ## cv2.imwrite("otsu.png", th1)


    im2,contours,hierarchy = cv2.findContours(th1, 1, 2)
    index = 1


    x = fn.split("/")[2].split(".")[0]
    newDir = x + "-output"
    os.makedirs(newDir)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        output = im[y:y+h, x:x+w]

        ySize, xSize, cSize = output.shape

        if(ySize < 100 and ySize > 50):
            cv2.imshow("Nearly", output)
            cv2.waitKey(3)
            cv2.imwrite( "./" + newDir + "/"+ str(index) + ".jpg", output)
            index += 1
            # cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 3)

    print index-1
        # ignore images of a certain size. 



    # cv2.imshow("Image", im)
    # cv2.imwrite("BoxedImage.png", im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()






    
	
#------------------------------------------------------------------------------
# End of Malarian.
#------------------------------------------------------------------------------
