import IRCam
import RGBCam
import cv2
import numpy as np
from time import time

# Setting the camera functions
IRCam = IRCam.SeekPro()
RGBCam = RGBCam.PiCam()

t0 = time()

#Initialise Camera Windows
cv2.namedWindow("Seek",cv2.WINDOW_NORMAL)
cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)

while True:
    t = time()
    print("fps:",1/(t-t0))
    t0 = time()
    
    #Get thermal image 
    IR = IRCam.get_image()
    #convert the raw data into 0-255 grayscale range
    IRdisp = IRCam.rescale(IR)
    
    #Rotation of the thermal image by 180 degrees
    #This takes up 5fps of processing time
    (h, w) = IRdisp.shape[:2]
    centre = (w/2,h/2)
    IRRotMat = cv2.getRotationMatrix2D(centre, 180, 1)
    IRRot = cv2.warpAffine(IRdisp, IRRotMat,(w, h))
    #applying themal image colourmap
    
    #Get Visual Image
    RGB = RGBCam.frameCapture()
    
    #displaying the rotated thermal image
    cv2.imshow("Seek",IRRot)
    cv2.imshow("RGB", RGBCam.frameCapture())

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break
      
    cv2.waitKey(1)

