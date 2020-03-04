##Giving OS Permissions
import os
import sys
import subprocess
if os.geteuid() != 0:
    subprocess.call(['sudo', 'python3'] + sys.argv)

import cv2
import IRCam
import RGBCam
import saliency 
import numpy as np
from time import time
import concurrent.futures

# Setting the camera functions
IRCam = IRCam.SeekPro()
RGBCam = RGBCam.PiCam()
saliency = saliency.findSaliency()

#Initialise Camera Windows
cv2.namedWindow("Seek",cv2.WINDOW_NORMAL)
cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)

#class droneCam():
def thermImageProc(): 
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
#      IRColor = cv2.applyColorMap(IRRot, cv2.COLORMAP_HOT)

      return IRRot

def visImageProc():
      #Get Visual Image
      RGB = RGBCam.frameCapture()
      return RGB


if __name__ == '__main__':
    IRImg = thermImageProc()
    prevRGBImg = visImageProc()
    
    t = 0
    t0 = 0
    while True:
          t = time() 
          print("fps:",1/(t-t0))
          t0 = time()
    #
          RGBImg = visImageProc()
    ##     fusedImg = IRImg + RGBImg
    #      if (IRImg.all() != thermImageProc().all()):
          IRImg = thermImageProc()
          
        
      #concurrent.futures.as_completed()
          #displaying the rotated thermal image
          cv2.imshow("Seek",IRImg)
          cv2.imshow("RGB", RGBImg)
    #     cv2.imshow("fused",fusedImg)
    
          ##Saliency 
          RGBSal = saliency.getSaliency(RGBImg)[0]
          cv2.imshow("Salient",RGBSal)
    
          prevIRImg = IRImg
          prevRGBImg = visImageProc()
          if cv2.waitKey(1) & 0xFF == ord('q'):
                break
          cv2.waitKey(1)






# ATTEMPT AT THREADING 
#    with concurrent.futures.ThreadPoolExecutor() as executor:
#        IRImg = executor.submit(thermImageProc())
#        RGBImg = executor.submit(visImageProc())
#        
#        IRImg = IRImg.result()
#        #concurrent.futures.as_completed(IRImg)
#        RGBImg = RGBImg.result()
#        #concurrent.futures.as_completed(RGBImg)
