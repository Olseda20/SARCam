# Original Author: Omar Ali

## giving OS Permissions
import os
import sys
import subprocess
if os.geteuid() != 0:
  subprocess.call(['sudo', 'python3'] + sys.argv)

import cv2
import IRCam
import RGBCam
import Saliency 
import numpy as np
from time import time

# setting the camera functions
IRCam = IRCam.SeekPro()
RGBCam = RGBCam.PiCam()
saliency = Saliency.findSaliency()

# initialise Camera Windows
cv2.namedWindow("Seek",cv2.WINDOW_NORMAL)
cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)

# initialise variables
RESOLUTION = (320,240)

def thermImageProc(): 
  """
   function to get and preprocess thermal camera image into droneCam
  """
  # get thermal image 
  IR = IRCam.get_image()
  # convert the raw data into 0-255 grayscale range
  IRrescale = IRCam.rescale(IR)
  # rotation of the thermal image by 180 degrees
  IRRot = imageRotation(IRrescale,180)
  # applying themal image colourmap
  # IRColor = cv2.applyColorMap(IRRot, cv2.COLORMAP_HOT)
  return IRRot

def imageRotation(img,angle):
  """
   function to rotate a given image with the provided angle
  """
  (h, w) = img.shape[:2]
  centre = (w/2,h/2)
  RotMat = cv2.getRotationMatrix2D(centre, angle, 1)
  RotImg = cv2.warpAffine(img, RotMat,(w, h))
  return RotImg

#Function to get and preprocess visual camera image into droneCam
def visImageProc():
  """
   function to get and preprocess visual camera image into droneCam
  """
  #Get Visual Image
  RGB = RGBCam.frameCapture()
  return RGB

def perspTForm(img):
  """
   perspective transformation to allow for mapping of the thermal image onto the visual image
  """
  #Transforming the thermal image to overlay the thermal over the visual
  pts1 = np.array([[14, 8], [103, 161], [294, 20], [278, 174]]) #the points on the thermal image
  pts2 = np.array([[48, 25], [99, 118], [232, 25], [216, 124]]) #the points being mapped onto on the visual image

  # Configured values
  M = cv2.getPerspectiveTransform(pts1,pts2)
  # M = np.array([[0.589,-0.0392,0],[-0.0051,0.6065,0],[40.31,22,1]])
  # M = np.array([[1.699,0.0944,0],[0.0207,1.6813,0],[-69.02,-40.83,1]])
  print(M)

  dst = cv2.warpPerspective(img, M, RESOLUTION)
  return dst 

if __name__ == '__main__':
  from time import strftime
  from time import sleep

  IRImg = thermImageProc()
  prevRGBImg = visImageProc()
  t = 0
  t0 = 0

  while True:
    t = time() 
    print("fps:",1/(t-t0))
    t0 = time()
    RGBImg = visImageProc()
    fusedImg = IRImg + RGBImg
    # if (IRImg.all() != thermImageProc().all()):
    IRImg = thermImageProc()

    #displaying the calibrated and rotated images
    cv2.imshow("Seek",IRImg)
    cv2.imshow("RGB", RGBImg)
    newImg = perspTForm(IRImg)
    print(newImg)
    cv2.imshow("img",newImg)
    cv2.imshow("fused",fusedImg)

    ## Saliency 
    RGBSal = Saliency.getSaliency(RGBImg)[0]
    cv2.imshow("Salient",RGBSal)
    # timestr = strftime("%d%m%Y-%H%M%S")
    # cv2.imwrite(f'/home/pi/SARCam/Vis&Therm/{timestr}VisualImg.png', RGBImg)
    # cv2.imwrite(f'/home/pi/SARCam/Vis&Therm/{timestr}ThermalImg.png', IRImg)

    prevIRImg = IRImg
    prevRGBImg = visImageProc()
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    cv2.waitKey(1)

    sleep(0.5)
