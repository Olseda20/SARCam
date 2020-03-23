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

import scipy.io as sio

from imutils import contours
import imutils

#Image displaying (the more purple the hotter)
LUTmat = sio.loadmat('PurpLUT.mat')
purpLUT = LUTmat['purpCol']

'''Spot to define all the the calibration and
   the perspective transformation data so it is all here and not in many regions'''
## RGB Calibration information
#Calibration matrix 
RGBtx = np.array([[626.80,0,85.301],[0,553.14,130.05],[0, 0, 1]])
#Distortion matrix
RGBdist = np.array([[0.23819,0.80864,0.024662,-0.18100,-0.36040]])

## IR Calibration Information
#Calibration matrix 
IRmtx = np.array([[639.06, 0, 135.45],[0, 637.74, 99.42],[0, 0, 1]])
#Distortion matrix 
IRdist = np.array([[-1.1945, 24.321, -0.00598, 0.01358, -0.02011]])

##Perspective Transfomration Matrix 
#(Found the points toalign the images using the cpselect tool in matlab)
thermPoints = np.array([[11, 6],[34, 226],[296, 223],[296,19],[101,160]])
visPoints = np.array([[49, 24],[57, 158],[232,156],[232,26],[99,117]])

# Setting the camera functions
IRCam = IRCam.SeekPro()
RGBCam = RGBCam.PiCam()
sal = saliency.findSaliency()

#Initialise Camera Windows
cv2.namedWindow("Seek",cv2.WINDOW_NORMAL)
cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)

#class droneCam():
def imageRotation(img,angle): 
      ''' Rotates the image by a desired angle incase cameras are flipped'''
      (h, w) = img.shape[:2]
      centre = (w/2,h/2)
      RotMat = cv2.getRotationMatrix2D(centre, angle, 1)
      RotImg = cv2.warpAffine(img, RotMat,(w, h))
      return RotImg

def imgUndistort(self, img, mtx, dist):
      '''Applying image calibration to remove image distortion on the visual
      visual image. 
      Note: The calibration distortion matrices have been determined using 
      a calibration script external to this script this will need to be 
      manually placed in after evaluation'''
      #Getting the image shape
      h,  w = img.shape[:2]
      #Using the calibration and distortion matrix generate the corrected image
      #with croppable regions to prevent epty regions in the image
      CorrectionMtx, imgCrop = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

      #Applying the image calibration
      imgUndistorted = cv2.undistort(img, mtx, dist, None, CorrectionMtx)

      ##Cropping the image using imCrop
      x,y,w,h = imgCrop
      imgCropped = imgUndistorted[y:y+h, x:x+w]
      return imgUndistored, imgCropped

#Perspective Transformation to allow for mapping of the thermal image onto the visual image
def perspTForm(img, movingPoints, fixedPoints):
      ## transformation matrix to align the thermal images onto thevisual image
      tform, status = cv2.findHomography(movingPoints, fixedPoints)
      ## warping the thermal image to map onto the visual image 
      dst = cv2.warpPerspective(img,tform,(320,240))#,cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 255)
      return dst 
  
#Function to get and preprocess thermal camera image into droneCam
def thermImageProc(): 
      ##Get thermal image 
      IR = IRCam.get_image()
      ##convert the raw data into 0-255 grayscale range
      IRrescale = IRCam.rescale(IR)
      ##Rotation of the thermal image by 180 degrees
      IRRot = imageRotation(IRrescale,180)
      ##applying themal image colourmap
      IRtform = perspTForm(IRRot, thermPoints, visPoints)
      # IRColor = cv2.applyColorMap(IRtform, cv2.COLORMAP_HSV)

      # IRColor = cv2.LUT(IRtform, purpLUT)
      return IRtform #, IRColor



#Function to get and preprocess visual camera image into droneCam
def visImageProc():
      #Get Visual Image
      RGB = RGBCam.frameCapture()
      RGB = imageRotation(RGB,180)
      return RGB

#Function to overlat the thermal and the visual images ontop of each other by the time it is in this functino
#the images should already be able to overlay ontop of each other
def imageFusion(background, foreground):
    if background.shape == foreground.shape:
        ## Adds the images together (has a transarancy factor)
        fusedImg = cv2.addWeighted(background,1,foreground,0.7,0)        
        return fusedImg
    else :
        print('shape of background image and foreground image might not be the same')
        print(f'Background shape is :{background.shape}')
        print(f'Foreground shape is :{foreground.shape}')


if __name__ == '__main__':
      from time import strftime
      from time import sleep
      from time import time

      prevIRImg = thermImageProc()
      prevRGBImg = visImageProc()
      orig = []

      t = 0
      t0 = 0
      while True:
            t = time() 
            print("fps:",1/(t-t0))
            t0 = time()
            RGBImg = visImageProc()
            orig = RGBImg.copy()
            #  attempt at improving efficiency   if (IRImg.all() != thermImageProc().all()):
            IRImg = prevIRImg
            #displaying the calibrated and rotated images
            cv2.imshow("Seek",IRImg)
            cv2.imshow("RGB", RGBImg)

            ## Fusing the images
#            IRCol = IRImg[1]
#            fusedImg = imageFusion(RGBImg, IRCol)
#            cv2.imshow("fused",fusedImg)
            
            # Saliency 
#            RGBSalThresh = sal.getSaliency(RGBImg)[1]
#            cv2.imshow("Salient",RGBSalThresh)
#
#
##            threshIR = cv2.Canny((IRImg[0]),40,200)
##            cv2.imshow("threshIR",threshIR)
#            
#             try:
#                 # find contours in the edge map
#                 cnts = cv2.findContours(RGBSalThresh.copy(), cv2.RETR_EXTERNAL,
#                 	cv2.CHAIN_APPROX_SIMPLE)
#                 cnts = imutils.grab_contours(cnts)
                
#     #            orig = RGBImg.copy
#                 # loop over the (unsorted) contours and label them
#                 for (i, c) in enumerate(cnts):
#                     	orig = contours.label_contour(orig, c, i, color=(240, 0, 159))
    
#                 # # loop over the sorting methods
#                 for method in ("left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"):
#                     	# sort the contours
#                     	(cnts, boundingBoxes) = contours.sort_contours(cnts, method=method)
#                     	clone = RGBImg.copy()                    
#                     	# loop over the sorted contours and label them
#                     	for (i, c) in enumerate(cnts):
#                     		sortedImage = contours.label_contour(clone, c, i, color=(240, 0, 159))
                    
#                     	# show the sorted contour image
#                     	cv2.imshow(method, sortedImage)
#             except:
#                 print('some error')
            
            
            
            
            ## Saving Image Data
            # timestr = strftime("%d-%m-%Y:%H-%M-%S-")
            # ms = str(int(round(time()*1000000)))[-2:]
            # cv2.imwrite(f'/home/pi/Desktop/Field Testing/Visual/{timestr}{ms}VisualImg.jpg', RGBImg)
            # cv2.imwrite(f'/home/pi/Desktop/Field Testing/Thermal/{timestr}{ms}ThermalImg.jpg', IRImg)


            # prevIRImg = IRImg
            # prevRGBImg = visImageProc()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
            cv2.waitKey(1)
            prevIRImg = thermImageProc()

            ## To control the speed at which images are taken
            # sleep(0.5)






# ATTEMPT AT THREADING 
#    with concurrent.futures.ThreadPoolExecutor() as executor:
#        IRImg = executor.submit(thermImageProc())
#        RGBImg = executor.submit(visImageProc())
#        
#        IRImg = IRImg.result()
#        #concurrent.futures.as_completed(IRImg)
#        RGBImg = RGBImg.result()
#        #concurrent.futures.as_completed(RGBImg)
