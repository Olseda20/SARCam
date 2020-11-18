##Giving OS Permissions to the script for the thermal camera to function
import os, sys, subprocess
if os.geteuid() != 0:
    subprocess.call(['sudo', 'python3'] + sys.argv)

#libraries for general image/data manipulation
import cv2
import numpy as np
# from imutils import contors
import imutils

#importing the saleincy library
from saliency import findSaliencyFineGrained, findSaliencySpectralResidual

#other general libraries
from time import time
import datetime
import concurrent.futures

#For visual cameras capture
import RGBCam
#For thermal camera capture
import IRCam


##Accessing the camera classes
PiCam = RGBCam.PiCam()
SeekPro = IRCam.SeekPro()
WebCam = RGBCam.WebCam(src=1).start()

##Getting the Saleincy functions from the saleincy script
findSaliencyFineGrained = findSaliencyFineGrained()
findSaliencySpectralResidual = findSaliencySpectralResidual()
contourProcessing = contourProcessing()
#setting the threshold for the saliency algorithm and initiating the saleincy counter
threshold = 65
saliencyCount = 1

##Calibration for the thermal camera
SeekProCalib    = np.loadtxt('/home/pi/SARCam/CameraCalibration/Images/SeekPro/cameraMatrix.txt', delimiter=',')
SeekProDist     = np.loadtxt('/home/pi/SARCam/CameraCalibration/Images/SeekPro/cameraDistortion.txt', delimiter=',')
SeekProUnDisMat, roiSeekPro = cv2.getOptimalNewCameraMatrix(SeekProCalib,SeekProDist,(RESOLUTION),1,(RESOLUTION))

##Loading in the moving and fixed points of corresponding points between the thermal and visual image
movingPoints    = np.loadtxt('/home/pi/SARCam/movingPoints.txt', delimiter=',')
fixedPoints     = np.loadtxt('/home/pi/SARCam/fixedPoints.txt', delimiter=',')

##Generating the perspective transformation matrix using the moving and fixed points
tform, status = cv2.findHomography(movingPoints, fixedPoints)

purpMap = np.loadtxt('/home/pi/SARCam/purpColCustom3.txt', delimiter=',').astype(int)


class imageModification():
    '''class to apply image transfomrations and rotations'''
    def imageRotation(self, image, angle): 
        ''' Rotates the image by a desired angle incase cameras are flipped'''
        (h, w) = image.shape[:2]
        centre = (w/2,h/2)
        RotMat = cv2.getRotationMatrix2D(centre, angle, 1)
        RotImg = cv2.warpAffine(image, RotMat,(w, h))
        return RotImg

    def imageCropping(self, img, roi):
        ''' function to crop the image to the centre of the thermal image  '''
        ## Cropping regions that are not in the image after, correction
        x,y,w,h = roi
        img = img[y:y+h, x:x+w]
        return img

    def customColourMap(self, img, LUT=purpMap):
        newImg = np.zeros((img.shape[0],img.shape[1],3)).astype(int)
        if img.shape[3] != 1:
            print(f'the image shape is incorrect for the custom colormap')
            return 

        for y in range (0,img.shape[0]):
            for x in range (0, img.shape[1]):
                newImg[y][x] = LUT[img[y][x]]

        return newImg
    
    def imageFusion(self, background, foreground):
        if background.shape == foreground.shape:
            ## Adds the images together (has a transarancy factor)
            fusedImg = cv2.addWeighted(background,1,foreground,0.5,0)        
            return fusedImg
        else :
            print('shape of background image and foreground are not the same')
            print(f'Background shape is :{background.shape}')
            print(f'Foreground shape is :{foreground.shape}')

    def saliencyGimbalCommand(self,img, threshold):
        #getting the saleincy maps from the openCV saliency Models
        (saliencyMap, threshMap) = findSaliencyFineGrained.getSaliency(img,threshold)
        (saliencyMap2, threshMap2) = findSaliencySpectralResidual.getSaliency(img,threshold)

        #getting the contours from the spectral residual image 
        (sortedImage, boundingBoxes) = contourProcessing.getContours(threshMap, saliencyMap2)
        gimbalMove = findSaliency.getContorPosition(sortedImage,boundingBoxes)
        return (saliencyMap, saliencyMap2, threshMap, threshMap2, sortedImage, boundingBoxes, gimbalMove)

    def skinDetector(self,img):
        '''Function to identify the skin of a person in an image'''
        #Skinmask HSV range
        lower = np.array([0, 48, 80], dtype = "uint8")
        upper = np.array([30, 255, 255], dtype = "uint8")

        # resize the frame, convert it to the HSV color space,
        # and determine the HSV pixel intensities that fall into
        # the speicifed upper and lower boundaries
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)

        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (7, 7), 0)
        skin = cv2.bitwise_and(frame, frame, mask = skinMask)

        return skin

    def thermHumanDetector(self, img):
        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (7, 7), 0)
        skin = cv2.bitwise_and(frame, frame, mask = skinMask)
        return skin


class CameraProcessing():

    def PiCamPost(self):
        '''post processing for the PiCam'''
        global WideImg
        ## Grabbing PiCam Image
        WideImg = PiCam.frameCapture()

        ## Image Processing region for the Wide Camera
        #PiCam image rotation of 180 degrees
        WideImg = imageModification().imageRotation(WideImg,180)
        return

    def WebCamPost(self):
        '''post processing for the WebCam'''
        global NarrowImg
        ##Grabbing WebCam Image 
        NarrowImg = WebCam.read()

        #rescaling the image to default size
        NarrowImg = imutils.resize(NarrowImg, width=320)
        return 

    def SeekProPost(self):
        '''post processing for the SeekPro before introduced'''
        global IRImg
        global IRWarp
        ##Grabbing SeekPro Image
        IRImg = SeekPro.rescale(SeekPro.get_image())

        ##Image processing of thermal image
        #Revoming image distortion
        IRWarp = cv2.undistort(IRImg, SeekProCalib, SeekProDist, None, SeekProUnDisMat)
        #Applying Perspective transformation matrix to overlay thermalimage onto visual
        IRWarp = cv2.warpPerspective(IRImg,tform,(RESOLUTION))
        return 

if __name__ == '__main__':
    from time import time
    from time import strftime
    from time import sleep
    import os
    import concurrent.futures

    t0 = time()

    def saliencyImplement(img, threshold):
        '''implementing the OpenCV saliency function from the saleincyCV library'''
        try:  
            saliencyMap, saliencyMap2, threshMap, threshMap2, sortedImage, boundingBoxes, gimbalMove = imageModification().saliencyGimbalCommand(img, threshold)

        except:
            print('GimbalMovement failed')

    def thermalSmoothing(img):
        '''function to apply smoothing to the thermal image'''
        IRsmooth = cv2.medianBlur(img,3)        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        IRsmooth = cv2.erode(IRsmooth, kernel, iterations = 3)
        IRsmooth = cv2.dilate(IRsmooth, kernel, iterations = 3)
        IRsmooth = cv2.medianBlur(IRsmooth,5)
        IRsmooth = cv2.medianBlur(IRsmooth,5)
        IRsmooth = cv2.medianBlur(IRsmooth,3)        
        return IRsmooth 

    def flooding(img):
        '''function to find the edged of an thermak object and fill it with white'''
        edges = cv2.Canny(img,50,255)
        
        # Copy the thresholded image.
        im_floodfill = edges.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = edges.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = edges | im_floodfill_inv
        return im_out

    while True:
        #Showing the framrate of the system
        t = time()
        print("fps:",1/(t-t0))
        t0 = time()

        #Images being captured using threads to prevent bottlenecks within image capture stag
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(CameraProcessing().SeekProPost())
            executor.submit(CameraProcessing().PiCamPost())
            executor.submit(CameraProcessing().WebCamPost())
        IR1 = IRImg
        RGB1 = WideImg
        RGB2 = NarrowImg
        IRWarp = IRWarp


        IRColor = cv2.applyColorMap(IRWarp, cv2.COLORMAP_RAINBOW)
        IRFused = imageModification().imageFusion(RGB1,IRColor)

        ##Applying the colourmap
        IRColor = cv2.applyColorMap(IRWarp, pupMat)
        IRFusedCropped = imageModification().imageCropping(IRFused, (47,55,190,140))

        ##Implementing Saliency
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(saliencyImplement(RGB1, 55))
       
       ##Choosing the images to show
        # IRColor = imageModification().customColourMap(IR1,purpMap)
        cv2.imshow("piCam", RGB1)       #Showing the PiCam Image
        # cv2.imshow("webCam", RGB2)    #Showing the WebCam Image
        cv2.imshow("seekPro",IR1)       #Showing the SeekPro Image
        cv2.imshow("seekProWarp",IRWarp)#Showing the transformed SeekPro Image
        cv2.imshow("irColor", IRColor)  #Showing the SeekPro with colourmap
        # cv2.imshow("Fused", IRFused)  #Showing the fused PiCam + SeekPro image


        ##Exit Code
        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Exit key 'q' has been pressed")
            WebCam.stop()
            PiCam.stop()
            break
        cv2.waitKey(1)


    # CameraProcessing.ReleaseCameras()
    WebCam.stop()
    PiCam.stop()
    cv2.destroyAllWindows()

    print("[INFO] RGB Cameras Released √")
    # return 



# pi@raspberrypi:~