#NEW CAMERA SCRIPT

##Giving OS Permissions
import os, sys, subprocess
if os.geteuid() != 0:
    subprocess.call(['sudo', 'python3'] + sys.argv)

#libraries for general image/data manipulation
import cv2
import numpy as np
# from imutils import contors
import imutils

from saliency import findSaliency, findSaliency2

#other general libraries
from time import time
import datetime
import concurrent.futures

#custom scripts for image capture and post processing
import RGBCam
import IRCam

RESOLUTION = 320,240


RGBCam.cam_initialise
PiCam = RGBCam.PiCam()
# WebCam = RGBCam.WebCam()
WebCam = RGBCam.WebCam(src=1).start()
SeekPro = IRCam.SeekPro()


findSaliency = findSaliency()
findSaliency2 = findSaliency2()

threshold = 65
saliencyCount = 1

#Calibration for the thermal camera
SeekProCalib    = np.loadtxt('/home/pi/SARCam/CameraCalibration/Images/SeekPro/cameraMatrix.txt', delimiter=',')
SeekProDist     = np.loadtxt('/home/pi/SARCam/CameraCalibration/Images/SeekPro/cameraDistortion.txt', delimiter=',')
SeekProUnDisMat, roiSeekPro = cv2.getOptimalNewCameraMatrix(SeekProCalib,SeekProDist,(RESOLUTION),1,(RESOLUTION))

movingPoints    = np.loadtxt('/home/pi/SARCam/movingPoints.txt', delimiter=',')
fixedPoints     = np.loadtxt('/home/pi/SARCam/fixedPoints.txt', delimiter=',')
tform, status = cv2.findHomography(movingPoints, fixedPoints)

purpMap = np.loadtxt('/home/pi/SARCam/purpMap.txt', delimiter=',').astype(int)
# RESOLUTIONWEBCAM = 960,760
# #Calibration for the narrow (web) camera 
# WebCamCalib = np.loadtxt('/home/pi/SARCam/CameraCalibration/Images/WebCam/cameraMatrix.txt', delimiter=',')
# WebCamDist = np.loadtxt('/home/pi/SARCam/CameraCalibration/Images/WebCam/cameraDistortion.txt', delimiter=',')
# WebCamUnDisMat, roiWebCam=cv2.getOptimalNewCameraMatrix(SeekProCalib,SeekProDist,(RESOLUTIONWEBCAM),1,(RESOLUTIONWEBCAM))


class FPS:
    '''class to read the current fps'''
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self
    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()
    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1
    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()
    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


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

    def customColorMap(self, img, LUT):
        # print(img.size)
        newImg = np.zeros((img.shape[0],img.shape[1],3)).astype(int)
        # print(newImg)
        # for y in range (0,img.shape[0]):
        #     for x in range (0, img.shape[1]):
        #         newImg[y][x] = LUT[img[y][x]]
        lut = np.arange(255, -1, -1, dtype = img.dtype )
        lut3 = np.column_stack((lut, lut, lut))

        LUT.shape = img.shape

        newImg = lut3[img, LUT]
        # newImg = cv2.LUT(img,LUT)
        # print(newImg)
        return newImg
    
    def imageFusion(self, background, foreground):
        if background.shape == foreground.shape:
            ## Adds the images together (has a transarancy factor)
            fusedImg = cv2.addWeighted(background,1,foreground,0.5,0)        
            return fusedImg
        else :
            print('shape of background image and foreground image might not be the same')
            print(f'Background shape is :{background.shape}')
            print(f'Foreground shape is :{foreground.shape}')

    # def saliency(self,img,threshold):
    #     (saliencyMap, threshMap) = findSaliency.getSaliency(img,threshold)
    #     try:
    #         sortedImage, boundingBoxes = findSaliency.getContors(threshMap, img)
    #         return saliencyMap,threshMap, sortedImage, boundingBoxes
    #     except:
    #         print('Error Sorting Saliency')

    # def gimbalMovement(self, sortedImage,boundingBoxed):
    #     # try:
    #     signal = findSaliency.getContorPosition(sortedImage, boundingBoxes)
    #     return signal
    # # except:
    #     # print('no gimbal movement')

    def saliencyGimbalCommand(self,img, threshold):
        (saliencyMap, threshMap) = findSaliency.getSaliency(img,threshold)
        (saliencyMap2, threshMap2) = findSaliency2.getSaliency(saliencyMap,threshold)

        (sortedImage, boundingBoxes) = findSaliency.getContors(threshMap, saliencyMap2)#img)
        gimbalMove = findSaliency.getContorPosition(sortedImage,boundingBoxes)
        return (saliencyMap, saliencyMap2, threshMap, threshMap2, sortedImage, boundingBoxes, gimbalMove)

    def skinDetector(self,img):
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

    
        
        # try:  
        #     saliencyMap2, threshMap2, sortedImage, boundingBoxes, gimbalMove = imageModification().saliencyGimbalCommand(IRCropped, 55)
        #     cv2.imshow('threshMap2',threshMap2)
        #     cv2.imshow('saliencyMap2',saliencyMap2)
        # except:
        #     print('GimbalMovement failed')

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
        global WideImg
        ## Grabbing PiCam Image
        WideImg = PiCam.frameCapture()
        ## Image Processing region for the Wide Camera
        #PiCam image rotation
        WideImg = imageModification().imageRotation(WideImg,180)

        # WideImg = cv2.warpPerspective(WideImg,tform,(RESOLUTION))
        return #WideImg 

    def WebCamPost(self):
        global NarrowImg
        ##Grabbing WebCam Image 
        NarrowImg = WebCam.read()

        #rescaling the image to default size
        NarrowImg = imutils.resize(NarrowImg, width=320)
        
        #Removing image distortion
        # NarrowImg = cv2.undistort(NarrowImg, WebCamCalib, WebCamDist, None, WebCamUnDisMat)
        return #NarrowImg

    def SeekProPost(self):
        ##Grabbing SeekPro Image
        global IRImg
        global IRWarp
        IRImg = SeekPro.rescale(SeekPro.get_image())
        ##Image processing region for the Thermal Camera

        #Revoming image distortion
        IRWarp = cv2.undistort(IRImg, SeekProCalib, SeekProDist, None, SeekProUnDisMat)
        # cv2.resize(IRImg, (320*2,240*2))
        IRWarp = cv2.warpPerspective(IRImg,tform,(RESOLUTION))


        #applying histogram equalisaion
        # IRImg = cv2.equalizeHist(IRImg)

        #applying image smoothing improve image quality
        # IRImg = cv2.GaussianBlur(IRImg,(5,5),2)

        return #IRImg

if __name__ == '__main__':
    from time import time
    from time import strftime
    from time import sleep
    import os
    import concurrent.futures

    #Setting up the camera frame
    # cv2.namedWindow("piCam",cv2.WINDOW_NORMAL)
    # # cv2.namedWindow("webCam",cv2.WINDOW_NORMAL)
    # cv2.namedWindow("seekPro",cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Fused",cv2.WINDOW_NORMAL)

    # prevIR = CameraProcessing().SeekProPost()
    t0 = time()

    def saliencyImplement(img, threshold):
        try:  
            saliencyMap, saliencyMap2, threshMap, threshMap2, sortedImage, boundingBoxes, gimbalMove = imageModification().saliencyGimbalCommand(img, threshold)
            # cv2.imshow('threshMap',threshMap)
            # cv2.imshow('saliencyMap',saliencyMap)
            # cv2.imshow('saliencyMap2',saliencyMap2)
            # cv2.imshow('threshMap2',threshMap2)

        except:
            print('GimbalMovement failed')

    def thermalSmoothing(img):
        # ##IR thresholding
        IRsmooth = cv2.medianBlur(img,3)        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        IRsmooth = cv2.erode(IRsmooth, kernel, iterations = 3)
        IRsmooth = cv2.dilate(IRsmooth, kernel, iterations = 3)
        IRsmooth = cv2.medianBlur(IRsmooth,5)
        IRsmooth = cv2.medianBlur(IRsmooth,5)
        IRsmooth = cv2.medianBlur(IRsmooth,3)        
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        IRsmooth = cv2.dilate(IRsmooth, kernel, iterations = 2)
        IRsmooth = cv2.erode(IRsmooth, kernel, iterations = 2)

        # ret,IRsmooth = cv2.threshold(IRsmooth,50,255,cv2.THRESH_BINARY)

        #applying histogram equalisaion
        # IRsmooth = cv2.equalizeHist(IRsmooth)

        # IRsmooth = cv2.medianBlur(IR1,3)
        # Create kernel
        # kernelSharp = np.array([[0, -1, 0], 
        #                 [-1, 5,-1], 
        #                 [0, -1, 0]])

        # # Sharpen image
        # IRsmooth = cv2.filter2D(IRsmooth, -1, kernelSharp)
        return IRsmooth 

    def flooding(img):
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
        t = time()
        print("fps:",1/(t-t0))
        t0 = time()

        # IR1 = prevIR 
        # RGB1 = CameraProcessing().PiCamPost()
        # RGB2 = CameraProcessing().WebCamPost()

# concurrent.futures.ProcessPoolExecutor()
#concurrent.futures.ThreadPoolExecutor()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(CameraProcessing().SeekProPost())
            executor.submit(CameraProcessing().PiCamPost())
            executor.submit(CameraProcessing().WebCamPost())

        IR1 = IRImg
        RGB1 = WideImg
        RGB2 = NarrowImg
        IRWarp = IRWarp


        IRColor = cv2.applyColorMap(IRWarp, cv2.COLORMAP_RAINBOW)
        # IRColor[:,:,2] = 0
        # print(IRColor)

        IRFused = imageModification().imageFusion(RGB1,IRColor)

        ##attempting custom colomap
        # IRColor = cv2.applyColorMap(IRWarp, pupMat)
        ## IRColor = cv2.LUT(IR1, purpMap)
        IRFusednew = imageModification().imageCropping(IRFused, (47,55,190,140))

        ##SALIENCY WORK
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(saliencyImplement(RGB1, 55))
       
       ##Choosing the images to show
        # IRColor = imageModification().customColorMap(IR1,purpMap)
        cv2.imshow("piCam", RGB1)
        # cv2.imshow("webCam", RGB2)
        cv2.imshow("seekPro",IR1)
        # cv2.imshow("seekProWarp",IRWarp)
        # cv2.imshow("Fused", IRFused)
        # cv2.imshow("irColor", IRColor)



        IRsmooth = thermalSmoothing(IR1)
        # IRsmooth = cv2.blur(IRsmooth,(5,5))
        
        im_out = flooding(IRsmooth)




        cv2.imshow("seekProblur",IRsmooth)
        cv2.imshow("edges",edges)
        cv2.imshow("im_out",im_out)
        

        IRThresh = cv2.adaptiveThreshold(IRsmooth,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
        ret,IRThresh = cv2.threshold(IRsmooth,127,255,cv2.THRESH_BINARY)


        # cv2.imshow("IRThresh",IRThresh)
        # cv2.imshow("IRFusednew", IRFusednew)

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            WebCam.stop()
            PiCam.stop()
            break
        cv2.waitKey(1)


        
        # ##Saving Image Data
        # timestr = strftime("%d%m%Y-%H%M%S")
        # cv2.imwrite(f'/home/pi/SARCam/Images/SeekPro/{timestr}thermalimg.png', IR1)
        # cv2.imwrite(f'/home/pi/SARCam/Images/PiCam/{timestr}piimg.png', RGB1)
        # cv2.imwrite(f'/home/pi/SARCam/Images/WebCam/{timestr}webimg.png', RGB2)
        # cv2.imwrite(f'/home/pi/SARCam/Images/Fused/{timestr}fusedimg.png', IRFused)
        # cv2.imwrite(f'/home/pi/SARCam/Images/IRColor/{timestr}ircolorimg.png', IRColor)

        # cv2.imwrite(f'/home/pi/SARCam/Images/PiCamTest/{timestr}piimg.png', RGB1)
        # cv2.imwrite(f'/home/pi/SARCam/Images/SeekProTest/{timestr}thermalimg.png', IR1)
        # cv2.imwrite(f'/home/pi/SARCam/Images/IRColorTest/{timestr}ircolorimg.png', IRColor)
        # cv2.imwrite(f'/home/pi/SARCam/Images/ImFuseTest/{timestr}fusedimg.png', IRFused)
        # cv2.imwrite(f'/home/pi/SARCam/Images/ImFuseCroppedTest/{timestr}fusedimg.png', IRFusednew)
        # cv2.imwrite(f'/home/pi/SARCam/Images/edgeTest/{timestr}edgeimg.png', edges)



        # sleep(0.3)
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     executor.submit(CameraProcessing().SeekProPost())
        # prevIR = CameraProcessing().SeekProPost()


    # CameraProcessing.ReleaseCameras()
    WebCam.stop()
    PiCam.stop()
    cv2.destroyAllWindows()

    print("[INFO] RGB Cameras Released √")
    # return 



