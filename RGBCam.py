# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

RESOLUTION = 320, 240
FRAMERATE = 15

#Calibration matrix 
mtx = np.array([[626.80,0,85.301],[0,553.14,130.05],[0, 0, 1]])
#Distortion matrix
dist = np.array([[0.23819,0.80864,0.024662,-0.18100,-0.36040]])

class PiCam():
    def __init__(self):
        # initialize the camera and grab a reference to the raw camera capture
        self.camera = PiCamera()
        self.camera.resolution = RESOLUTION
        self.camera.framerate = FRAMERATE
        self.rawCapture = PiRGBArray(self.camera, RESOLUTION)
        # allow the camera to warmup
        time.sleep(0.1)
        
    def frameCapture(self):
        '''Capturing the RGB Image'''
        #Clearing the image stream for the next frame (otherwise there will be a buffer error)
        self.rawCapture.truncate(0)

        #Capturing a frame from the video stream to display 
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            img = frame.array
            return img
    
    def RGBCalibration(self, img, mtx, dist):
        '''Applying image calibration to remove image distortion on the visual
            visual image. 
            Note: The calibration distortion matrices have been determined using 
            a calibration script external to this script this will need to be 
            manually placed in after evaluation'''

        #Getting the image shape
        h,  w = img.shape[:2]
        #Using the calibration and distortion matrix generate the corrected image
        #with croppable regions to prevent epty regions in the image
        CorrectionMtx, imCrop=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        
        #Applying the image calibration
        RGBCalibrated = cv2.undistort(img, mtx, dist, None, CorrectionMtx)

        ##Cropping the image using imCrop
        # x,y,w,h = imCrop
        # dst = dst[y:y+h, x:x+w]
        return RGBCalibrated



if __name__ == '__main__':
    import cv2
    import time  
    from time import strftime
    from time import sleep

  
    RGBCam = PiCam()
    cv2.namedWindow("RGB",cv2.WINDOW_NORMAL)
    
    while True:
        timestr = strftime("%d%m%Y-%H%M%S")

        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        # show the frame
        cv2.imshow("RGB", RGBCam.frameCapture())
        cv2.imshow("Persp",RGBCam.RGBCalibration(RGBCam.frameCapture(), mtx, dist))




        key = cv2.waitKey(1) & 0xFF
        # clear the stream in preparation for the next frame
        #rawCapture.truncate(0)
        # if the `q` key was pressed, break from the loop
        cv2.imwrite(f'/home/pi/SARCam/Camera Calibration/Vis_01/{timestr}visimg.png', RGBCam.frameCapture())
        if key == ord("q"):
            break
        cv2.waitKey(1)
  
    
