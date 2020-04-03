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

        #Convert Data into readable array        
        self.rawCapture = PiRGBArray(self.camera, RESOLUTION)
        
        #Camera Warmup Time
        time.sleep(0.1)

    def frameCapture(self):
        '''Capturing the RGB Image'''
        #Clearing the image stream for the next frame (otherwise there will be a buffer error)
        self.rawCapture.truncate(0)

        #Capturing a frame from the video stream to display         
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            img1 = frame.array
            return img1

    def releaseCapture(self):
        self.camera.close()
        
class WebCam():

    def __init__(self):
        self.camera2 = cv2.VideoCapture(1)
        # self.camera2.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        # self.camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

        # allow the camera to warmup
        time.sleep(0.1)  
        
    def frameCapture(self):
        '''Capturing the RGB Image (CAMERA 2)'''
        ret, img2 = self.camera2.read()
        (h,w) = img2.shape[:2]
        centre = (w/2,h/2)
        img2Narrow = img2[int(w/2)-int(w/4):int(w/2)+int(w/4), int(h/2)-int(h/4):int(h/2)+int(h/4)]
        img2Resize = cv2.resize(img2Narrow, RESOLUTION)
        return img2Resize

    def releaseCapture(self):
        self.camera2.release()

class cam_initialise():
    def __init__(self):          
        RGBCam1 = PiCam()
        RGBCam1.frameCapture
        RGBCam1.releaseCapture()

        RGBCam2 = WebCam()
        RGBCam2.frameCapture()
        RGBCam2.releaseCapture()



if __name__ == '__main__':
    import cv2
    import time  
    from time import strftime
    from time import sleep

    # Initialising the RGB Cameras
    print("[INFO] Camera initialising...")
    cam_initialise()
    RGBCam1 = PiCam()
    RGBCam2 = WebCam()
    cv2.namedWindow("piCam",cv2.WINDOW_NORMAL)
    cv2.namedWindow("webCam",cv2.WINDOW_NORMAL)
    print("[INFO] Camera initialised √")
    
    print("[INFO] First Image Captured...")
    img1 = RGBCam1.frameCapture()
    img2 = RGBCam2.frameCapture()

    while True:
        img1 = RGBCam1.frameCapture()
        img2 = RGBCam2.frameCapture()

        cv2.imshow("piCam", img1)
        cv2.imshow("webCam", img2)

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        cv2.waitKey(1)

    RGBCam1.releaseCapture()
    RGBCam2.releaseCapture()
    print("[INFO] RGB Cameras Released √")
    cv2.destroyAllWindows()

    # cv2.imshow("Persp",RGBCam.RGBCalibration(RGBCam.frameCapture(), mtx, dist))
    # timestr = strftime("%d%m%Y-%H%M%S")
     #cv2.imwrite(f'/home/pi/SARCam/Camera Calibration/Vis_01/{timestr}visimg.png', RGBCam.frameCapture())

'''
#    def RGBCalibration(self, img, mtx, dist):
#        Applying image calibration to remove image distortion on the visual
#            visual image. 
#            Note: The calibration distortion matrices have been determined using 
#            a calibration script external to this script this will need to be 
#            manually placed in after evaluation
#
#        #Getting the image shape
#        h,  w = img.shape[:2]
#        #Using the calibration and distortion matrix generate the corrected image
#        #with croppable regions to prevent epty regions in the image
#        CorrectionMtx, imCrop=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
#        
#        #Applying the image calibration
#        RGBCalibrated = cv2.undistort(img, mtx, dist, None, CorrectionMtx)
#
#        ##Cropping the image using imCrop
#        # x,y,w,h = imCrop
#        # dst = dst[y:y+h, x:x+w]
#        return RGBCalibrated 
'''