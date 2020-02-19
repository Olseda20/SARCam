# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

RESOLUTION = 320, 240
FRAMERATE = 32

class PiCam():
    def __init__(self):
        # initialize the camera and grab a reference to the raw camera capture
        self.camera = PiCamera()
        self.camera.resolution = RESOLUTION
        self.camera.framerate = FRAMERATE
        self.rawCapture = PiRGBArray(self.camera, RESOLUTION)
        # allow the camera to warmup
        time.sleep(0.1)
        # capture frames from the camera
    
    def frameCapture(self):
        self.rawCapture.truncate(0)
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            return image
            

if __name__ == '__main__':
    import cv2
    import time
  
    RGBCam = PiCam()
    cv2.namedWindow("RGB",cv2.WINDOW_NORMAL)
    
    while True:
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        # show the frame
        cv2.imshow("RGB", RGBCam.frameCapture())
        key = cv2.waitKey(1) & 0xFF
        # clear the stream in preparation for the next frame
        #rawCapture.truncate(0)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        
  
    
