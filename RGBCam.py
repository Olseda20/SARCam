# Original Author: Omar Ali
### This module is where the visual cameras image can be captured

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

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
    """
     method to capture and undistory rgb camera image to match that of the thermal image 
    """
    self.rawCapture.truncate(0)
    for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
      img = frame.array

      # calibration matrix
      mtx = np.array([[626.80,0,85.301], [0,553.14,130.05], [0, 0, 1]])
      # distortion matrix
      dist = np.array([[0.23819,0.80864, 0.024662, -0.18100, -0.36040]])
      h, w = img.shape[:2]
      newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

      # undistort visual camera image
      dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
      # crop the image
      x,y,w,h = roi
      dst = dst[y:y+h, x:x+w]

      return dst

if __name__ == '__main__':
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
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    #rawCapture.truncate(0)
    cv2.imwrite(f'/home/pi/SARCam/Camera Calibration/Vis_01/{timestr}visimg.png', RGBCam.frameCapture())

    sleep(0.5)
    if key == ord("q"):
      break
