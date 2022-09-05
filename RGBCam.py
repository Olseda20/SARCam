# import the necessary packages
from datetime import datetime
from picamera import PiCamera
from picamera.array import PiRGBArray
from threading import Thread
import numpy as np
import cv2
import imutils
import time

RESOLUTION = 320, 240
FRAMERATE = 25

class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
    def start(self):
        # start the timer
        self._start = datetime.now()
        return self
    def stop(self):
        # stop the timer
        self._end = datetime.now()
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

class PiCam():
    def __init__(self):
        # initialize the camera and grab a reference to the raw camera capture
        self.camera = PiCamera()
        self.camera.resolution = RESOLUTION
        self.camera.framerate = FRAMERATE
        # convert the Data into readable array
        self.rawCapture = PiRGBArray(self.camera, RESOLUTION)
        # camera warmup time
        time.sleep(0.1)

    def frameCapture(self):
        '''Capturing the RGB Image'''
        # clearing the image stream for the next frame (otherwise there will be a buffer error)
        self.rawCapture.truncate(0)
        # capturing a frame from the video stream to display
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            img1 = frame.array
            return img1
            
    def stop(self):
        self.camera.close()

class WebCam():
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class cam_initialise():
    def __init__(self):
        print("[INFO] Camera initialising...")
        RGBCam1 = PiCam()
        RGBCam1.frameCapture
        RGBCam1.releaseCapture()
        time.sleep(0.2)
        RGBCam2 = WebCam()
        RGBCam2.frameCapture()
        RGBCam2.releaseCapture()
        ## Diplay capture window for debugging
        # cv2.namedWindow("piCam",cv2.WINDOW_NORMAL)
        # cv2.namedWindow("webCam",cv2.WINDOW_NORMAL)
        print("[INFO] Camera initialised √")



if __name__ == '__main__':
    vs = WebCam(src=1).start()
    fps = FPS().start()

    # loop over some frames...this time using the threaded stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        # frame = imutils.resize(frame, width=320)

        # check to see if the frame should be displayed to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    # end cv2 window "Frame" after coompleted run
    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     import cv2
#     import time  
#     from time import strftime
#     from time import sleep

#     # Initialising the RGB Cameras
#     cam_initialise()
#     RGBCam1 = PiCam()
#     RGBCam2 = WebCam()

#     #Setting up the camera frame
#     cv2.namedWindow("piCam",cv2.WINDOW_NORMAL)
#     cv2.namedWindow("webCam",cv2.WINDOW_NORMAL)
#     #Capturing the image
#     img1 = RGBCam1.frameCapture()
#     img2 = RGBCam2.frameCapture()

#     while True:
#         img1 = RGBCam1.frameCapture()
#         img2 = RGBCam2.frameCapture()

#         cv2.imshow("piCam", img1)
#         cv2.imshow("webCam", img2)

#         # if the `q` key was pressed, break from the loop
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break
#         cv2.waitKey(1)

#     RGBCam1.releaseCapture()
#     RGBCam2.releaseCapture()
#     print("[INFO] RGB Cameras Released √")
#     cv2.destroyAllWindows()

#     # cv2.imshow("Persp",RGBCam.RGBCalibration(RGBCam.frameCapture(), mtx, dist))
#     # timestr = strftime("%d%m%Y-%H%M%S")
#      #cv2.imwrite(f'/home/pi/SARCam/Camera Calibration/Vis_01/{timestr}visimg.png', RGBCam.frameCapture())

# class oldWebCam():
#     def __init__(self):
#         self.camera2 = cv2.VideoCapture(1)
#         # allow the camera to warmup
#         time.sleep(0.1)  

#     def frameCapture(self):
#         '''Capturing the RGB Image (CAMERA 2)'''
#         ret, img2 = self.camera2.read()
#         # (h,w) = img2.shape[:2]
#         # centre = (w/2,h/2)
#         # img2Narrow = img2[int(w/2)-int(w/4):int(w/2)+int(w/4), int(h/2)-int(h/4):int(h/2)+int(h/4)]
#         # img2Resize = cv2.resize(img2Narrow, RESOLUTION)
#         return img2

#     def releaseCapture(self):
#         self.camera2.release()