# Original Author: Victor Couty
# Modified by: Omar Ali

### This module is where the thermal camera image can be captured

##Giving OS Permissions
import os
import sys
import subprocess

if os.geteuid() != 0:
  subprocess.call(['sudo', 'python3'] + sys.argv)

import usb.core
import usb.util
import numpy as np
import cv2

# Address enum
READ_CHIP_ID                    = 54 # 0x36
START_GET_IMAGE_TRANSFER        = 83 # 0x53

GET_OPERATION_MODE              = 61 # 0x3D
GET_IMAGE_PROCESSING_MODE       = 63 # 0x3F
GET_FIRMWARE_INFO               = 78 # 0x4E
GET_FACTORY_SETTINGS            = 88 # 0x58

SET_OPERATION_MODE              = 60 # 0x3C
SET_IMAGE_PROCESSING_MODE       = 62 # 0x3E
SET_FIRMWARE_INFO_FEATURES      = 85 # 0x55
SET_FACTORY_SETTINGS_FEATURES   = 86 # 0x56

WIDTH = 320
HEIGHT = 240
RAW_WIDTH = 342
RAW_HEIGHT = 260


class SeekPro():
  """
  Seekpro class:
    Can read images from the Seek Thermal pro camera
    Can apply a calibration from the integrated black body
    Can locate and remove dead pixels
  This class only works with the PRO version !
  """
  def __init__(self):
    self.dev = usb.core.find(idVendor=0x289d, idProduct=0x0011)
    if not self.dev:
      raise IOError('Device not found')
    self.dev.set_configuration()
    self.calib = None
    for i in range(5):
      # Sometimes, the first frame does not have id 4 as expected...
      # Let's retry a few times
      if i == 4:
        # If it does not work, let's forget about dead pixels!
        print("Could not get the dead pixels frame!")
        self.dead_pixels = []
        break
      self.init()
      status,ret = self.grab()
      if status == 4:
        self.dead_pixels = self.get_dead_pix_list(ret)
        break

  def get_dead_pix_list(self,data):
    """
    Get the dead pixels image and store all the coordinates
    of the pixels to be corrected
    """
    img = self.crop(np.frombuffer(data,dtype=np.uint16).reshape(
          RAW_HEIGHT,RAW_WIDTH))
    return list(zip(*np.where(img<100)))

  def correct_dead_pix(self,img):
    """
    For each dead pix, take the median of the surrounding pixels
    """
    for i, j in self.dead_pixels:
      img[i,j] = np.median(img[max(0,i-1):i+2,max(0,j-1):j+2])
    return img

  def crop(self,raw_img):
    """
    Get the actual image from the raw image
    """
    return raw_img[4:4+HEIGHT,1:1+WIDTH]

  def send_msg(self,bRequest,  data_or_wLength,
      wValue=0, wIndex=0,bmRequestType=0x41,timeout=None):
    """
    Wrapper to call ctrl_transfer with default args to enhance readability
    """
    assert (self.dev.ctrl_transfer(bmRequestType, bRequest, wValue, wIndex,
      data_or_wLength, timeout) == len(data_or_wLength))

  def receive_msg(self,bRequest, data, wValue=0, wIndex=0,bmRequestType=0xC1,
      timeout=None):
    """
    Wrapper to call ctrl_transfer with default args to enhance readability
    """
    return self.dev.ctrl_transfer(bmRequestType, bRequest, wValue, wIndex,
          data, timeout)

  def deinit(self):
    """
    Is it useful ?
    """
    for i in range(3):
        self.send_msg(0x3C, b'\x00\x00')

  def init(self):
    """
    Sends all the necessary data to init the camera
    """
    self.send_msg(SET_OPERATION_MODE, b'\x00\x00')
    #print(receive_msg(GET_FIRMWARE_INFO, 4))
    #print(receive_msg(READ_CHIP_ID, 12))
    self.send_msg(SET_FACTORY_SETTINGS_FEATURES, b'\x06\x00\x08\x00\x00\x00')
    #print(receive_msg(GET_FACTORY_SETTINGS, 12))
    self.send_msg(SET_FIRMWARE_INFO_FEATURES,b'\x17\x00')
    #print(receive_msg(GET_FIRMWARE_INFO, 64))
    self.send_msg(SET_FACTORY_SETTINGS_FEATURES, b"\x01\x00\x00\x06\x00\x00")
    #print(receive_msg(GET_FACTORY_SETTINGS,2))
    for i in range(10):
      for j in range(0,256,32):
        self.send_msg(
            SET_FACTORY_SETTINGS_FEATURES,b"\x20\x00"+bytes([j,i])+b"\x00\x00")
        #print(receive_msg(GET_FACTORY_SETTINGS,64))
    self.send_msg(SET_FIRMWARE_INFO_FEATURES,b"\x15\x00")
    #print(receive_msg(GET_FIRMWARE_INFO,64))
    self.send_msg(SET_IMAGE_PROCESSING_MODE,b"\x08\x00")
    #print(receive_msg(GET_IMAGE_PROCESSING_MODE,2))
    self.send_msg(SET_OPERATION_MODE,b"\x01\x00")
    #print(receive_msg(GET_OPERATION_MODE,2))

  def grab(self):
    """
    Asks the device for an image and reads it
    """
    # Send read frame request
    self.send_msg(START_GET_IMAGE_TRANSFER, b'\x58\x5b\x01\x00')
    toread = 2*RAW_WIDTH*RAW_HEIGHT
    ret  = self.dev.read(0x81, 13680, 1000)
    remaining = toread-len(ret)
    # 512 instead of 0, to avoid crashes when there is an unexpected offset
    # It often happens on the first frame
    while remaining > 512:
      # print(remaining," remaining")
      ret += self.dev.read(0x81, 13680, 1000)
      remaining = toread-len(ret)
    status = ret[4]
    if len(ret) == RAW_HEIGHT*RAW_WIDTH*2:
      return status,np.frombuffer(ret,dtype=np.uint16).reshape(
            RAW_HEIGHT,RAW_WIDTH)
    else:
      return status,None

  def get_image(self):
    """
    Method to get an actual IR image
    """
    while True:
      status,img = self.grab()
      #print("Status=",status)
      #Program breaking
      if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
      if status == 1: # Calibration frame
        self.calib = self.crop(img)-1600
      elif status == 3: # Normal frame
        if self.calib is not None:
          return self.correct_dead_pix(self.crop(img)-self.calib)

  def rescale(self, img):
    """
    To adapt the range of values to the actual min and max and cast it into
    an 8 bits image
    """
    #shifting the range of the image
    img = img + 30000
    # #implemented to prevent the thermal image
    # img[img >60000] = 0

    if img is None:
        return np.array([0])
    mini = img.min()
    maxi = img.max()
    imgScale = (np.clip(img-mini,0,maxi-mini)/(maxi-mini)*255.).astype(np.uint8) 
    
    #Calibration matrix
    mtx = np.array([[639.06, 0, 135.45],[0, 637.74, 99.42],[0, 0, 1]])
    #Distortion matrix
    dist = np.array([[-1.1945, 24.321, -0.00598, 0.01358, -0.02011]])
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (WIDTH,HEIGHT), 1, (WIDTH,HEIGHT))
    
    # undistort
    dst = cv2.undistort(imgScale, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    return dst

if __name__ == '__main__':
  from time import time
  from time import strftime
  from time import sleep
  import scipy.io as sio

  thermdata = {}
  thermrescaledata = {}

  # Setting thermal camera
  IRCam = SeekPro()
  cv2.namedWindow("Seek",cv2.WINDOW_NORMAL)
  t0 = time()

  while True:
    t = time()
    print("fps:",1/(t-t0))
    t0 = time()

    ir_img = IRCam.get_image()
    ir_img_rescale = IRCam.rescale(ir_img)

    cv2.imshow("Seek", ir_img_rescale)
    timestr = strftime("%d%m%Y-%H%M%S")
    thermdata['thermcameradata'] = ir_img
    thermrescaledata['thermcameradatarescale'] = ir_img_rescale

    # replace path with relevant storage position
    sio.savemat(f'/home/pi/SARCam/thermalmat/{timestr}thermalmat.mat', thermdata) 
    sio.savemat(f'/home/pi/SARCam/thermalmat/{timestr}thermalrescale.mat', thermrescaledata)
    cv2.imwrite(f'/home/pi/SARCam/thermalmat/{timestr}thermalimg.png', ir_img_rescale)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    sleep(0.1)
    cv2.waitKey(1)
