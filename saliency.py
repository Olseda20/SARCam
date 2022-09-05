# Original Author: Omar Ali

import cv2
from time import time
import RGBCam

class findSaliency():
  def __init__(self):
    # select between the different saliency types
    self.saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    # self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

  def getSaliency(self,img):
    """
     captures the saliency and thershold map of a given image
    """
    (success, saliencyMap) = self.saliency.computeSaliency(img)
    threshMap = cv2.threshold((255*saliencyMap).astype("uint8"),0,255, 
      cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
    return saliencyMap, threshMap

def getRGB():
  """
   capture current RGB frame
  """
  ret, frame = cap.read()
  return frame 

def visImageProc():
  """
   
  """
  #Get Visual Image
  RGB = RGBCam.frameCapture()
  return RGB

if __name__ == '__main__':
  RGBCam = RGBCam.PiCam()

  cap = cv2.VideoCapture(0)
  if not (cap.isOpened()):
    print("Could not open video device")

  findSaliency = findSaliency()
  t = 0 
  t0 = 0

  while True:
    t = time()
    print("fps:", 1/(t-t0))
    t0 = time()

    RGB = visImageProc()
    saliencyMap = findSaliency.getSaliency(RGB)[0]
    threshMap = findSaliency.getSaliency(RGB)[1]

    cv2.imshow("SalMap",saliencyMap)
    cv2.imshow("threshMap",threshMap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    cv2.waitKey(1)