import cv2
import RGBCam
from time import time

##import image
RGBCam = RGBCam.PiCam()

# from IRCam import getImage() as IR 

# cap = cv2.VideoCapture(0)
# if not (cap.isOpened()):
#     print("Could not open video device")

# def getRGB():    
#     ret, frame = cap.read()
#     return frame 

##Saleincy analysis
##find ROI in image

cv2.namedWindow("RGB",cv2.WINDOW_NORMAL)

def visImageProc():
      #Get Visual Image
      RGB = RGBCam.frameCapture()
      return RGB
t=0
t0 = 0
while True:
      t = time()
      print("fps:",1/(t-t0))
      t0 = time()

      RGB = visImageProc()
      # saliency = cv2.saliency.StaticSaliencyFineGrained_create()    
      saliency = cv2.saliency.StaticSaliencySpectralResidual_create()    
      (success, saliencyMap) = saliency.computeSaliency(RGB)
      # threshMap = cv2.threshold(saliencyMap.astype("uint8"),100,255,
      # cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
      cv2.imshow("RGB",RGB)
      # cv2.imshow("threshMap",threshMap)
      cv2.imshow("SalMap",saliencyMap)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

      if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      cv2.waitKey(1)
    