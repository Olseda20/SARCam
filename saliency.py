import cv2

class findSaliency():
      def __init__(self):
            self.saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            # self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

      def getSaliency(self,img):
            (success, saliencyMap) = self.saliency.computeSaliency(img)
            threshMap = cv2.threshold((255*saliencyMap).astype("uint8"),0,255, 
                  cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
            return saliencyMap, threshMap

if __name__ == '__main__':
      from time import time

      import RGBCam
      from time import time

      #import image
      RGBCam = RGBCam.PiCam()

      cap = cv2.VideoCapture(0)
      if not (cap.isOpened()):
          print("Could not open video device")

      def getRGB():    
          ret, frame = cap.read()
          return frame 

      def visImageProc():
            #Get Visual Image
            RGB = RGBCam.frameCapture()
            return RGB

      findSaliency = findSaliency()
      t = 0 
      t0 =0

      while True:
            t = time()
            print("fps:",1/(t-t0))
            t0 = time()

            RGB = visImageProc()
            saliencyMap = findSaliency.getSaliency(RGB)[0]
            threshMap = findSaliency.getSaliency(RGB)[1]

            cv2.imshow("SalMap",saliencyMap)
            cv2.imshow("threshMap",threshMap)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
            cv2.waitKey(1)