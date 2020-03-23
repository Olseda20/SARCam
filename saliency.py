import cv2
import numpy as np
from imutils import contours
import imutils

class findSaliency():
      def __init__(self):
            ''' Initiating the saliency model to be use for this process'''
#            self.saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

      def getSaliency(self,img,threshperc):
            ''' functin to capture the saliency in a given image and output a threshold map based on 
            a given threshold percentage desire'''
            #Grabbing the saliency and converting it into a grescale integer value
            saliencyMap = np.array(255*self.saliency.computeSaliency(img)[1],dtype="uint8")
            #Creating a threshold percentatge to grescale conversion
            threshpercconv = np.int((threshperc/100)*255)
            #applying thresholding the to top 'threshperc' of the image
            threshMap = cv2.threshold(saliencyMap,threshpercconv,255,cv2.THRESH_BINARY)[1]
            return saliencyMap, threshMap

      def getContors(self,threshMap, orig):
            # find contours in the edge map
            cnts = cv2.findContours(threshMap, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            #Sort Contors
            method = "bottom-to-top"
            (cnts, boundingBoxes) = contours.sort_contours(cnts, method=method)
            imgshape = threshMap.shape

            # initialiising array 
            try: 
                  ROIPoints = np.zeros(shape=(len(boundingBoxes),2), dtype='uint8')
            except:
                  print('no ROI points')
            try:

                  for (i, c) in enumerate(cnts):
                        sortedImage = contours.label_contour(orig, c, i, color=(240, 0, 159))
                  return sortedImage
            except:
                  print('sort fail')
            

      '''def getContorPoition(self):
            #Left Right
            L, C, R = 0, 0, 0
            #Up Down
            T, M, B = 0, 0, 0

            imgshape = threshMap.shape #Whatever image input is placed into here

            #Creating a tally of positions where the most interesting features are
            for i in range (0,len(boundingBoxes)):
            try:
                  #Saving the points of the image that 
                  ROIPoints[i][0] = boundingBoxes[i][0]
                  if ROIPoints[i][0]<= int(imshape[1]*1/3):
                        L = L + 1
                  elif ROIPoints[i][0] > int(imshape[1]*1/3) and ROIPoints[i][0] < int(imshape[1]*2/3):
                        C = C + 1 
                  elif ROIPoints[i][0] > int(imshape[1]*2/3):
                        R = R + 1 
                  else:
                        print('horizontal positional error')

                  #
                  ROIPoints[i][1] = boundingBoxes[i][1]
                  #Remember downwards positive axis
                  if ROIPoints[i][1] >= int(imshape[0]*2/3):
                        B = B + 1
                  elif ROIPoints[i][1] > int(imshape[0]*1/3) and ROIPoints[i][1] < int(imshape[0]*2/3):
                        M = M + 1 
                  elif ROIPoints[i][1] <= int(imshape[0]*1/3):
                        T = T + 1 
                  else:
                        print('vertical positional error')
            except:
                  print('error in the ROI Location append loop')


            #prioritising the centre if there are an even number of points on interest around the centre
            # while L == R | T == B:
            #     if L != 0 & R != 0:
            #         L = L - 1
            #         R = R - 1
            #         C = C + 1

            #     if T != 0 & B != 0:
            #         T = T - 1
            #         B = B - 1
            #         M = M + 1

            Hsignal = 0
            Vsignal = 0

            #Left right signal 
            if L > C or L > R:
            Hsignal = -1
            if R > C or R > L:
            Hsignal = 1
            if T > M or T > B:
            Vsignal = 1
            if B > M or B > T:
            Vsignal = -1

            print(f'horizontal {Hsignal}')
            print(f'Vertical {Vsignal}')
            return'''

if __name__ == '__main__':

      from time import time
      import RGBCam
      from time import time

      #IMAGE CAPTUREimage
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

      t = 0 
      t0 =0
      findSaliency = findSaliency()
      threshold = 10
      while True:
            t = time()
            print("fps:",1/(t-t0))
            t0 = time()

            if cv2.waitKeyEx(1) & 0xFF == ord('w'):
                  threshold = threshold + 5
            if cv2.waitKeyEx(1) & 0xFF == ord('s'):
                  threshold = threshold - 5

            print(threshold)
            

            RGB = visImageProc()
            (saliencyMap, threshMap) = findSaliency.getSaliency(RGB,threshold)
            contor = findSaliency.getContors(threshMap, RGB)

            cv2.imshow('rgb',RGB)
            cv2.imshow("SalMap",saliencyMap)

            cv2.imshow("threshMap",threshMap)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
            cv2.waitKey(1)