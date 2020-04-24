import cv2
import numpy as np
from imutils import contours
import imutils

class findSaliency():
      def __init__(self):
            ''' Initiating the saliency model to be use for this process'''
            self.saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            # self.saliency = cv2.saliency.()
            # self.saliency = cv2.saliency.StaticSaliencyBinWangApr2014_create()
            # self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

            # self.saliency = cv2.saliency.StaticSaliency()

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
            '''Function to find all the thresholded regions in a saleint image 
            and to identify all of the possible ROI'''

            # find contours in the edge map
            cnts = cv2.findContours(threshMap, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            #Sort Contors
            method = "bottom-to-top"
            (cnts, boundingBoxes) = contours.sort_contours(cnts, method=method)
            imgshape = threshMap.shape
            try:
                  for (i, c) in enumerate(cnts):
                        sortedImage = contours.label_contour(orig, c, i, color=(240, 0, 159))
                  return sortedImage, boundingBoxes
            except:
                  print('sort fail')

      def getContorPosition(self, img, boundingBoxes):
            #Left Right
            L, C, R = 0, 0, 0
            #Up Down
            T, M, B = 0, 0, 0
            # initialiising array 
            try: 
                  ROIPoints = np.zeros(shape=(len(boundingBoxes),2), dtype='uint16')
            except:
                  print('no ROI points')

            imgshape = img.shape #Whatever image input is placed into here

            #Creating a tally of positions where the most interesting features are
            for i in range (0,len(boundingBoxes)):
                  try:
                        #Saving the points of the image that 
                        ROIPoints[i][0] = boundingBoxes[i][0]
                        if ROIPoints[i][0] <= int(imgshape[1]*1/3):
                              L = L + 1
                        elif ROIPoints[i][0] > int(imgshape[1]*1/3) and ROIPoints[i][0] < int(imgshape[1]*2/3):
                              C = C + 1 
                        elif ROIPoints[i][0] >= int(imgshape[1]*2/3):
                              R = R + 1 
                        else:
                              print('horizontal positional error')

                        ROIPoints[i][1] = boundingBoxes[i][1]
                        #Remember downwards positive axis
                        if ROIPoints[i][1] >= int(imgshape[0]*2/3):
                              B = B + 1
                        elif ROIPoints[i][1] > int(imgshape[0]*1/3) and ROIPoints[i][1] < int(imgshape[0]*2/3):
                              M = M + 1 
                        elif ROIPoints[i][1] <= int(imgshape[0]*1/3):
                              T = T + 1 
                        else:
                              print('vertical positional error')
                  except:
                        print('error in the ROI Location append loop')

            #Left right signal 
            if L > C or L > R:
                  Hsignal = -1
                  print('left')
            elif R > C or R > L:
                  Hsignal = 1
                  print('right')
            else:
                  Hsignal = 0
            # Up down signal
            if T > M or T > B:
                  Vsignal = 1
                  print('up')
            elif B > M or B > T:
                  Vsignal = -1
                  print('down')
            else:
                  Vsignal = 0

            signal = Hsignal, Vsignal
            return signal

class findSaliency2():
      def __init__(self):
            ''' Initiating the saliency model to be use for this process'''
            # self.saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

            # self.saliency = cv2.saliency.StaticSaliency()

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

if __name__ == '__main__':

      from time import time
      from time import time


      t = 0 
      t0 =0
      findSaliency = findSaliency()
      findSaliency2 = findSaliency2()
      threshold = 85
      i = 0

      # for image 
      # RGB = cv2.imread('/home/pi/SARCam/Images/PiCam/11042020-132426piimg.png')
      # RGB = cv2.imread('/home/pi/SARCam/Images/PiCam/11042020-132353piimg.png')
      # RGB = cv2.imread('/home/pi/SARCam/Images/PiCam/11042020-132323piimg.png')
      # RGB = cv2.imread('/home/pi/SARCam/Images/PiCam/11042020-132319piimg.png')

      RGB = cv2.imread('/home/pi/SARCam/Images/PiCamTest/14042020-200950piimg.png')
      (saliencyMap, threshMap) = findSaliency.getSaliency(RGB,30)
      (saliencyMap2, threshMap2) = findSaliency2.getSaliency(saliencyMap,55)
      # (saliencyMap2, threshMap2) = findSaliency2.getSaliency(RGB,55)

      # sortedImage, boundingBoxes = findSaliency.getContors(threshMap, RGB)
      # signal = findSaliency.getContorPosition(sortedImage, boundingBoxes)
      cv2.imshow('rgb',RGB)
      cv2.imshow("SalMap",saliencyMap)
      cv2.imshow("SalMap2",saliencyMap2)

      cv2.imshow("threshMap",threshMap)
      cv2.imshow("threshMap2",threshMap2)

      # cv2.imshow("sortedImage",sortedImage)
      cv2.waitKey(0)


      # import RGBCam
      # # #IMAGE CAPTUREimage
      # RGBCam = RGBCam.PiCam()
      # cap = cv2.VideoCapture(0)
      # if not (cap.isOpened()):
      #     print("Could not open video device")
      # def getRGB():    
      #     ret, frame = cap.read()
      #     return frame 
      # def visImageProc():
      #       #Get Visual Image
      #       RGB = RGBCam.frameCapture()
      #       return RGB

      #for video
      # while True:
      #       t = time()
      #       print("fps:",1/(t-t0))
      #       t0 = time()

      #       if cv2.waitKeyEx(1) & 0xFF == ord('w'):
      #             threshold = threshold + 5
      #       if cv2.waitKeyEx(1) & 0xFF == ord('s'):
      #             threshold = threshold - 5

      #       # print(threshold)

      #       #image capture
      #       RGB = visImageProc()
            
      #       #image saliency and thresholding
      #       (saliencyMap, threshMap) = findSaliency.getSaliency(RGB,threshold)
      #       (saliencyMap2, threshMap) = findSaliency2.getSaliency(saliencyMap,55)

            
      #       #finding image contors
      #       try:
      #             sortedImage, boundingBoxes = findSaliency.getContors(threshMap, RGB)
      #       except:
      #             print('image contoring error')

      #       # print(boundingBoxes)
      #       #organising contor positons and finding the gimbal movement direction
      #       if i == 10:
      #             try:
      #                   signal = findSaliency.getContorPosition(sortedImage, boundingBoxes)
      #             except:
      #                   print('no gimbal movement')
      #             i = 0
      #       i = i+1

      #       cv2.imshow('rgb',RGB)
      #       cv2.imshow("SalMap",saliencyMap)
      #       cv2.imshow("SalMap2",saliencyMap2)

      #       cv2.imshow("threshMap",threshMap)
            
      #       if cv2.waitKey(1) & 0xFF == ord('q'):
      #             break
      #       cv2.waitKey(1)