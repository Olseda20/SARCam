import cv2
# import RGBCam

##import image
# from RGBCam import PiCam().frameCapture() as RGB
# from IRCam import getImage() as IR 

def getRGB():
    cap = cv2.VideoCapture(0)
    if not (cap.isOpened()):
        print("Could not open video device")
    ret, frame = cap.read()
    return frame 

##Saleincy analysis
##find ROI in image

cv2.namedWindow("RGB",cv2.WINDOW_NORMAL)

while True:
    RGB = getRGB()
    # saliency = cv2.saliency.StaticSaliencyFineGrained_create()    
    # (success, saliencyMap) = saliency.computeSaliency(RGB)
    # threshMap = cv2.threshold(saleincyMap.astype("uint8"),0,255,
    # cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
    cv2.imshow("RGB",RGB)
    # cv2.imshow("threshMap",threshMap)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break