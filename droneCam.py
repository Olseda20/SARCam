import IRCam
import cv2



from time import time

# Setting thermal camera
IRCam = IRCam.SeekPro()
t0 = time()

while True:
    t = time()
    print("fps:",1/(t-t0))
    t0 = time()
  
    cv2.namedWindow("Seek",cv2.WINDOW_NORMAL)
    
    r = IRCam.get_image()
    rdisp = IRCam.rescale(r)
    #print(rdisp)
    cv2.imshow("Seek",rdisp)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break
      
    cv2.waitKey(1)

