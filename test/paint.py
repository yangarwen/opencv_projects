import cv2
import numpy as np

img=np.zeros((512,512,3), np.uint8)
cv2.line(img,(0,0),(500,500),(255,251,0),5)
cv2.rectangle(img,(100,120),(300,300),(0,15,150),3)                  
cv2.circle(img,(100,200), 150, (40, 156, 10) ,2)
cv2.ellipse(img,(256,256),(100,50),0,0,360,255,-1) 

points = np.array([[0,0],[150,200],[300,200],[41,100]])
print(points)
cv2.polylines(img,[points],True,(0,200,100),5)    

font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2)

cv2.imshow('line', img)
cv2.waitKey(0)
cv2.destroyAllWindows()