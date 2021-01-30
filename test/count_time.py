import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('hu.jpg',1)
b = cv2.imread('hu.jpg',1)
e1 = cv2.getTickCount()
img1 = b[300:400,100:200]
img2 = b[100:200,300:400]
dst=cv2.addWeighted(img1,0.7,img2,0.3,0)
b[100:200,300:400]= dst
e2 = cv2.getTickCount()
t = (e2-e1) / cv2.getTickFrequency()
print(t)

cv2.namedWindow('hu', cv2.WINDOW_NORMAL)
cv2.imshow('hu', img)
cv2.imshow('b', b)

cv2.waitKey(0)
cv2.destroyAllWindows()