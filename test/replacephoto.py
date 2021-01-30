import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('hu.jpg',1)
b = cv2.imread('hu.jpg',1)
img1 = b[300:400,100:200]
img2 = b[100:200,300:400]
dst=cv2.addWeighted(img1,0.7,img2,0.3,0)
b[100:200,300:400]= dst


cv2.imshow('b', b)

cv2.waitKey(0)
cv2.destroyAllWindows()