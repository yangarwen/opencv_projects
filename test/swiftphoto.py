import cv2
import numpy as np
from matplotlib import pyplot as plt


WAIT = 3000
img1 = cv2.imread('forest1.jpg',1)
img2 = cv2.imread('forest2.jpg',1)

dst=cv2.addWeighted(img1,0.7,img2,0.3,0)

for it in range(WAIT+1):
        if it % 100 == 0:
            weight = it / WAIT
            res = cv2.addWeighted(img1, 1-weight, img2, weight, 0)
            cv2.imshow('images', res)
            cv2.waitKey(100)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()