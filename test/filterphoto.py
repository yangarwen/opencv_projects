import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('hu.jpg',1)
b = cv2.imread('hu.jpg',1)
b[100:200,:,0]=0
b[300:400,:,0]=0
print(b)

# roi=img[149:194,460:608]
# img[77:122,640:788]=roi

# img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# plt.imshow(img1)
# plt.xticks([]), plt.yticks([])
# plt.show()

cv2.namedWindow('hu', cv2.WINDOW_NORMAL)
# cv2.imshow('hu', img)
cv2.imshow('b', b)

cv2.waitKey(0)
cv2.destroyAllWindows()