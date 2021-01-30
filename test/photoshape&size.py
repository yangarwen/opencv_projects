import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('hu.jpg',1)
px=img[100,100] =[0,200,0]
print(img.shape, img.size)

# img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# print(img)
# plt.imshow(img1)
# plt.xticks([]), plt.yticks([])
# plt.show()

# cv2.namedWindow('hu', cv2.WINDOW_NORMAL)
# cv2.imshow('hu', img)
# cv2.imwrite('hu2.jpg',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()