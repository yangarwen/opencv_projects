import cv2
flags=[i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)
cv2.cvtColor(img, cv2.COLOR_RGB2HSV)