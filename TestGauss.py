# Python (OpenCV) 示例
import cv2
import numpy as np

img = cv2.imread('1.jpg')

# 使用 5x5 的核，sigma=1.0
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)

cv2.imshow('Original', img)
cv2.imshow('Gaussian Blurred', blurred)
cv2.waitKey(0)

