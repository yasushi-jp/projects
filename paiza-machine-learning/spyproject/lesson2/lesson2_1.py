# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 22:32:04 2018

@author: yasushi
"""

import cv2

img = cv2.imread('../images/000.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('gray000.png', img)
print(img)
print(img.shape)
print(img.ravel().shape)    