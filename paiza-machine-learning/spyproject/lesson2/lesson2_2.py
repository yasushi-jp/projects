# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 23:10:07 2018

@author: yasushi
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# ヒストグラムを描画
def plot_hist(img):
    img_hist = np.histogram(img.ravel(), 256, [0, 256])
    hist = img_hist[0]
    plt.bar(np.arange(256), hist)
    plt.show()

plot_hist(cv2.imread('../images/000.png', cv2.IMREAD_GRAYSCALE))
plot_hist(cv2.imread('../images/001.png', cv2.IMREAD_GRAYSCALE))
plot_hist(cv2.imread('../images/002.png', cv2.IMREAD_GRAYSCALE))