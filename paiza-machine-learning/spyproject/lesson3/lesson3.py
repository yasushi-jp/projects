# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 23:12:46 2018

@author: yasushi
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

targets_data = pd.read_csv('../y_classified.csv')
print(targets_data['Kirishima'])

images = []
for i in range(100):
    file = ('../images/%03d.png' %(i))
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    images.append(img)

images_data = np.empty((100, len(images[0].ravel())), int)
for i in range(100):
    images_data[i] = np.array([images[i].ravel()])

print(images_data.shape)