# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 00:13:33 2018

@author: yasushi
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

targets_data = pd.read_csv('../y_classified.csv')

images_data = np.empty((0, 256), int)
for i in range(100):
    png = ('../images/%03d.png' %(i))
    img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    images.append(img)
    hist = np.histogram(img.ravel(), 256, [0,256])
    images_data = np.append(images_data, np.array([hist[0]]), axis=0)
    
print(images_data.shape)

# 教師データとテストデータ、教師データの答え、テストデータの答えに分割する
X_train, X_test, y_train, y_test = train_test_split(images_data, targets_data['Kirishima'], random_state=0)

# 分割結果を確認する
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print(knn.predict(np.array([X_test[0]])))
print(knn.predict(np.array([X_test[0], X_test[1], X_test[2], X_test[3]])))
print(y_test)

y_pred = knn.predict(X_test)
print(y_pred)
print(np.mean(y_pred == y_test))
