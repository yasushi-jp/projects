# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 20:55:43 2018

@author: yasushi
"""

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#---------------------------
#次の一行を変更してください
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape) 
#---------------------------