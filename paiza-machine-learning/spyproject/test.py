# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 19:26:12 2018

@author: yasushi
"""

import tensorflow as tf
hello = tf.constant('Hello TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
