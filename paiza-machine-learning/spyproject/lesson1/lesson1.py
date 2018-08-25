# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 22:12:53 2018

@author: yasushi
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

x = np.arange(10)
y = np.random.randint(1, 100, 10)
print(x)
print(y)

plt.plot(x, y)
plt.show()