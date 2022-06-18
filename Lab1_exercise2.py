# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 13:14:00 2022

@author: niyan
"""

"""linear regression
Close form solution"""


import numpy as np
from matplotlib import pyplot as plt

x = 2 * np.random.rand(100,1)
y = 4 + 3 * x + np.random.rand(100,1)


x_b = np.c_[np.ones((100,1)),x]

theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
print(theta_best)


