# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 13:55:34 2022

@author: niyan
"""

"""Polynomial regression"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(6).reshape(3,2)
print(x)

poly = PolynomialFeatures(degree=2)
transformed = poly.fit_transform(x)
print(transformed)
