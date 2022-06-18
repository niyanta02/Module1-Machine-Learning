# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 12:24:04 2022

@author: niyan
"""

""" Module 1 - lab1 supervised learning
built some learning curve"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = 2 * np.random.rand(100,1)
y = 4 + 3 * x + np.random.rand(100,1)

def plot_learning_curve_niyanta(model,x,y):
    x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    train_error, test_error= [],[]
    for m in range(1,len(x_train)):
        model.fit(x_train[:m],y_train[:m])
        y_train_predict = model.predict(x_train[:m])
        y_test_predict = model.predict(x_test)
        train_error.append(mean_squared_error(y_train[:m],y_train_predict[:m]))
        test_error.append(mean_squared_error(y_test,y_test_predict))
    plt.plot(np.sqrt(train_error), "r-+" , linewidth=2, label = "train")
    plt.plot(np.sqrt(test_error), "b-+" , linewidth=3, label = "test")    
    plt.axis([0,50,0,10])
    plt.xlabel("training size")
    plt.ylabel("rmse")
    plt.title(model)
    plt.show()
"""desfine an object lin_reg using linear regression"""
lin_reg=LinearRegression()
plot_learning_curve_niyanta(lin_reg, x, y)    



"""pipline  : jusy a way to organise the code and make it neater"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_regression_3 = Pipeline([
    ("poly_features_3", PolynomialFeatures(degree=3, include_bias=False)),
       ("lin_reg", LinearRegression()),
    ])
polynomial_regression_10 = Pipeline([
    ("poly_features_10", PolynomialFeatures(degree=10, include_bias=False)),
       ("lin_reg", LinearRegression()),
    ])
plot_learning_curve_niyanta(polynomial_regression_3, x, y)
plot_learning_curve_niyanta(polynomial_regression_10, x, y)








