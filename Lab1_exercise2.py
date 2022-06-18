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

##concationation ones
x_b = np.c_[np.ones((100,1)),x]

theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
print(theta_best)

##make predictions
x_new = np.array([[0],[2]])
x_new_b= np.c_[np.ones((2,1)),x_new]
y_predict = x_new_b.dot(theta_best)
y_predict


##use sklearn
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)
print(lin_reg.intercept_,lin_reg.coef_)
lin_reg.predict(x_new)

###the linear regression class is based on scipy.lstsq() function
theta_best_svd, residuals , rank ,s = np.linalg.lstsq(x_b,y,rcond=1e-6)
print(theta_best_svd)

#### using gradient descent 
eta=0.1 ##learning rate
n_iterations=1000
m=1000
theta=np.random.randn(2,1) #random initialization

for iteration in range(n_iterations):
    gradients = 2/m * x_b.T.dot(x_b.dot(theta) - y)
    theta = theta - eta * gradients
    
print(theta)
plt.plot(x_new, y_predict, "r-")
plt.plot(x, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

#### ## doing Stochastic Gradient Descent USING SKLEARN
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters
m = 100
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)  # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
print (gradients)
## doing Stochastic Gradient Descent USING SKLEARN
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(x, y.ravel())
print(sgd_reg.intercept_, sgd_reg.coef_)

 ### mini batch
       


