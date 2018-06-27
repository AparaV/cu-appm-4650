'''
	Aparajithan Venkateswaran
	APPM 4650
	Assignment 03
'''

import numpy as np

## Trapezoidal Rule

def trapezoidal(f, x0, xn, h):
	''' Use trapezoidal rule to approximate the integral of f '''

	X = np.arange(x0, xn, step=h, dtype=float)  # create list of points
	Y = [f(x) for x in X]                       # create list of function values
	Y.append(f(xn))                             # end point is missed by arange
	n = len(Y)
	ans = 0.0
	for i in range(n-1):
		ans += Y[i] + Y[i+1]
	ans *= h / 2

	return ans

## Simpson's Rule

def simpson_1_3(f, x0, xn, h):
	''' Use Simpson's 1/3 rule to approximate the integral of f '''

	X = np.arange(x0, xn, step=h, dtype=float)  # create list of points
	Y = [f(x) for x in X]                       # create list of function values
	Y.append(f(xn))                             # end point is missed by arange
	n = len(Y)
	ans = 0.0
	i = 0
	while i < n-2:
		ans += Y[i] + 4*Y[i+1] + Y[i+2]
		i += 2
	ans *= h / 3

	return ans

## Gaussian Quadrature

def gauss_2(f):
	''' Gaussian Quadrature approximation with n = 2 '''

	x0 = -1 / np.sqrt(3)
	x1 = 1 / np.sqrt(3)
	c0 = 1
	c1 = 1
	ans = c0 * f(x0) + c1 * f(x1)

	return ans

def gauss_3(f):
	''' Gaussian Quadrature approximation with n = 3 '''
	
	x0 = -np.sqrt(3 / 5)
	x1 = 0
	x2 = np.sqrt(3 / 5)
	c0 = 5 / 9
	c1 = 8 / 9
	c2 = 5 / 9
	ans = c0 * f(x0) + c1 * f(x1) + c2 * f(x2)

	return ans
