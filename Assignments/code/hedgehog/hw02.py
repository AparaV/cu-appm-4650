'''
	Aparajithan Venkateswaran
	APPM 4650
	Assignment 02
'''

import numpy as np

## Interpolating Polynomials

def lagrange_poly(X, Y, a, isVerbose=False):
	''' Construct the lagrange approximation at a using data points '''

	assert len(X) == len(Y)
	
	ans = 0.0
	for idx, x in enumerate(X):
		prod = Y[idx]
		for x_i in X:
			if x_i != x:
				prod *= (a - x_i) / (x - x_i)
		ans += prod

	if isVerbose:
		print("f({}) = {:.6f} using Lagrange polynomial approximation of degree {}".
			format(str(a), ans, len(X)-1))

	return ans

def neville_method(X, Y, a, isVerbose=False):
	''' Construct the lagrange approximation at a using Neville's method '''

	assert len(X) == len(Y)

	# create matrix
	Q = [[0]*len(X) for i in range(len(X))]
	# initialize base case
	for i in range(0, len(Q)):
		Q[i][0] = Y[i]

	# compute remaining values
	for i in range(1, len(Q)):
		for j in range(1, i+1):
			Q[i][j] = (a - X[i-j]) / (X[i] - X[i-j]) * Q[i][j-1]
			Q[i][j] += (a - X[i]) / (X[i-j] - X[i]) * Q[i-1][j-1]

	if isVerbose:
		print("f({}) = {:.6f} using Lagrange polynomial approximation of degree {}".
			format(str(a), Q[len(X)-1][len(X)-1], len(X)-1))

	return Q