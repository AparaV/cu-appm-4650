'''
	Aparajithan Venkateswaran
	APPM 4650
	Assignment 05
'''

import numpy as np
from numpy import linalg as LA

## Gauss-Seidel

def converged(x, x_0, tol):
	err = LA.norm(x - x_0)
	if err > tol:
		return False
	return True

def find_r_i(A, b, i, x_i):
	N = len(A)
	r_i = 0
	for j in range(N):
		r_i -= A[i][j] * x_i[j][0]
	r_i += b[i][0]
	return r_i

def gauss_seidel(A, b, x_0, w, tol=10e-4, isVerbose=False):
	
	# cast as floats
	A = A.astype(np.float64)
	b = b.astype(np.float64)
	x_0 = x_0.astype(np.float64)
	w = float(w)
	
	N = len(x_0)
	x_i = x_0.copy()
	idx = 0
	first = True
	
	while first or not converged(x_i, x_0, tol):
		x_0 = x_i.copy()
		for i in range(N):
			r_i = find_r_i(A, b, i, x_i)
			x_i[i][0] += w * (r_i / A[i][i])
		idx += 1
		if first:
			first = False	
		if isVerbose:
			print(x_i)

	return x_0, idx

## Power Method

def power_method(A, x_0, max_iter=20, isVerbose=False):

	A = A.astype(np.float64)
	x = x_0.astype(np.float64)

	for i in range(max_iter):
		prev = x
		x = np.matmul(A, x)
		if isVerbose:
			print(i)
			print(x)
		if i + 1 != max_iter:
			x = x / np.max(x)
		if isVerbose:
			print(x)
			print("----")

	eigen = []
	for i in range(len(A)):
		eigen.append(x[i][0] / prev[i][0])

	return eigen, x