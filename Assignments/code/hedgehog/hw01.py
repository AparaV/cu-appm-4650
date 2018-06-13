'''
	Aparajithan Venkateswaran
	APPM 4650
	Assignment 01
'''

import numpy as np

#
# Convergence Testing
#

def x_ratio_err(x1, x2, tol):
	'''
	Ratio of last two root estimates
	'''
	if x1 != 0:
		err = (x2 - x1) / x1
	else:
		err = tol + 1
	return np.abs(err)

def has_converged(x1, x2, tol, test="x_ratio", isVerbose=False):
	'''
	Checks if the sequence has converged
		x1   - Old estimate
		x2   - New estimate
		tol  - accepted tolerance (error) value
		test - what type of error test should be performed
			Defaults to 'x_ratio'
	'''

	if test == "x_ratio":
		err = x_ratio_err(x1, x2, tol)

	if isVerbose:
		print("Error: {:.8f}".format(err))
	if err <= tol:
		return True
	return False

#
# Root Finding Techniques
#

def bisection(f, a, b, max_iter=None, tol=1e-4, test="x_ratio", isVerbose=False):
	'''
	Bisection method for root finding
		f         - function whose root is to be found
		a, b      - first guess, second guess. Note that (a,b) must straddle the root
		max_iter  - maximum number of iterations before stopping
		 	Defaults to infinte (which is bad...)
		tol       - tolerance value for convergence
		test      - type of test used to check for convergence.
			Defaults to 'x_ratio'
		isVerbose - Prints results from each iteration
	'''

	if max_iter is None:
		max_iter = float("inf")

	y1 = f(a)
	y2 = f(b)

	# check for sign change or already found root
	if y1 * y2 > 0:
		raise ValueError("Criteria (sign change in f) for a and b not met.")
	if y1 == 0:
		if isVerbose:
			print("Already found root")
		return a
	if y2 == 0:
		if isVerbose:
			print("Already found root")
		return b

	# loop until convergence or max_iter
	i = 1
	while i <= max_iter:

		# update best guess
		c = (a + b) / 2
		y3 = f(c)
		if isVerbose:
			print("Iter: {:3}\t Best Guess: {:.6f}".format(i, c))

		# check for convergence or divergence
		if y3 == 0 or has_converged(a, c, tol=tol, test=test):
			if isVerbose:
				print("Converged")
				has_converged(a, c, tol=tol, test=test, isVerbose=isVerbose)
			return c
		if c == float("inf") or c == float("nan"):
			if isVerbose:
				print("Divergent")
			return c
		
		# update values based on sign change
		if y1 * y3 < 0:
			temp = b
			b = c
			y2 = y3
		else:
			temp = a
			a = c
			y1 = y3
		i += 1

	has_converged(temp, c, tol=tol, test=test, isVerbose=isVerbose)
	return c

def newton_raphson(f, fp, x0, max_iter=None, tol=1e-4, test="x_ratio", isVerbose=False):
	'''
	Newton-Raphson method for root finding
		f         - function whose root is to be found
		fp        - first derivative of function f
		x0        - initial root guess
		max_iter  - maximum number of iterations before stopping
		 	Defaults to infinte (which is bad...)
		tol       - tolerance value for convergence
		test      - type of test used to check for convergence.
			Defaults to 'x_ratio'
		isVerbose - Prints results from each iteration
	'''

	if max_iter is None:
		max_iter = float("inf")

	# loop until convergence or max_iter
	i = 1
	while i <= max_iter:

		# update best guess
		if fp(x0) != 0:
			x1 = x0 - f(x0) / fp(x0)
		else:
			x1 = float("-inf")
		if isVerbose:
			print("Iter: {:3}\t Best Guess: {:.6f}".format(i, x1))

		# check for convergence or divergence
		if has_converged(x0, x1, tol=tol, test=test):
			if isVerbose:
				print("Converged")
				has_converged(x0, x1, tol=tol, test=test, isVerbose=isVerbose)
			return x1
		if x1 == float("inf") or x1 == float("-inf") or x1 == float("nan"):
			if isVerbose:
				print("Divergent")
			return x1
		
		# update values
		temp = x0
		x0 = x1
		i += 1

	has_converged(temp, x1, tol=tol, test=test, isVerbose=isVerbose)
	return x0

def fixed_point_iter(g, x0, max_iter=None, tol=1e-4, test="x_ratio", isVerbose=False):
	'''
	Fixed point iteration method for root finding
		g         - function such that x - g(x) = f(x) = 0
		x0        - initial root guess
		max_iter  - maximum number of iterations before stopping
		 	Defaults to infinte (which is bad...)
		tol       - tolerance value for convergence
		test      - type of test used to check for convergence.
			Defaults to 'x_ratio'
		isVerbose - Prints results from each iteration
	'''

	if max_iter is None:
		max_iter = float("inf")

	# loop until convergence or max_iter
	i = 1
	while i <= max_iter:

		# update best guess
		x1 = g(x0)
		if isVerbose:
			print("Iter: {:3}\t Best Guess: {:.6f}".format(i, x1))

		# check for convergence or divergence
		if has_converged(x0, x1, tol=tol, test=test):
			if isVerbose:
				print("Converged")
				has_converged(x0, x1, tol=tol, test=test, isVerbose=isVerbose)
			return x1
		if x1 == float("inf") or x1 == float("nan"):
			if isVerbose:
				print("Divergent")
			return x1

		# update values
		temp = x0
		x0 = x1
		i += 1

	has_converged(temp, x1, tol=tol, test=test, isVerbose=isVerbose)
	return x0

def secant(f, x0, x1, max_iter=None, tol=1e-4, test="x_ratio", isVerbose=False):
	'''
	Secant method for root finding
		f         - function whose root is to be found
		x0        - first root guess
		x1        - second root guess
		max_iter  - maximum number of iterations before stopping
		 	Defaults to infinte (which is bad...)
		tol       - tolerance value for convergence
		test      - type of test used to check for convergence.
			Defaults to 'x_ratio'
		isVerbose - Prints results from each iteration
	'''

	if max_iter is None:
		max_iter = float("inf")

	# loop until convergence or max_iter
	i = 1
	while i <= max_iter:

		# update best guess
		x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
		if isVerbose:
			print("Iter: {:3}\t Best Guess: {:.6f}".format(i, x2))
		
		# check for convergence or divergence
		if has_converged(x1, x2, tol=tol, test=test):
			if isVerbose:
				print("Converged")
				has_converged(x1, x2, tol=tol, test=test, isVerbose=isVerbose)
			return x2
		if x2 == float("inf") or x2 == float("nan"):
			if isVerbose:
				print("Divergent")
			return x2

		# update values
		x0 = x1
		temp = x1
		x1 = x2
		i += 1

	has_converged(temp, x2, tol=tol, test=test, isVerbose=isVerbose)
	return x0, x1

def false_position(f, x0, x1, max_iter=None, tol=1e-4, test="x_ratio", isVerbose=False):
	'''
	False position method for root finding
		f         - function whose root is to be found
		x0        - first root guess
		x1        - second root guess
		max_iter  - maximum number of iterations before stopping
		 	Defaults to infinte (which is bad...)
		tol       - tolerance value for convergence
		test      - type of test used to check for convergence.
			Defaults to 'x_ratio'
		isVerbose - Prints results from each iteration
	'''

	if max_iter is None:
		max_iter = float("inf")

	y1 = f(x0)
	y2 = f(x1)

	# check for sign change or already found root
	if y1 * y2 > 0:
		raise ValueError("Criteria (sign change in f) for x0 and x1 not met.")
	if y1 == 0:
		if isVerbose:
			print("Already found root")
		return x0
	if y2 == 0:
		if isVerbose:
			print("Already found root")
		return x1

	# loop until convergence or max_iter
	i = 1
	while i <= max_iter:

		# update best guess
		x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
		y3 = f(x2)
		if isVerbose:
			print("Iter: {:3}\t Best Guess: {:.6f}".format(i, x2))

		# check for convergence or divergence
		if x1 < x0:
			t = x1
		else:
			t = x0
		if y3 == 0 or has_converged(t, x2, tol=tol, test=test):
			if isVerbose:
				print("Converged")
				has_converged(t, x2, tol=tol, test=test, isVerbose=isVerbose)
			return x2
		if x2 == float("inf") or x2 == float("nan"):
			if isVerbose:
				print("Divergent")
			return x2

		# update values based on sign change
		if y1 * y3 < 0:
			temp = x1
			x1 = x2
			y2 = y3
		else:
			temp = x0
			x0 = x2
			y1 = y3
		i += 1

	has_converged(temp, x2, tol=tol, test=test, isVerbose=isVerbose)
	return x2

def steffenson(g, x0, max_iter=None, tol=1e-4, test="x_ratio", isVerbose=False):
	'''
	Steffensen method for find roots
		g - function sucht that x - g(x) = f(x) = 0
		x0 - initial root guess
		max_iter  - maximum number of iterations before stopping
		 	Defaults to infinte (which is bad...)
		tol       - tolerance value for convergence
		test      - type of test used to check for convergence.
			Defaults to 'x_ratio'
		isVerbose - Prints results from each iteration
	'''

	if max_iter is None:
		max_iter = float("inf")

	# loop until convergence or max_iter
	i = 1
	while i <= max_iter:

		# update best guess
		x1 = g(x0)
		x2 = g(x1)
		# print(x1)
		# print(x2)
		x = x0 - np.power((x1 - x0), 2) / (x2 - 2*x1 + x0)
		if isVerbose:
			print("Iter: {:3}\t Best Guess: {:.6f}".format(i, x))

		# check for convergence or divergence
		if x == 0 or has_converged(x0, x, tol=tol, test=test):
			if isVerbose:
				print("Converged")
				has_converged(x0, x, tol=tol, test=test, isVerbose=isVerbose)
			return x
		if x == float("inf") or x == float("nan"):
			if isVerbose:
				print("Divergent")
			return x

		# update values
		temp = x0
		x0 = x
		i += 1

	has_converged(temp, x, tol=tol, test=test, isVerbose=isVerbose)
	return x0