'''
	Aparajithan Venkateswaran
	APPM 4650
	Assignment 04
'''

import numpy as np

## Differential Equations (ODE)

def runge_kutta_4(system, u0, h, t0, tf, isVerbose=True):
	'''
	Runge-Kutta Order 4
	system and u0 are numpy arrays of same size
	h is step size
	t0 is initial time
	tf is final time
	'''

	assert len(system) == len(u0)               # error check

	n = len(system)                             # size of the system
	system = list(system.reshape(n,))           # convert into list for list comprehensions
	ti = t0                                     # initial time
	ui = u0                                     # initial conditions
	N = int((tf - ti) / h)                      # number of steps

	for i in range(N):

		# calculate the different k's
		k1 = h * np.array([f(ti, ui) for f in system]).reshape(n, 1)
		k2 = h * np.array([f(ti + h/2, ui + k1/2) for f in system]).reshape(n, 1)
		k3 = h * np.array([f(ti + h/2, ui + k2/2) for f in system]).reshape(n, 1)
		k4 = h * np.array([f(ti + h, ui + k3) for f in system]).reshape(n, 1)

		# print function values at different times
		if isVerbose:
			output = "t = {:.7f}\t( ".format(ti)
			for i in range(n-1):
				output += "{:.7f}, ".format(ui[i][0])
			output += "{:.7f} )".format(ui[n-1][0])
			print(output)

		# update step
		ui = ui + (k1 + 2*k2 + 2*k3 + k4) / 6
		ti += h

	# print final value
	if isVerbose:
			output = "t = {:.7f}\t( ".format(ti)
			for i in range(n-1):
				output += "{:.7f}, ".format(ui[i][0])
			output += "{:.7f} )".format(ui[n-1][0])
			print(output)

	return ui

def euler(f, y0, h, x0, xf, isVerbose=True):
	''' Euler's method '''

	# setup initial values
	xi = x0
	yi = y0
	N = int((xf - x0) / h)

	for i in range(N):

		# log values
		if isVerbose:
			print("t = {:.6f}\t{:.6f}".format(xi, yi))

		# update step
		yi = yi + h * f(xi, yi)
		xi += h

	# log final value
	if isVerbose:
		print("t = {:.6f}\t{:.6f}".format(xi, yi))

	return yi

def adams_bashforth_2(f, y0, y1, h, x0, x1, xf, isVerbose=True):
	''' Adams-Bashforth 2 point method '''

	# setup initial values
	xi = x1                  # x_{i}
	xi_1 = x0                # x_{i-1}
	yi = y1                  # x_{i}
	yi_1 = y0                # y_{i-1}
	N = int((xf - x0) / h)

	# use a different h value for first iteration
	h_orig = h
	h = (x0 + h) - x1

	first = True             # keep track of first iteration
	for i in range(N):

		# update step
		temp = yi
		yi = yi + h / 2 * (3*f(xi, yi) - f(xi_1, yi_1))
		yi_1 = temp

		# different update step for first iteration
		# this necessary to forget the false seed value
		h = h_orig
		if first:
			first = False
			xi = x0 + h
			yi_1 = y0
			xi_1 = x0
		else:
			xi_1 = xi
			xi += h

		# log values
		if isVerbose:
			print("t = {:.6f},\tx = {:.6f}".format(xi, yi))

	return yi

def improved_euler(f, y0, h, x0, xf, isVerbose=True):
	''' Improved Euler's method '''

	# setup values
	xi = x0
	yi = y0
	N = int((xf - x0) / h)

	for i in range(N):

		# update step
		temp = yi + h * f(xi, yi)
		yi_new = yi + h * (f(xi, yi) + f(xi + h, temp)) / 2

		# log values
		if isVerbose:
			print("t = {:.6f}\t{:.6f}".format(xi, yi))

		# finish the update
		yi = yi_new
		xi += h

	# log final value
	if isVerbose:
		print("t = {:.6f}\t{:.6f}".format(xi, yi))

	return yi

def predictor_corrector(f, y0, y1, y2, y3, h, x0, x1, x2, x3, xf, isVerbose=True):
	'''
	Predictor Corrector
	Predictor: Adams-Bashforth 4 point method
	Corrector: Adams-Moulton 3 point method
	'''

	# setup initial values
	xi_0 = x3                        # x_{i}
	xi_1 = x2                        # x_{i-1}
	xi_2 = x1                        # x_{i-2}
	xi_3 = x0                        # x_{i-3}
	yi_0 = y3                        # y_{i}
	yi_1 = y2                        # y_{i-1}
	yi_2 = y1                        # y_{i-2}
	yi_3 = y0                        # y_{i-3}
	N = int((xf - x0) / h)

	# log initial values
	if isVerbose:
		print("t = {:.6f},\tx = {:.6f}".format(x0, y0))
		print("t = {:.6f},\tx = {:.6f}".format(x1, y1))
		print("t = {:.6f},\tx = {:.6f}".format(x2, y2))
		print("t = {:.6f},\tx = {:.6f}".format(x3, y3))

	# use a different step size for first iteration
	h_orig = h
	h = (x0 + h) - x3

	first = [True, True, True]       # keep track for first 3 iterations
	for i in range(N):

		# calculate the function values at different times
		fi_0 = f(xi_0, yi_0)
		fi_1 = f(xi_1, yi_1)
		fi_2 = f(xi_2, yi_2)
		fi_3 = f(xi_3, yi_3)

		# predict using AB-4
		pred = yi_0 + h/24 * (55*fi_0 - 59*fi_1 + 37*fi_2 - 9*fi_3)
		# correct using AM-3
		corr = yi_0 + h/24 * (9*f(xi_0+h, pred) +19*fi_0 - 5*fi_1 + fi_2)

		# different update steps for first 3 iterations
		# this is necessary to forget the false seed values we generated
		h = h_orig
		if first[0]:
			yi_3 = y0
			yi_2 = y2
			yi_1 = y1
			yi_0 = corr
			xi_3 = x0
			xi_2 = x2
			xi_1 = x1
			xi_0 = x0 + h
			first[0] = False
		elif first[1]:
			yi_3 = y0
			yi_2 = y2
			yi_1 = yi_0
			yi_0 = corr
			xi_3 = x0
			xi_2 = x2
			xi_1 = xi_0
			xi_0 += h
			first[1] = False
		elif first[2]:
			yi_3 = y0
			yi_2 = yi_1
			yi_1 = yi_0
			yi_0 = corr
			xi_3 = x0
			xi_2 = xi_1
			xi_1 = xi_0
			xi_0 += h
			first[2] = False
		else:
			yi_3 = yi_2
			yi_2 = yi_1
			yi_1 = yi_0
			yi_0 = corr
			xi_3 = xi_2
			xi_2 = xi_1
			xi_1 = xi_0
			xi_0 += h

		# log values
		if isVerbose:
			print("t = {:.6f},\tx = {:.6f}".format(xi_0, yi_0))

	return yi_0

def adams_moulton_2(g, x0, x1, xf, y0, y1, h, isVerbose=True):
	'''
	Adams-Moulton 2 point method
		g is the function that solves for y_{i+1}
		g is *not* the derivative
		g takes in 6 inputs - x_{i-1}, x_{i}, x_{i+1}, y_{i-1}, y_{i}, h
		g returns y_{i+1}
	'''

	# setup initial values
	xi_0 = x1                      # x_{i}
	xi_1 = x0                      # x_{i-1}
	yi_0 = y1                      # y_{i}
	yi_1 = y0                      # y_{i-1}
	N = int((xf - x0) / h)

	# log initial values
	if isVerbose:
		print("t = {:.7f},\tx = {:.7f}".format(x0, y0))
		print("t = {:.7f},\tx = {:.7f}".format(x1, y1))

	# use different step size for first iteration
	h_orig = h
	h = (x0 + h) - x1

	first = True                   # keep track of first iteration
	for i in range(N):

		# calculate
		old_x = [xi_1, xi_0]
		old_y = [yi_1, yi_0]
		xi_0 += h
		yi_0 = g(old_x[0], old_x[1], xi_0, old_y[0], old_y[1], h)

		# log values
		if isVerbose:
			print("t = {:.7f},\tx = {:.7f}".format(xi_0, yi_0))

		# different update step for first iteration
		# this is necessary to forget the false seed value we generated
		h = h_orig
		if first:
			first = False
			yi_1 = y1
			xi_1 = x1
		else:
			yi_1 = old_y[1]
			xi_1 = old_x[1]

	return yi_0