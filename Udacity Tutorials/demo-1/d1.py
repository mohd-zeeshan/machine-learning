# Intro to Deep Learning - Udacity Nanodegree
# https://www.youtube.com/watch?v=XdM6ER7zTLk&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3&index=2

from numpy import *

# google: sum of squared errors for the equation
def compute_error_for_given_points(b, m, points):
	totalError = 0
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y - (m * x + b)) ** 2
	return totalError / float(len(points))


# google: partial derivative
def step_gradient(curr_b, curr_m, points, learn_rate):
	# gradient decent
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		b_gradient += -(2/N) * (y - ((curr_m * x) + curr_b))
		m_gradient += -(2/N) * x * (y - ((curr_m * x) + curr_b))
	new_b = curr_b - (learn_rate * b_gradient)
	new_m = curr_m - (learn_rate * m_gradient)
	return [new_b, new_m]
	
	
def gradient_decent_runner(points, start_b, start_m, learn_rate, num_iter):
	b = start_b
	m = start_m
	
	for i in range(num_iter):
		b, m = step_gradient(b, m, array(points), learn_rate)
	return [b, m]

	
def run():
	points = genfromtxt('data.csv', delimiter=',')
	
	# hyperparameters - don't always know; guess and check
	learn_rate = 0.0001 
	
	# y=mx+b
	initial_b = 0
	initial_m = 0
	
	num_iter = 1000

	[b, m] = gradient_decent_runner(points, initial_b, initial_m, learn_rate, num_iter)
	print("b: ", b)
	print("m: ", m)
	print("e: ", compute_error_for_given_points(b, m, points))
	
if __name__ == '__main__':
	run()