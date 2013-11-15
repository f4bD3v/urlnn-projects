import random
import numpy as np
import math
from random import choice

def gaussian(i, mu, sigma, g_const):
	exponent = math.exp(-math.pow((i-mu),2)/math.pow(sigma,2))
	return g_const*exponent

def simulation(theta, eta, tao, w, x, dt):
 	return w
# params

# numerical integration: time-step dt=1

# goal: plot y as a function of z0
# z=P(X=x0),1-z=P(X=0)
# power exponent is p=2

def main():

	i = np.linspace(1, 100, 100, endpoint=True)
	print i
	js = np.linspace(1, 5, 5, endpoint=True)

	mu = [10,30,50,70,90]
	sigma = 10

	gf = np.vectorize(gaussian, excluded=['mu', 'sigma', 'g_const'])
	g_const = 1/math.sqrt(2*math.pi*math.pow(sigma,2))
	all_is = [gf(i, j, sigma, g_const) for j in mu]

	satisfied=True

	while satisfied:

		# choose random set of inputs from the five gaussians
		j = np.random.choice(js)
		all_is[j]

		mu_w0 = 3.0
		sigma_w0 = 1.0 # ^= sd
		w0 = np.random.normal(mu_w0, sigma_w0)
		theta0 = 2.5
		# force w_i >= 0

	return

	
if __name__ == "__main__":
	main()


