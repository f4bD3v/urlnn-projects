import random
import numpy as np
import math
from random import choice

def gaussian(i, mu, sigma, g_const):
	exponent = math.exp(-math.pow((i-mu),2)/math.pow(sigma,2))
	return g_const*exponent

def f(y):
	if y > 0:
		return y
	else:
		return 0

def simulate(theta, eta, tao, w, all_is, js, dt):
	deltaw = 1
	unsatisfied = True
	vf = np.vectorize(f)

	while unsatisfied:
		# choose random set of inputs from the five gaussians
		j = np.asscalar(np.random.choice(js))
		print type(j)
		print int(j)
		x = all_is[int(j)]
		y=vf(w*x)
		print y

		dtheta=(-theta+pow(y,2))/tao
		ntheta=theta+dtheta
 		dw=eta*x*(pow(y,2)-y*theta)
 		wn=w+dt*dw
 		if wn.any() < 0:
 			break
 		deltaw=wn-w	
 		if deltaw.all() < 1E-5:
 			unsatisfied = False
 		theta=ntheta
 		w=wn

 		print "theta: "+str(ntheta)
 		print "w: "+str(w)

 	return y 

# params

# numerical integration: time-step dt=1

# goal: plot y as a function of z0
# z=P(X=x0),1-z=P(X=0)
# power exponent is p=2

def main():

	i = np.linspace(1, 100, 100, endpoint=True)
	print i
	js = np.linspace(0, 4, 5, endpoint=True)

	mu = [10,30,50,70,90]
	sigma = 10

	gf = np.vectorize(gaussian, excluded=['mu', 'sigma', 'g_const'])
	g_const = 1/math.sqrt(2*math.pi*math.pow(sigma,2))
	all_is = [gf(i, j, sigma, g_const) for j in mu]

	unsatisfied=True

	mu_w0 = 3.0
	sigma_w0 = 1.0 # ^= sd
	rounds = 1000

	while rounds > 0:
		eta = 5E-2
		tao = 1E2
		dt = 1
	
		w = [np.random.normal(mu_w0, sigma_w0) for elem in i]
		theta = 2.5

		yf = simulate(theta, eta, tao, w, all_is, js, dt)
		# force w_i >= 0
		#rounds=rounds-1
		rounds=0

	return

	
if __name__ == "__main__":
	main()


