import random
import numpy as np
import math
from random import choice
import matplotlib.pyplot as plt

def gaussian(i, mu, sigma, g_const):
	exponent = math.exp(-math.pow((i-mu),2)/math.pow(sigma,2))
	return g_const*exponent

def f(y):
	if y > 0:
		return y
	else:
		return 0

def time_step(theta, eta, tao, w, x, y_t):
	vf = np.vectorize(f)
	'''
	deltaw = 1
	vf = np.vectorize(f)
	ct = 0
	#keep track of the time-varying quantities
	thetatime = np.empty(0)
	y1time = np.empty(0)
    y2time = np.empty(0)
    y3time = np.empty(0)
    y4time = np.empty(0)
    y5time = np.empty(0)
    ys = np.empty(0)
    Fs = np.empty(0)

	while unsatisfied:
		# choose random set of inputs from the five gaussians
				#print "x:"+str(x)
	'''
	# update y
	y=f(np.dot(w,x))
	y_t = np.append(y_t, y)
	#print "y: "+str(y)

	# update theta
	dtheta=(pow(y,2)-theta)/tao
	ntheta=theta+dtheta

	# update w
 	dw=np.multiply(eta*(pow(y,2)-y*ntheta), x)
 	wn=w+dw

 	# constrain w to zero
	w = vf(wn)
	#print w	

 	theta = ntheta
 	#print "theta: "+str(theta)

	#deltaw=wn-w	
	Fs = np.mean(np.power(y_t,3))/np.sqrt(np.mean(np.power(y_t,2)))

	'''
	do not use convergence, try and get feeling of how many iterations to simulate

	if all([abs(dw)<1E-12 for dw in deltaw]):   

	'''

	return (w, theta, Fs, y_t)

# params
# numerical integration: time-step dt=1

# goal: plot y as a function of z0
# z=P(X=x0),1-z=P(X=0)
# power exponent is p=2

def main():

	i = np.linspace(1, 100, 100, endpoint=True)
	js = np.linspace(0, 4, 5, endpoint=True)

	mu = [10,30,50,70,90]
	sigma = 10

	gf = np.vectorize(gaussian, excluded=['mu', 'sigma', 'g_const'])

	g_const = 1/math.sqrt(2*math.pi*math.pow(sigma,2))
	all_is = [gf(i, j, sigma, g_const) for j in mu]
	print gf(i, 10, sigma, g_const)

	mu_w0 = 3.0
	sigma_w0 = 1.0 # ^= sd
	
	eta = 5E-2
	tao = 1E2
	theta = 2.5

	w = np.array([np.random.normal(mu_w0, sigma_w0) for elem in i])

	#print "all is: "+str(all_is)

	y_t = np.empty(0)
	y1time = np.empty(0)
	y2time = np.empty(0)
	y3time = np.empty(0)
	y4time = np.empty(0)
	y5time = np.empty(0)
	theta_t = np.empty(0)
	Fs_t = np.empty(0)

	rounds = 150000
	for wround in range(rounds):
		j = np.asscalar(np.random.choice(js))
		#print "j: "+str(j)
		x = all_is[int(j)]
		ret = time_step(theta, eta, tao, w, x, y_t)
		w = ret[0]
		theta = ret[1]
		Fs = ret[2]
		y_t = ret[3]
		
		if wround%5000 == 0:
			theta_t = np.append(theta_t, theta)
			y1time = np.append(y1time,np.dot(all_is[0],w))
			y2time = np.append(y2time,np.dot(all_is[1],w))
			y3time = np.append(y3time,np.dot(all_is[2],w))
			y4time = np.append(y4time,np.dot(all_is[3],w))
			y5time = np.append(y5time,np.dot(all_is[4],w))
			Fs_t = np.append(Fs_t, Fs)
			print wround

	print w
	print Fs_t

   	plt.figure(1)
   	t = np.linspace(0, 150000, 30)
   	print t
   	print len(t)
   	print len(Fs_t)

   	plt.plot(t.T, Fs_t.T)
   	plt.show()
   	#plt.savefig('sim2F'+str(50000)+'.jpg')
	plt.figure(2)
	plt.clf()
	plt.plot(i.T,w.T)
	plt.show()
	#plt.savefig('sim20.jpg')

   	'''
	rounds=rounds-1
	#average the weights over the different rounds
	weights = np.add(weights, w/num_rounds);
	print weights

	thetatime = ret[1]
	y1time = ret[2]
	y2time = ret[3]
	y3time = ret[4]
	y4time = ret[5]
	y5time = ret[6]
	Fs = ret[7]
            
    x = rounds
    print "thetat:"+str(thetatime.shape)
    plt.clf()
   	plt.plot(thetatime)
   	plt.savefig('sim21'+str(x)+'.jpg')
    plt.clf()
   	plt.plot(y1time)
   	plt.savefig('sim22'+str(x)+'.jpg')
    plt.clf()
   	plt.plot(y2time)
   	plt.savefig('sim23'+str(x)+'.jpg')
    plt.clf()
   	plt.plot(y3time)
   	plt.savefig('sim24'+str(x)+'.jpg')
    plt.clf()
   	plt.plot(y4time)
   	plt.savefig('sim25'+str(x)+'.jpg')
    plt.clf()
   	plt.plot(y5time)
   	plt.savefig('sim26'+str(x)+'.jpg')
   	plt.clf()

	print y1time
	print y2time
	print y3time
	print y4time
	print y5time        
	
	y1 = np.dot(all_is[0],weights)
	y2 = np.dot(all_is[1],weights)
	y3 = np.dot(all_is[2],weights)
	y4 = np.dot(all_is[3],weights)
	y5 = np.dot(all_is[4],weights)
	print "y1:"+str(y1)
	print "y2:"+str(y2)
	print "y3:"+str(y3)
	print "y4:"+str(y4)
	print "y5:"+str(y5)
	'''

	return
	

	
if __name__ == "__main__":
	main()


