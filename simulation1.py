import random
import numpy as np

def simulation(theta, eta, tao, w, x, dt):
	deltaw = 1
	while deltaw>1E-5:
		y=w*x
		dtheta=(-theta+pow(y,2))/tao
		ntheta=theta+dtheta
 		dw=eta*x*(pow(y,2)-y*theta)
 		wn=w+dt*dw
 		deltaw=wn-w	
 		theta=ntheta
 		w=wn

 		print "theta: "+str(ntheta)
 		print "w: "+str(w)

 		print wn

 	return w*x

# params

# numerical integration: time-step dt=1

# goal: plot y as a function of z0
# z=P(X=x0),1-z=P(X=0)
# power exponent is p=2

def main():
	z_steps = 13
	zths = np.linspace(0.2, 0.8, z_steps, endpoint=True)

	num_rounds = 10
	ys = np.empty(0)

	for i in range(0,z_steps):
	# should we simulate a few rounds for specific z and then average the w values, compute y and plot over z
		zys = np.empty(0)
		rounds = num_rounds

		while rounds > 0:
			x0=1
			z=random.random()
			if z>zths[i]:
				x0=0	

			# initial conditions
			theta=theta0=0
			eta=5E-4
			tao=5E2

			w=w0=1
			x=x0
			dt = 1

			y = simulation(theta, eta, tao, w, x, dt)
			zys = np.append(zys, y)
			rounds = rounds-1
			print "zys :"+str(zys)

		print "i :"+str(i)
		if i == 0:
			ys = zys
		else:
			ys = np.append(ys, zys, axis=0)

		print "ys :"+str(ys)

	return

	
if __name__ == "__main__":
	main()


