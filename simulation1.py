import random
import numpy as np
import matplotlib.pyplot as plt

def simulation(theta, eta, tao, w, x, dt,zi):
	deltaw = 1
	while abs(deltaw)>1E-5:
	        z=random.random()
	        if z>zi:
		   x=0
		y=w*x
		dtheta=(-theta+pow(y,2))/tao
		ntheta=theta+dtheta*dt
 		dw=eta*x*(pow(y,2)-y*ntheta)
 		wn=w+dt*dw
 		deltaw=wn-w	
 		theta=ntheta
 		w=wn

 		print "theta: "+str(ntheta)
 		print "w: "+str(w)

 		print wn
 		x = 1

 	return w*x

# params

# numerical integration: time-step dt=1

# goal: plot y as a function of z0
# z=P(X=x0),1-z=P(X=0)
# power exponent is p=2

def main():
	z_steps = 13
	zths = np.linspace(0.2, 0.8, z_steps, endpoint=True)

	num_rounds = 1000
	ys = np.empty(0)

	for i in range(0,z_steps):
	# should we simulate a few rounds for specific z and then average the w values, compute y and plot over z
		zys = np.empty(0)
		rounds = num_rounds

		while rounds > 0:
			x0=1	

			# initial conditions
			theta=theta0=0
			eta=5E-4
			tao=5E2

			w=w0=1
			x=x0
			dt = 1

			y = simulation(theta, eta, tao, w, x, dt,zths[i])
			zys = np.append(zys, y)
			rounds = rounds-1
			print "zys :"+str(zys)

		print "i :"+str(i)
		if i == 0:
			ys = sum(zys) / num_rounds
		else:
			ys = np.append(ys,sum(zys)/num_rounds)

		print "ys :"+str(ys)
		
	print ""+str(len(ys))
	plt.clf()
	plt.plot(zths,ys)
	plt.ylabel('y')
	plt.xlabel('z')
	plt.savefig('sim1.jpg')

	return

	
if __name__ == "__main__":
	main()


