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

def simulate(theta, eta, tao, w, all_is, js, dt):
     deltaw = 1
     unsatisfied = True
     vf = np.vectorize(f)
     #keep track of the timevarying quantities theta, y and objective function
     ct = 0
     thettime = np.empty(0);
     y1time = np.empty(0);    
     y2time = np.empty(0);    
     y3time = np.empty(0);    
     y4time = np.empty(0);    
     y5time = np.empty(0);
     

     while unsatisfied:
        # choose random set of inputs from the five gaussians
        j = np.asscalar(np.random.choice(js))
        #print "Simulation "+str(ct)
        #print "type j:" + str(type(j))
        #print "int j: "+str(int(j))
        x = all_is[int(j)]
        #print "x:"+str(x)
        y=f(np.dot(w,x))
        #print "y:"+str(y)

        dtheta=(-theta+pow(y,2))/tao
        ntheta=theta+dtheta*dt
        dw=eta*x*(pow(y,2)-y*ntheta)
        wn=w+dt*dw

        # constrain w to zero
        wn = vf(wn)
          
        deltaw=wn-w    
           
        if all([abs(dw)<1E-50 for dw in deltaw]):            #corrected here: absolute value
                unsatisfied = False
        theta=ntheta
        w=wn

        #print "theta: "+str(ntheta)
        #print "w: "+str(w)
        if ct%500 == 0:
                        np.append(thettime,ntheta)
                        np.append(y1time,np.dot(all_is[0],wn))
                        np.append(y2time,np.dot(all_is[1],wn))
                        np.append(y3time,np.dot(all_is[2],wn))
                        np.append(y4time,np.dot(all_is[3],wn))
                        np.append(y5time,np.dot(all_is[4],wn))
        ct = ct+1
        print "iterations:"+str(ct)
     return (w,thettime,y1time,y2time,y3time,y4time,y5time)

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

     unsatisfied=True

     mu_w0 = 3.0
     sigma_w0 = 1.0 # ^= sd
     num_rounds = 1
     rounds = num_rounds
     
     weights = np.zeros((100,))
     print "weight:"+str(weights.shape)

     while rounds > 0:
        print "Rounds: "+str(rounds)
        eta = 5E-2
        tao = 1E2
        dt = 1
     
        w = np.array([np.random.normal(mu_w0, sigma_w0) for elem in i])
        theta = 2.5

        ret = simulate(theta, eta, tao, w, all_is, js, dt)
        w = ret[0]
        plt.plot(ret[1])
        plt.plot(ret[2])
        plt.plot(ret[3])
        plt.plot(ret[4])
        plt.plot(ret[5])
        plt.plot(ret[6])
        plt.show()
                                  
        rounds=rounds-1
        #average the weights over the different rounds
        weights = np.add(weights, w/num_rounds);
     print w
     #plt.plot(i.T,weights.T)
     #plt.savefig('sim21.jpg')
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
     return


     

     
if __name__ == "__main__":
     main()


