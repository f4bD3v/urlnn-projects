from pylab import *
import math

# this is a dummy class, use it as template inserting your algorithm.

class car:
   
    def __init__(self):

        # setup your parameters here.
        self.n_actions = 9
        self.n_neurons = 1082
        self.weights = zeros((self.n_actions,self.n_neurons))
        self.epsilon = 0.1
        self.qs = zeros(self.n_actions)
        self.sigmap = 1./30
        self.sigmav = 0.2
        self.posGridDist = 1./30
        self.velGridDist = 1./30
        self.old_q= None        
        self.old_rp = None
        self.old_rv = None
        self.old_action = None

    def reset(self) :
    
        # reset values before each trial.
        
        self.time = 0
        self.eligibility_trace = zeros((self.n_actions,self.n_neurons))
        self.gamma = 0.95
        self.eta = 0.005
        self.lambdaa = 0.95
        self.old_q = None
        self.old_rp = None
        self.old_rv = None
        self.old_action = None

    def choose_action(self, position, velocity, R, learn = True):
        # This method must:
        # -choose your action based on the current position and velocity.
        # -update your parameters according to SARSA. This step can be turned off by the parameter 'learn=False'.
        #
        # The [x,y] values of the position are always between 0 and 1.
        # The [vx,vy] values of the velocity are always between -1 and 1.
        # The reward from the last action R is a real number

        posx,posy = position[0],position[1]
        velx,vely = velocity[0],velocity[1]

    	rp = [math.exp(-(math.pow(posx-(j%31)*self.posGridDist,2) + math.pow(posy-(int(j/31))*self.posGridDist,2))/2/math.pow(self.sigmap,2)) for j in range(961)]
        rv = [math.exp(-(math.pow(velx-(j%31)*self.velGridDist,2) + math.pow(vely-(int(j/31))*self.velGridDist,2))/2/math.pow(self.sigmav,2)) for j in range(121)]
        qs = [dot(self.weights[i,0:961],rp) + dot(self.weights[i,961:1082],rv) for i in range(9)]
    	
        if(rand()<self.epsilon):
            action = int(rand()*9)  
        else:
            #action that leads to maximal Q
            action = qs.index(max(qs))
    	qact = qs[action] #Q corresponding to the taken action
        
        if learn and self.old_action != None:   
            # I am anything but certain about this. I don't really get the s' and a' stuff :S
            #calculate TD error
            delta = R - (self.old_q - self.gamma*qact)
            #update eligibility trace
            self.eligibility_trace = self.eligibility_trace*self.lambdaa*self.gamma
            self.eligibility_trace[self.old_action,:] += append(self.old_rp,self.old_rv)
            #update weights
            self.weights += self.eta*delta*self.eligibility_trace
    	
        self.old_rv = rv
        self.old_rp = rp
        self.old_action = action
        self.old_q = qact  
    	self.time += 1

    	return action


    









       