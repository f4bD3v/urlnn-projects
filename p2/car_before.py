from pylab import *
import math


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
        self.posGridDist = 1./30 #Since it goes from 0 to 1, with 31 samples
        self.velGridDist = 0.2 #going from -1 to 1, with 11 samples
        self.gamma = 0.95 # discount factor
        self.eta = 0.005 # learning rate
        self.lambdaa = 0.95 # eligibility trace decay rate

    def reset(self) :
    
        # reset values before each trial.
        self.time = 0
        self.eligibility_trace = zeros((self.n_actions,self.n_neurons))
        self.old_q = None
        self.old_rp = None
        self.old_rv = None
        self.old_action = None

    def eval_activation(self, posx, posy, sigma):
        r = (np.exp( -((posx-rp)**2) ) + np.exp( -((posy-vp)**2) )) / (2*(sigma**2))
        return r


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

        v_place = np.linspace(0,1,31)
        v_velocity = np.linspace(-1,1,11)

    	rp = [math.exp(-(math.pow(posx-(j%31)*self.posGridDist,2) + math.pow(posy-(int(j/31))*self.posGridDist,2))/2/math.pow(self.sigmap,2)) for j in range(961)]
        rv = [math.exp(-(math.pow(velx-(j%11)*self.velGridDist-1,2) + math.pow(vely-(int(j/11))*self.velGridDist-1,2))/2/math.pow(self.sigmav,2)) for j in range(121)]
        qs = [dot(self.weights[i,0:961],rp) + dot(self.weights[i,961:1082],rv) for i in range(9)]
    	
        if(rand()<self.epsilon):
            action = int(rand()*9)  
        else:
            #action that leads to maximal Q
            #one could incorporate here a random choice when multiple entries have the same value. 
            #However, this is a marginal case since we're working with continuous variables and hence that wouldn't change too much.
            action = qs.index(max(qs))
    	qact = qs[action] #Q corresponding to the taken action
        
        if learn and self.old_action != None:   
            #calculate TD error
            delta = R + self.gamma*qact - self.old_q
            #update eligibility trace
            self.eligibility_trace = self.eligibility_trace*self.lambdaa*self.gamma
            #add activation for action a (how to get state s?, have to get neuron for location s)
            self.eligibility_trace[self.old_action,:] += append(self.old_rp,self.old_rv)
            #update weights
            self.weights += self.eta*delta*self.eligibility_trace
    	
        self.old_rv = rv
        self.old_rp = rp
        self.old_action = action
        self.old_q = qact  
    	self.time += 1

    	return action


    def plot_navigation(self):
        scale = 3
        xx = 30
        x_dir = zeros((xx+1,xx+1))
        y_dir = zeros((xx+1,xx+1))
        actions = zeros((xx+1,xx+1))
        rv = [math.exp(-(math.pow(0-(j%11)*self.velGridDist-1,2) + math.pow(0-(int(j/11))*self.velGridDist-1,2))/2/math.pow(self.sigmav,2)) for j in range(121)]
        for l in range(xx+1):
            for k in range(xx+1):
                rp = [math.exp(-(math.pow(l/float(xx)-(j%31)*self.posGridDist,2) + math.pow(k/float(xx)-(int(j/31))*self.posGridDist,2))/2/math.pow(self.sigmap,2)) for j in range(961)]
                qs = [dot(self.weights[i,0:961],rp) + dot(self.weights[i,961:1082],rv) for i in range(9)]
                actions[k,l] = qs.index(max(qs))
        x_dir[actions == 0] = 0*scale
        y_dir[actions == 0] = 0*scale
        x_dir[actions == 1] = 0.707*scale
        y_dir[actions == 1] = 0.707*scale
        x_dir[actions == 2] = 1*scale
        y_dir[actions == 2] = 0*scale
        x_dir[actions == 3] = 0.707*scale
        y_dir[actions == 3] = -0.707*scale
        x_dir[actions == 4] = 0*scale
        y_dir[actions == 4] = -1*scale
        x_dir[actions == 5] = -0.707*scale
        y_dir[actions == 5] = -0.707*scale
        x_dir[actions == 6] = -1*scale
        y_dir[actions == 6] = 0*scale
        x_dir[actions == 7] = -0.707*scale
        y_dir[actions == 7] = 0.707*scale
        x_dir[actions == 8] = 0*scale
        y_dir[actions == 8] = 1*scale

        quiver([i/float(xx) for i in range(xx+1)], [i/float(xx+1) for i in range(xx+1)],x_dir,y_dir)
        axis([-0.1,1.1,-0.1,1.1])
        draw()
        show()

    def choose_action_no_rand(self, position, velocity, R, learn = True):
        posx,posy = position[0],position[1]
        velx,vely = velocity[0],velocity[1]

        rp = [math.exp(-(math.pow(posx-(j%31)*self.posGridDist,2) + math.pow(posy-(int(j/31))*self.posGridDist,2))/2/math.pow(self.sigmap,2)) for j in range(961)]
        rv = [math.exp(-(math.pow(velx-(j%31)*self.velGridDist,2) + math.pow(vely-(int(j/31))*self.velGridDist,2))/2/math.pow(self.sigmav,2)) for j in range(121)]
        qs = [dot(self.weights[i,0:961],rp) + dot(self.weights[i,961:1082],rv) for i in range(9)]
        
        #action that leads to maximal Q
        action = qs.index(max(qs))
        qact = qs[action] #Q corresponding to the taken action
        
        if learn and self.old_action != None:   
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

    def choose_action_softmax(self, position, velocity, R, learn = True):
        posx,posy = position[0],position[1]
        velx,vely = velocity[0],velocity[1]

        rp = [math.exp(-(math.pow(posx-(j%31)*self.posGridDist,2) + math.pow(posy-(int(j/31))*self.posGridDist,2))/2/math.pow(self.sigmap,2)) for j in range(961)]
        rv = [math.exp(-(math.pow(velx-(j%31)*self.velGridDist,2) + math.pow(vely-(int(j/31))*self.velGridDist,2))/2/math.pow(self.sigmav,2)) for j in range(121)]
        qs = [dot(self.weights[i,0:961],rp) + dot(self.weights[i,961:1082],rv) for i in range(9)]
        
        '''
            SOFT max
        '''

        # tau - temperature coefficient - low temperature causes all actions to be equiprobable
        tau = .01 # increase to 1 for exploitation
        pas = exp(tau*qacts)/sum(exp(tau*qacts))

        u = rand()

        cpf = [sum(pas[0:i+1]) for i in range(pas.shape[0])]

        for i in range(len(cpf)):
            cp = cpf[i]
            if u <= cp:
               action = i 
               break

        action = qs.index(max(qs))
        qact = qs[action] #Q corresponding to the taken action
        
        if learn and self.old_action != None:   
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


def choose_action_eps(self, position, velocity, R, eps, learn = True):
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
        rv = [math.exp(-(math.pow(velx-(j%11)*self.velGridDist-1,2) + math.pow(vely-(int(j/11))*self.velGridDist-1,2))/2/math.pow(self.sigmav,2)) for j in range(121)]
        qs = [dot(self.weights[i,0:961],rp) + dot(self.weights[i,961:1082],rv) for i in range(9)]
        
        if(rand()<self.epsilon):
            action = int(rand()*9)  
        else:
            #action that leads to maximal Q
            #one could incorporate here a random choice when multiple entries have the same value. 
            #However, this is a marginal case since we're working with continuous variables and hence that wouldn't change too much.
            action = qs.index(max(qs))
        qact = qs[action] #Q corresponding to the taken action
        
        if learn and self.old_action != None:   
            #calculate TD error
            delta = R + self.gamma*qact - self.old_q
            #update eligibility trace
            self.eligibility_trace = self.eligibility_trace*self.lambdaa*self.gamma
            #add activation for action a (how to get state s?, have to get neuron for location s)
            self.eligibility_trace[self.old_action,:] += append(self.old_rp,self.old_rv)
            #update weights
            self.weights += self.eta*delta*self.eligibility_trace
        
        self.old_rv = rv
        self.old_rp = rp
        self.old_action = action
        self.old_q = qact  
        self.time += 1

        return action







       