from pylab import *

import car
import track

ion()

# This function trains a car in a track. 
# Your car.py class should be able to be accessed in this way.
# You may call this file by:

# import race
# final_car = race.train_car()
# race.show_race(final_car)

def train_car():
    
    close('all')
    
    # create instances of a car and a track
    monaco = track.track()
    ferrari = car.car()
        
    n_trials = 1000
    n_time_steps = 1000  # maximum time steps for each trial
    for j in arange(n_trials):	

        # before every trial, reset the track and the car.
        # the track setup returns the initial position and velocity. 
        (position_0, velocity_0) = monaco.setup()	
        ferrari.reset()
        
        # choose a first action
        action = ferrari.choose_action(position_0, velocity_0, 0)
        
        # iterate over time
        for i in arange(n_time_steps) :	
            
            # the track receives which action was taken and 
            # returns the new position and velocity, and the reward value.
            (position, velocity, R) = monaco.move(action)	
            
            # the car chooses a new action based on the new states and reward, and updates its parameters
            action = ferrari.choose_action(position, velocity, R)	
            
            # check if the race is over
            if monaco.finished is True:
                break
        
        if j%100 == 0:
            # plots the race result every 100 trials
            monaco.plot_world()
            
        if j%10 == 0:
            print 'Trial:', j

    return ferrari #returns a trained car
    
# This function shows a race of a trained car, with learning turned off
def show_race(ferrari):

    close('all')

    # create instances of a track
    monaco = track.track()

    n_time_steps = 1000  # maximum time steps
    
    # choose to plot every step and start from defined position
    (position_0, velocity_0) = monaco.setup(plotting=True)	
    ferrari.reset()

    # choose a first action
    action = ferrari.choose_action(position_0, velocity_0, 0)

    # iterate over time
    for i in arange(n_time_steps) :	

        # inform your action
        (position, velocity, R) = monaco.move(action)	

        # choose new action, with learning turned off
        action = ferrari.choose_action(position, velocity, R, learn=False)	

        # check if the race is over
        if monaco.finished is True:
            break

def show_race_no_rand(ferrari):
    close('all')

    # create instances of a track
    monaco = track.track()

    n_time_steps = 1000  # maximum time steps
    
    # choose to plot every step and start from defined position
    (position_0, velocity_0) = monaco.setup(plotting=True)  
    ferrari.reset()

    # choose a first action
    action = ferrari.choose_action_no_rand(position_0, velocity_0, 0)

    # iterate over time
    for i in arange(n_time_steps) : 

        # inform your action
        (position, velocity, R) = monaco.move(action)   

        # choose new action, with learning turned off
        action = ferrari.choose_action_no_rand(position, velocity, R, learn=False)  

        # check if the race is over
        if monaco.finished is True:
            break

# This function generates the averaged plots for latency and rewards
def average_trainings():
    close('all')
    
    # create instances of a car and a track
    monaco = track.track()
    ferrari = car.car()
        
    n_trials = 1000
    n_time_steps = 1000  # maximum time steps for each trial
    n_indep_cars = 20.
    times = zeros(n_trials)
    rewards = zeros(n_trials)
    avg_times = zeros(n_trials)
    avg_rewards = zeros(n_trials)
    for k in arange(n_indep_cars):
        for j in arange(n_trials):  

            # before every trial, reset the track and the car.
            # the track setup returns the initial position and velocity. 
            (position_0, velocity_0) = monaco.setup()   
            ferrari.reset()
            
            # choose a first action
            action = ferrari.choose_action(position_0, velocity_0, 0)
            
            # iterate over time
            for i in arange(n_time_steps) : 
                
                # the track receives which action was taken and 
                # returns the new position and velocity, and the reward value.
                (position, velocity, R) = monaco.move(action)   
                
                # the car chooses a new action based on the new states and reward, and updates its parameters
                action = ferrari.choose_action(position, velocity, R)   
                
                # check if the race is over
                if monaco.finished is True:
                    break
            
            if j%100 == 0:
                # plots the race result every 100 trials
                monaco.plot_world()
                
            if j%10 == 0:
                print 'Trial:', j

            times[j] = monaco.time
            rewards[j] = monaco.total_reward
        avg_times = avg_times + avg_times/n_indep_cars
        avg_rewards = avg_rewards + avg_rewards/n_indep_cars

    figure(1)
    plot(times)
    ylabel('Latency')
    xlabel('Trial')
    show()
    figure(2)
    plot(rewards)
    ylabel('Total reward')
    xlabel('Trial')
    show()

def train_and_show_navigation():
    close('all')
    
    # create instances of a car and a track
    monaco = track.track()
    ferrari = car.car()
        
    n_trials = 1000
    n_time_steps = 1000  # maximum time steps for each trial
    ferrari.reset()
    ferrari.plot_navigation(0)
    for j in arange(n_trials):  

        # before every trial, reset the track and the car.
        # the track setup returns the initial position and velocity. 
        (position_0, velocity_0) = monaco.setup()   
        ferrari.reset()
        
        # choose a first action
        action = ferrari.choose_action(position_0, velocity_0, 0)
        
        # iterate over time
        for i in arange(n_time_steps) : 
            
            # the track receives which action was taken and 
            # returns the new position and velocity, and the reward value.
            (position, velocity, R) = monaco.move(action)   
            
            # the car chooses a new action based on the new states and reward, and updates its parameters
            action = ferrari.choose_action(position, velocity, R)   
            
            # check if the race is over
            if monaco.finished is True:
                break
        
        if j%100 == 0:
            # plots the race result every 100 trials
            monaco.plot_world()
            
        if j%10 == 0:
            print 'Trial:', j

        if j == 0 or j == 50 or j==100 or j==250 or j==500 or j==750 or j==999:
            ferrari.plot_navigation(j+1)

    return ferrari #returns a trained car