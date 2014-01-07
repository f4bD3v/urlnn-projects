import race 

eps_list = [0.1, 0.2, 0.3, 0.6, 0.9]

for eps in eps_list:
	final_car = race.average_trainings_last_trials(eps)
	#race.show_race()

eps = .9
final_car = race.average_trainings_last_trials(eps, true)

# plot several learning curves for different values of eps quantify the performance by averaging the latencies in the last 10 trials	