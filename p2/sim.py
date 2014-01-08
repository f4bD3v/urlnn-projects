import race 
from pylab import *

eps_list = [0.1, 0.2, 0.3, 0.6, 0.9]
colors = ["red", "green", "blue", "yellow"]

eps_times = np.zeros(len(eps_list))
for eps in eps_list:
	avg_times = race.average_trainings_last_trials(eps)
	eps_times.append(eps_times)
	#race.show_race()

figure(1)
for i in len(eps_list):
	plot(eps_times[i,:], color=colors[i], linewidth=1.0, linestyle='-', label="epsilon: "+str(eps[i]) )

ylabel('Latency')
xlabel('Trial')
out_str = 'all_10_cars_latency'+str(eps)+'_avgtime_lasttrials'+str(avgtime_lasttrials)+'.png'
plt.legend(loc='lower center')
savefig(out_str)

eps = .9
final_car = race.average_trainings_last_trials(eps, True)

# plot several learning curves for different values of eps quantify the performance by averaging the latencies in the last 10 trials	