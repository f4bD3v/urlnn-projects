import race 
from pylab import *

eps_list = [0.5, 0.7, 0.9]
colors = ["red", "green", "blue", "yellow"]

eps_times = np.zeros((len(eps_list), 1000))
for i in range(len(eps_list)):
	print i
	eps = eps_list[i]
	avg_times = race.average_trainings_last_trials(eps, True)
	eps_times[i,:]=avg_times
	#race.show_race()

figure(1)

for i in len(eps_list):
	plot(eps_times[i,:], color=colors[i], linewidth=1.0, linestyle='-', label="epsilon: "+str(eps_list[i]) )

ylabel('Latency')
xlabel('Trial')
out_str = 'all_10_cars_latency'+str(eps)+'_avgtime_lasttrials'+str(avgtime_lasttrials)+'.png'
plt.legend(loc='lower center')
savefig(out_str)
