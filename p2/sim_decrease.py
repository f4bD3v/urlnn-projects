import race 
from pylab import *
from multiprocessing import Pool

eps_list = [1, 0.9, 0.7]
colors = ["red", "green", "blue", "yellow"]

#p = Pool(3)
eps_times=np.zeros((len(eps_list),1000))
#eps_times = p.map(race.average_trainings_last_trials, eps_list)

for i in range(len(eps_list)):
	print i
	eps = eps_list[i]
	avg_times = race.average_trainings_last_trials(eps, False)
	eps_times[i,:]=avg_times
	#race.show_race()

figure(1)

for i in range(len(eps_list)):
	plot(eps_times[i,:], color=colors[i], linewidth=1.0, linestyle='-', label="epsilon: "+str(eps_list[i]))

ylabel('Latency')
xlabel('Trial')
out_str = 'all_10_cars_latency'+str(eps)+'_avgtime_lasttrials'+str(avgtime_lasttrials)+'.png'
plt.legend(loc='lower center')
savefig(out_str)
