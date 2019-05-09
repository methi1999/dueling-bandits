import numpy as np
import json

from matplotlib import pyplot as plt

d = json.load(open("data/10_art.json", "r"))

names = ['RUCB_regrets', 'BTM_regrets', 'RMED2_regrets', 'DOUBLER_regrets', 'RCS_regrets', 'SAVAGE_regrets', 'IF_regrets', 'RMED1_regrets']
colors = ['b','g','r','c','m','y','k',(0.75,0.5,0.5)]

for i in range(len(names)):

	val = d[names[i]]
	key = names[i]

	if names[i] not in ['RUCB_regrets', 'RCS_regrets']:
		
		samples, iterations = len(val), len(val[0])
		
		avg = np.zeros(iterations)
		
		for sample in val:
			avg += np.array(sample)
		
		avg /= samples
		plt.plot(avg, label=key.split('_')[0], color=colors[i])

	else:
		plt.plot(d[names[i]], label=key.split('_')[0], color=colors[i])


plt.legend()
# plt.savefig('all_plots.png')
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Cumulative Regret')
plt.show()