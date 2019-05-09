import numpy as np
import json

from matplotlib import pyplot as plt

rf = json.load(open("5_real.json", "r"))
af = json.load(open("5_art.json", "r"))
rt = json.load(open("10_real.json", "r"))
at = json.load(open("10_art.json", "r"))

names = ['RUCB_regrets', 'BTM_regrets', 'RMED2_regrets', 'DOUBLER_regrets', 'RCS_regrets', 'SAVAGE_regrets', 'IF_regrets', 'RMED1_regrets']
colors = ['b','g','r','m']

for i in range(len(names)):

	key = names[i]
	a = rf[key]
	b = af[key]
	c = rt[key]
	d = at[key]

	if key not in ['RUCB_regrets', 'RCS_regrets']:
		
		samples, iterations = len(a), len(a[0])
		
		avg_a = np.zeros(iterations)
		avg_b = np.zeros(iterations)
		avg_c = np.zeros(iterations)
		avg_d = np.zeros(iterations)
		
		for sample in a:
			avg_a += np.array(sample)
		avg_a /= samples
		
		for sample in b:
			avg_b += np.array(sample)
		avg_b /= samples
		
		for sample in c:
			avg_c += np.array(sample)
		avg_c /= samples

		for sample in d:
			avg_d += np.array(sample)
		avg_d /= samples

		plt.plot(avg_a, label='Real, K = 5', color=colors[0])
		plt.plot(avg_b, label='Artificial, K = 5', color=colors[1])
		plt.plot(avg_c, label='Real, K = 10', color=colors[2])
		plt.plot(avg_d, label='Artificial, K = 10', color=colors[3])

	else:
		plt.plot(a, label='Real, K = 5', color=colors[0])
		plt.plot(b, label='Artificial, K = 5', color=colors[1])
		plt.plot(c, label='Real, K = 10', color=colors[2])
		plt.plot(d, label='Artificial, K = 10', color=colors[3])


	plt.legend()
	plt.grid(True)
	plt.title(key.split('_')[0])
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('Time')
	plt.ylabel('Cumulative Regret')
	# plt.show()
	plt.savefig(key.split('_')[0]+'.png')
	plt.clf()


