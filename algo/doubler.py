import numpy as np
import math
import json
import matplotlib.pyplot as plt

class sbm():

	def __init__(self, n_arms):
		
		self.alpha = 1
		self.k = n_arms

	def reset(self):

		self.averages = np.ones(self.k)*math.inf
		self.pulls = np.ones(self.k)
		self.time = 1

	def advance(self):

		bound = self.averages + np.sqrt((self.alpha+2)*np.log(self.time)/(2*self.pulls))
		choice = np.argmax(bound)
		return choice

	def feedback(self, arm, reward):

		self.time += 1
		
		if self.averages[arm] == math.inf:
			self.averages[arm] = reward
		else:
			self.averages[arm] = (self.averages[arm]*self.pulls[arm] + reward)/(self.pulls[arm]+1)
		
		self.pulls[arm] += 1


class Doubler():

	def __init__(self, horizon, pref, regret_func):

		self.pref_matrix = np.array(pref)
		n_arms = len(pref[0])

		self.sbm = sbm(n_arms)

		self.l = np.random.randint(n_arms, size=1)
		self.i, self.t = 1, 1

		self.horizon = horizon
		self.regret_func = regret_func

	def run(self):

		regret = [0]
		
		while self.t < self.horizon:
						
			self.sbm.reset()
			new_l = set()

			for j in range(2**self.i):
				xt = np.random.choice(self.l)
				yt = self.sbm.advance()
				new_l.add(yt)
				bt = np.random.binomial(1, self.pref_matrix[xt][yt], 1)

				if bt == 1:
					self.sbm.feedback(yt, 0)
					self.sbm.feedback(xt, 1)
				else:
					self.sbm.feedback(yt, 1)
					self.sbm.feedback(xt, 0)

				# self.sbm.pulls[xt] += 1

				regret.append(regret[-1]+self.regret_func(xt,yt))

				self.t += 1

				if self.t >= self.horizon:
					break

			self.l = np.array(list(new_l))
			self.i += 1

		return list(np.around(regret,3)), np.argmax(self.sbm.averages)

# def master():

# 	samples = 2

# 	algo = Doubler(10000, np.ones((4,4))-0.5)
# 	regret_final = algo.run()

# 	for sample in range(1,samples):
# 		print(sample)
# 		algo = Doubler(10000, np.ones((4,4))-0.5)
# 		regret_final += algo.run()

# 		# print(regret_final)

# 	regret_final /= samples

# 	with open('doubler.json', 'w') as f:
# 		json.dump({'regret':list(regret_final)}, f, indent=4)

# 	# plt.xscale('log')
# 	plt.plot(regret_final)
# 	plt.show()

# master()
