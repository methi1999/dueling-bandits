import numpy as np
import math
import random
import matplotlib.pyplot as plt

def arg_max(matrix):
	# A is assumed to be a 1D array
	top_indexes = np.nonzero(matrix == matrix.max())[0]
	return top_indexes[random.randint(0, top_indexes.shape[0]-1)]


def arg_min_2d(matrix):
	# A is assumed to be a 2D array
	m = matrix.min()
	r, c = np.nonzero(matrix == m)
	index = random.randint(0, r.shape[0]-1)
	return r[index], c[index]


def intersect(a, b):
	""" return the intersection of two lists """
	return list(set(a) & set(b))


def union(a, b):
	""" return the union of two lists """
	return list(set(a) | set(b))


def unique(a):
	""" return the list with duplicate elements removed """
	return list(set(a))


class SAVAGE():

	def __init__(self, horizon, pref_matrix, regret_func):
		
		self.nArms = len(pref_matrix)  # Number of arms
		self.iArms = range(self.nArms)   # The indices of the arms
		self.numPlays = 2*np.ones([self.nArms, self.nArms])
		self.RealWins = np.ones([self.nArms, self.nArms])
		self.horizon = horizon
		self.delta = 1. / self.horizon
		self.t = 1
		self.PMat = self.RealWins / self.numPlays
		self.C = self.c(self.numPlays)
		self.C_plus_PMat = self.c(self.numPlays) + self.RealWins/self.numPlays
		self.firstPlace = 0
		self.secondPlace = 0
		self.activePairs = np.triu(np.ones([self.nArms, self.nArms]), 1)
		self.exploit = False
		self.champ = []
		self.chatty = False
		self.bestArm = 0 #init
		self.regret_func = regret_func

		self.pref_matrix = pref_matrix

	def c(self, N):
		return np.sqrt(math.log(0.0+self.nArms*(self.nArms-1)*self.horizon**2, 2)/(2*N))

	def indep_test(self):
		uI = list(np.nonzero(self.activePairs.any(axis=1))[0])
		uJ = list(np.nonzero(self.activePairs.any(axis=0))[0])
		indepArms = list(np.nonzero((self.C_plus_PMat > 0.5).sum(axis=1) < self.nArms)[0])
		newIndepArms = unique(intersect(indepArms, union(uI, uJ)))

		for i in newIndepArms:
			self.activePairs[i, :] = 0
			self.activePairs[:, i] = 0

	def stop_explore(self):
		I, J = np.nonzero(self.activePairs)
		if sum(I.shape) == 0:
			return True
		U_cop = (self.PMat > 0.5).sum(axis=1)
		self.bestArm = arg_max(U_cop)
		if U_cop[self.bestArm] == self.nArms-1 and (self.PMat[self.bestArm, :]-self.C[self.bestArm, :] > 0.5).sum() == self.nArms-1:
			return True

	def select_arms(self):
		# This returns two arms to compare.
		if self.exploit:
			if not self.champ:
				PMat = self.RealWins / self.numPlays
				self.champ = arg_max((PMat > 0.5).sum(axis=1))
			self.firstPlace = self.champ
			self.secondPlace = self.champ
		else:
			self.firstPlace, self.secondPlace = arg_min_2d(
				self.numPlays * self.activePairs + (self.activePairs == 0)*(self.numPlays.max()+1))

		return self.firstPlace, self.secondPlace

	def update_scores(self, winner, loser, score=1):

		# first = winner
		# second = loser
		if self.exploit:
			self.t += 1
			return
		self.RealWins[winner, loser] += score
		self.numPlays[winner, loser] += 1
		self.numPlays[loser, winner] += 1
		self.indep_test()
		if self.stop_explore():
			self.exploit = True
			self.numPlays = self.RealWins+self.RealWins.T
			PMat = self.RealWins / self.numPlays
			self.champ = arg_max((PMat > 0.5).sum(axis=1))
		self.t += 1
		self.C[winner, loser] = self.c(self.numPlays[winner, loser])
		self.PMat[winner, loser] = self.RealWins[winner, loser]/self.numPlays[winner, loser]
		self.C_plus_PMat[winner, loser] = self.C[winner, loser] + self.PMat[winner, loser]
		self.C[loser, winner] = self.c(self.numPlays[loser, winner])

		self.PMat[loser, winner] = self.RealWins[loser, winner] / self.numPlays[loser, winner]
		self.C_plus_PMat[loser, winner] = self.C[loser, winner] + self.PMat[loser, winner]
		return


	def run(self):

		horizon = self.horizon
		pref_matrix = self.pref_matrix
		# Initializing the regret vector.
		regret = np.zeros(horizon)

		# Initializing the cumulative regret vector.
		cumulative_regret = np.zeros(horizon)

		for t in range(horizon):

			# Selecting the arms.
			[chosen_left_arm, chosen_right_arm] = self.select_arms()

			# Acquiring the rewards
			bt = np.random.binomial(1, self.pref_matrix[chosen_left_arm][chosen_right_arm], 1)

			# Choosing the better arm.
			if bt == 1:
				# Left arm won
				self.update_scores(chosen_left_arm, chosen_right_arm)

			else:
				# Right arm won
				self.update_scores(chosen_right_arm, chosen_left_arm)

			# By definition
			regret[t] = self.regret_func(chosen_left_arm, chosen_right_arm)

			if t == 1:
				cumulative_regret[t] = regret[t]
			else:
				cumulative_regret[t] = regret[t] + cumulative_regret[t-1]

		return list(np.around(cumulative_regret,3)), self.bestArm


# def master(iterations, horizon):

# 	pref = np.array([[0,0.05,0.05,0.04,0.11,0.11],
# 				[-0.05,0,0.05,0.04,0.08,0.10],
# 				[-0.05,-0.05,0,0.04,0.01,0.06],
# 				[-0.04,-0.04,-0.04,0,0.04,0],
# 				[-0.11,-0.08,-0.01,-0.04,0,0.01],
# 				[-0.11,-0.1,-0.06,0,-0.01,0]])

# 	n_arms = len(pref[0])
# 	obj = SAVAGE(horizon, pref)

# 	# Initializing the results vector.
# 	results = np.zeros(horizon)

# 	for iteration in range(iterations):
# 		# Adding the regret.
# 		results += obj.run(horizon, pref)

# 	return results/iterations

# regret = master(2, 50000)
# plt.plot(regret)
# plt.show()