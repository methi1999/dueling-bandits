import numpy as np

from math import log, sqrt


class InterleavedFilter():

    def __init__(self, K, T, comparison_function, regret_fn):
        self.T = T
        # self.bandit = bandit
        self.delta = 1 / (T * (K**2))
        self.W = np.arange(K)
        # stores self.result of comparison between i and j
        self.result = np.zeros((K, K))
        self.bandit = comparison_function
        self.t = 0
        self.regret = 0
        self.regrets = []
        self.regret_fn = regret_fn

    def calc_P(self, i, j):
        '''
        Empirical probability that i beats j
        '''
        if(self.result[i][j] + self.result[j][i] == 0):
            return 0.5
        return self.result[i][j] / (self.result[i][j] + self.result[j][i])

    def calc_C(self, p, i, j):
        '''
        Returns true if 0.5 doesn't lie in confidence interval
        '''
        t = self.result[i][j] + self.result[j][i] + 10**-12
        c = sqrt(log(1 / self.delta) / t)
        return (0.5 < p - c or 0.5 > p + c)

    def algo(self):
        b_hat = self.W[np.random.randint(len(self.W))]
        self.W = np.setdiff1d(self.W, np.array([b_hat]))
        p_hat = {}
        while((len(self.W)) and self.t < self.T):
            for b in self.W:
                res = self.bandit(b_hat, b)
                self.result[b_hat][b] += res
                self.result[b][b_hat] += (1 - res)
                p_hat[b] = self.calc_P(b_hat, b)  # Updated p_hat
                self.regret += self.regret_fn(b_hat, b)
                self.regrets.append(float(self.regret))
                self.t += 1
            '''
            Removing arms which are worse than the randomly selected arm
            '''
            to_remove = [x for x in p_hat.items() if (
                x[1] > 0.5 and self.calc_C(x[1], b_hat, x[0]))]
            for rem in to_remove:
                p_hat.pop(rem[0])
                self.W = np.setdiff1d(self.W, np.array([rem[0]]))
            new_candidate = [x for x in p_hat.items() if (
                x[1] < 0.5 and self.calc_C(x[1], b_hat, x[0]))]
            if(len(new_candidate)):
                b_hat = new_candidate[0][0]
                self.W = np.setdiff1d(self.W, np.array([new_candidate[0]]))
                p_hat = {}  # New round
        for i in range(self.t, self.T):
            self.regret += self.regret_fn(b_hat, b_hat)
            self.regrets.append(float(self.regret))
            
        return self.regrets[:self.T], b_hat, self.t
