import itertools
import numpy as np
from math import log, sqrt


class RMED():

    def __init__(self, mode, K, T, comparison_function, f, regret_fn, alpha=1):
        self.T = T
        self.mode = mode
        if mode in ['RMED1', 'RMED2']:
            self.L = 1
        else:
            self.L = alpha * log(log(T))
        self.result_mat = np.zeros((K, K))
        self.bandit = comparison_function
        self.K = K
        self.L_c = np.arange(K)
        self.L_r = np.arange(K)
        self.L_n = np.arange(0)
        self.t = 0
        self.I_t = np.zeros((K, 1))
        self.f = f
        self.regret = 0
        self.regrets = []
        self.regret_fn = regret_fn
        self.alpha = alpha
        self.b_hat_star = np.zeros((self.K, 1))

    def get_mean(self, i, j):
        if self.result_mat[int(i)][int(j)] + self.result_mat[int(j)][int(i)] == 0:
            return 0.5
        return self.result_mat[int(i)][int(j)] / (self.result_mat[int(i)][int(j)] + self.result_mat[int(j)][int(i)])

    def KL_divergence(self, p, q):
        if p == 0:
            return 0
        return p * log(p / q) + (1 - p) * (log((1 - p) / (1 - q)))

    def KL_plus(self, p, q):
        if p < q:
            return self.KL_divergence(p, q)
        return 0

    def update_It(self):
        for i in range(self.K):
            O_i = [x for x in np.arange(self.K) if (
                x != i and self.get_mean(i, x) <= 0.5)]
            res = 0
            for j in O_i:
                res += (self.result_mat[i][j] + self.result_mat[j][i]) * \
                    (self.KL_divergence(self.get_mean(i, j), 0.5))
            self.I_t[i] = res

    def update_b_hat_star(self, i):
        i_star = np.argmin(self.I_t)
        p = dict([(x, (self.get_mean(i_star, x) +
                       self.get_mean(i_star, i)) / (self.KL_plus(self.get_mean(i, x), 0.5))) for x in np.arange(self.K) if x != i])
        self.b_hat_star[i] = min(p, key=p.get)

    def rmed1(self, l_t):
        O_lt = [x for x in np.arange(self.K) if (
            x != l_t and self.get_mean(l_t, x) <= 0.5)]
        if np.argmin(self.I_t) in O_lt or len(O_lt) == 0:
            return np.argmin(self.I_t)
        else:
            mu = dict([(int(x), self.get_mean(l_t, x))
                       for x in np.arange(self.K) if x != l_t])
            return min(mu, key=mu.get)

    def rmed2(self, l_t):
        # print(l_t)
        l_t = int(l_t)
        self.update_b_hat_star(l_t)
        O_lt = [x for x in np.arange(self.K) if (
            x != l_t and self.get_mean(l_t, x) <= 0.5)]
        if self.b_hat_star[l_t] in O_lt:
            return self.b_hat_star[l_t]
        if np.argmin(self.I_t) in O_lt or len(O_lt) == 0:
            return np.argmin(self.I_t)
        else:
            mu = dict([(int(x), self.get_mean(l_t, x))
                       for x in np.arange(self.K) if x != l_t])
            return min(mu, key=mu.get)

    def algo(self):
        for i in range(self.L):
            for j in itertools.combinations(np.arange(self.K), 2):
                # print(self.K)
                # print(j)
                res = self.bandit(j[0], j[1])
                self.result_mat[j[0]][j[1]] += res
                self.result_mat[j[1]][j[0]] += (1 - res)
                self.regret += self.regret_fn(j[0], j[1])
                self.regrets.append(float(self.regret))
        self.t = self.L * self.K * (self.K - 1) / 2
        self.update_It()
        '''
        Insert rmed2fh here
        '''
        while self.t < self.T:
            '''
            Insert rmed2 here
            '''
            if self.mode == "RMED2":
                for j in itertools.combinations(np.arange(self.K), 2):
                    # print(self.K)
                    # print(j)
                    while self.result_mat[j[0]][j[1]] + self.result_mat[j[1]][j[0]] < self.alpha * log(log(self.t)):
                        res = self.bandit(j[0], j[1])
                        self.result_mat[j[0]][j[1]] += res
                        self.result_mat[j[1]][j[0]] += (1 - res)
                        self.regret += self.regret_fn(j[0], j[1])
                        self.regrets.append(float(self.regret))
                        self.t += 1

            for l_t in self.L_c:
                self.update_It()
                if self.mode == 'RMED1':
                    m_t = self.rmed1(l_t)
                else:
                    m_t = self.rmed2(l_t)
                res = self.bandit(l_t, m_t)
                self.result_mat[int(l_t)][int(m_t)] += res
                self.result_mat[int(m_t)][int(l_t)] += (1 - res)
                self.regret += self.regret_fn(l_t, m_t)
                self.regrets.append(float(self.regret))
                self.L_r = np.setdiff1d(self.L_r, np.array([l_t]))
                self.L_n = np.union1d(self.L_n, np.array(
                    [x for x in np.arange(self.K) if (x not in self.L_r and self.J(x))]))
                self.t += 1
            self.L_c = self.L_n
            self.L_r = self.L_n
            self.L_n = np.arange(0)
        return self.regrets, np.argmin(self.I_t), self.t

    def J(self, x):
        return ((self.I_t[x] - min(self.I_t)) <= (log(self.t) + self.f(self.K)))
