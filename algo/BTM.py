import numpy as np

from math import log, sqrt


class BeatTheMean():

    def __init__(self, K, T, comparison_function, regret_fn):
        self.T = T
        # self.bandit = bandit
        self.delta = 1 / (T * (K**2))
        self.W = np.arange(K)
        self.wins = np.zeros((K, K))
        self.plays = np.zeros((K, K))
        self.bandit = comparison_function
        self.t = 0
        self.regret = 0
        self.regrets = []
        self.regret_fn = regret_fn
        self.K = K

    def algo_base(self, N, T, c_del_gamma):
        n_star = np.min(np.sum(self.plays, axis=1))

        while(len(self.W) > 1 and self.t < T and n_star < N):
            n_star = np.min(np.sum(self.plays, axis=1))
            if n_star == 0:
                c_star = 1
            else:
                c_star = c_del_gamma(n_star)

            b = np.argmin(np.sum(self.plays, axis=1))
            b_dash = np.setdiff1d(self.W, np.array([b]))[
                np.random.randint(len(self.W) - 1)]
            res = self.bandit(b, b_dash)
            self.wins[b][b_dash] += res
            self.plays[b][b_dash] += 1
            self.t += 1
            self.regret += self.regret_fn(b, b_dash)
            self.regrets.append(float(self.regret))
            w = np.sum(self.wins, axis=1)
            b = np.sum(self.plays, axis=1)
            p_hat = dict([x for x in (zip(np.arange(len(w)), np.divide(
                w, b, out=np.array([0.5] * len(w)), where=b != 0))) if x[0] in self.W])
            if np.min(np.array(list(p_hat.values()))) + c_star < np.max(
                    np.array(list(p_hat.values()))) - c_star:
                # print("Hey")
                b_dash = min(p_hat, key=p_hat.get)
                self.wins[:, b_dash] = 0
                self.plays[:, b_dash] = 0
                self.W = np.setdiff1d(self.W, np.array(b_dash))
        best_arm = max(p_hat, key=p_hat.get)
        return (self.regrets, best_arm, self.W)

    def algo(self, gamma=1.2):
        delta = 1 / (2 * self.T * self.K)

        def confidence_bounds(n):
            return 3 * gamma * gamma * sqrt(log(1 / delta) / n)
            
        return self.algo_base(10**15, self.T, confidence_bounds)
