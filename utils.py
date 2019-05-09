import numpy as np
import json

from matplotlib import pyplot as plt

from IF1 import InterleavedFilter
from RMED import RMED
from BTM import BeatTheMean
from doubler import Doubler
from savage import SAVAGE
from rucb import run_rucb
from rcs import run_rcs

# dataset = "5_real"
# best_arm = 0

# dataset = "5_art"
# best_arm = 0

# dataset = "10_real"
# best_arm = 7

dataset = "10_art"
best_arm = 0

with open(dataset+".npy", 'rb') as f:
    pref_mat = np.load(f)    


def generator_fnc(i, j):
    return np.random.binomial(n=1, p=pref_mat[int(i)][int(j)], size=1)


def regret_fn(i, j):
    return pref_mat[best_arm][int(i)] + pref_mat[best_arm][int(j)] - 1


def generate_dataset(path):
    pref_mat_new = np.zeros((10, 10))
    pref_mat = np.zeros((10, 10))
    comparisons = np.genfromtxt(path, delimiter=',')
    for j in comparisons:
        # print(j)
        pref_mat_new[int(j[0]) - 1][int(j[1]) - 1] += 1
    for i in range(10):
        for j in range(10):
            if (pref_mat_new[i][j] + pref_mat_new[j][i] == 0):
                pref_mat[i][j] = 0.5
            else:
                pref_mat[i][j] = pref_mat_new[i][j] / \
                    (pref_mat_new[i][j] + pref_mat_new[j][i])
    np.savetxt('car_data.csv', pref_mat, delimiter=',')

# generate_dataset('prefs1.csv')

def dump_all(d):
    
    with open(dataset+".json", 'w') as f:
        json.dump(d, f)

def plot():

    try:
        d = json.load(open(dataset+".json", "r"))

        for key, val in d.items():

            if key not in ['RUCB_regrets', 'RCS_regrets']:
            
                samples, iterations = len(val), len(val[0])
                
                avg = np.zeros(iterations)
                
                for sample in val:
                    avg += np.array(sample)
                
                avg /= samples
                plt.plot(avg, label=key.split('_')[0])

            else:
                plt.plot(val, label=key.split('_')[0])

        
        plt.legend()
        # plt.savefig('all_plots.png')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    except:
        print("Cant find json")
        exit(0)

def f_rmed(k):
    return 0.3 * (k**1.01)

def gen():

    d = {"RMED1_regrets" : [],
        "RMED2_regrets" : [],
        "IF_regrets" : [],
        "BTM_regrets" : [],
        "SAVAGE_regrets" : [],
        "DOUBLER_regrets" : []
        }

    horizon = 100000
    samples = 10
    
    for i in range(samples):

        print("On sample number", i)

        x = RMED('RMED1', len(pref_mat), horizon,  generator_fnc, f_rmed, regret_fn)
        reg_RMED = np.array(x.algo())
        #reg_BTM[0] contains the regret values, [1] contains the best arm
        print("RMED done, best arm : ", reg_RMED[1])
        d["RMED1_regrets"].append(list(np.around(reg_RMED[0],3)))
        # json.dump(RMED1_regrets, open("RMED1_regrets.json", "w"))

        x = RMED('RMED2', len(pref_mat), horizon,  generator_fnc, f_rmed, regret_fn)
        reg_RMED = np.array(x.algo())
        print("RMED done, best arm : ", reg_RMED[1])
        d["RMED2_regrets"].append(list(np.around(reg_RMED[0],3)))
        # json.dump(RMED2_regrets, open("RMED2_regrets.json", "w"))

        x = InterleavedFilter(len(pref_mat), horizon,  generator_fnc,  regret_fn)
        reg_IF = np.array(x.algo())
        print("IF done, best arm : ", reg_IF[1])
        d["IF_regrets"].append(list(np.around(reg_IF[0],3)))
        # json.dump(IF_regrets, open("IF_regrets.json", "w"))

        x = BeatTheMean(len(pref_mat), horizon,  generator_fnc,  regret_fn)
        reg_BTM = np.array(x.algo())
        print("BTM done, best arm : ", reg_BTM[1])
        d["BTM_regrets"].append(list(np.around(reg_BTM[0],3)))
        # json.dump(BTM_regrets, open("BTM_regrets.json", "w"))

        x = Doubler(horizon, pref_mat, regret_fn)
        reg_DOUBLER = x.run()
        print("Doubler done, best arm : ", reg_DOUBLER[1])
        d["DOUBLER_regrets"].append(reg_DOUBLER[0])
        # json.dump(DOUBLER_regrets, open("DOUBLER_regrets.json", "w"))

        x = SAVAGE(horizon, pref_mat, regret_fn)
        reg_SAVAGE = x.run()
        print("SAVAGE done, best arm : ", reg_SAVAGE[1])
        d["SAVAGE_regrets"].append(reg_SAVAGE[0])
        # json.dump(SAVAGE_regrets, open("SAVAGE_regrets.json", "w"))

        if i%3== 0:
            print("Dumped all")
            dump_all(d)

    dump_all(d)

    d['RUCB_regrets'] = run_rucb(samples, horizon, pref_mat)
    print("RUCB done")
    d['RCS_regrets'] = run_rcs(samples, horizon, pref_mat)
    print("RCS done")
    dump_all(d)

gen()
# plot()
