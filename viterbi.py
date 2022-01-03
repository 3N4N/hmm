import numpy as np
from scipy.stats import norm

def load_data(fn_data):
    fdata = open(fn_data)
    y = None
    for line in fdata:
        data = float(line.strip())
        if y is None:
            y = np.array(data)
        else:
            y = np.append(y, data)
    fdata.close()
    return y

def load_param(fn_param):
    fparam = open(fn_param)

    n_states = int(fparam.readline().strip())

    P = None
    for i in range(n_states):
        p_ij = list(map(float, fparam.readline().strip().split('\t')))
        if P is None:
            P = np.array([p_ij])
        else:
            P = np.concatenate((P, [p_ij]))

    mu = list(map(float, fparam.readline().strip().split('\t')))
    sd = list(map(float, fparam.readline().strip().split('\t')))

    fparam.close()
    return n_states, P, mu, sd


def get_stationary_distribution(P, n):
    y = P
    for _ in range(n):
        y = np.matmul(y, P)
    return y


def viterbi(y, P, E, Pi):
    T = len(y)




y = load_data("data/input/data.txt")
N, P, mu, sd = load_param("data/input/parameters.txt")
print(N, P, mu, sd, sep="\n", end="\n\n")

Pi = get_stationary_distribution(P, 50)
print(Pi)
Pi = Pi[0]
print(Pi)

E = None
for i in range(N):
    if E is None:
        E = [np.array(norm(mu[i],sd[i]).pdf(y))]
    else:
        E = np.concatenate((E, [np.array(norm(mu[i],sd[i]).pdf(y))]))
print(E.shape)
