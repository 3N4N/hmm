import numpy as np


def load_param(fn_param):
    fparam = open(fn_param)

    n_states = int(fparam.readline().strip())
    print(n_states)

    P = None
    for i in range(n_states):
        pij = list(map(float, fparam.readline().strip().split('\t')))
        if P is None:
            P = np.array([pij])
        else:
            P = np.concatenate((P, [pij]))
    print(P)

    mu = list(map(float, fparam.readline().strip().split('\t')))
    sd = list(map(float, fparam.readline().strip().split('\t')))
    print(mu, sd)

    fparam.close()


load_param("data/input/parameters.txt")
