import numpy as np

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
    return y, y.shape[0]

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


load_param("data/input/parameters.txt")
y, T = load_data("data/input/data.txt")
N, P, mu, sd = load_param("data/input/parameters.txt")
print(T, N, P, mu, sd, sep="\n", end="\n\n")
