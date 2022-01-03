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

    Tm = None
    for i in range(n_states):
        p_ij = list(map(float, fparam.readline().strip().split('\t')))
        if Tm is None:
            Tm = np.array([p_ij])
        else:
            Tm = np.concatenate((Tm, [p_ij]))

    mu = np.array(list(map(float, fparam.readline().strip().split('\t'))))
    sd = np.sqrt(np.array(list(map(float, fparam.readline().strip().split('\t')))))

    fparam.close()
    return n_states, Tm, mu, sd


def get_stationary_distribution(Tm, n):
    y = Tm
    for _ in range(n):
        y = np.matmul(y, Tm)
    return y


def viterbi(y, Tm, Em, Pi):
    N = len(y)
    S = Tm.shape[0]

    trellis = np.zeros((N, S))
    trellis[0, :] = np.log(Pi, Em[:, 0])
    pointers = np.zeros((N-1, S)).astype(np.int32)

    for n in range(1, N):
        for s in range(S):
            probability = trellis[n - 1] + np.log(Tm[:, s]) + np.log(Em[s, n])
            pointers[n-1, s] = np.argmax(probability)
            trellis[n, s] = np.max(probability)

    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(trellis[N-1, :])

    for n in range(N-2, -1, -1):
        S_opt[n] = pointers[n, int(S_opt[n+1])]

    return S_opt




y = load_data("data/input/data.txt")
S, Tm, mu, sd = load_param("data/input/parameters.txt")

Pi = get_stationary_distribution(Tm, 50)
Pi = Pi[0]

Em = None
for s in range(S):
    probs = np.array(norm(mu[s],sd[s]).pdf(y))
    if Em is None:
        Em = [probs]
    else:
        Em = np.concatenate((Em, [probs]))
# np.savetxt("Em.txt", Em)


S_opt = viterbi(y, Tm, Em, Pi).astype(str)
S_opt[S_opt == '0'] = "El Nino"
S_opt[S_opt == '1'] = "La Nina"
np.savetxt("data/output/states.txt", S_opt, fmt="\"%s\"")
