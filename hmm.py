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


def gaussian_distribution(x, mu, sd):
    p = norm.pdf(x, loc=mean, scale=std)
    if p == 0.0:
        p = 1e-323
    return p

def get_emission_matrix_from_gaussian(mu, sd, N, y):
    _B = None
    for s in range(N):
        probs = np.array(norm(mu[s],sd[s]).pdf(y))
        if _B is None:
            _B = [probs]
        else:
            _B = np.concatenate((_B, [probs]))
    return _B

def get_stationary_distribution(Tm, n=50):
    y = Tm
    for _ in range(n):
        y = np.matmul(y, Tm)
    return y


def viterbi(y, Tm, Em, Pi):
    N = len(y)
    S = Tm.shape[0]

    trellis = np.zeros((N, S))
    trellis[0, :] = np.log(Pi * Em[:, 0])
    pointers = np.zeros((N-1, S)).astype(np.int32)

    for n in range(1, N):
        for s in range(S):
            probability = trellis[n - 1] + np.log(Tm[:, s]) + np.log(Em[s, n])
            pointers[n-1, s] = np.argmax(probability)
            trellis[n, s] = np.max(probability)

    np.savetxt("dp.txt", trellis.T)
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(trellis[N-1, :])

    for n in range(N-2, -1, -1):
        S_opt[n] = pointers[n, int(S_opt[n+1])]

    return S_opt





def forward(y, A, B, Pi):
    M = A.shape[0]
    T = len(y)

    c_t = np.array([0.0] * T)
    alpha = np.zeros((T, M))
    alpha[0,:] = Pi * B[:, 0]
    c_t[0] = 1.0 / np.sum(alpha[0,:])
    alpha[0,:] *= c_t[0]

    for t in range(1, T):
        for i in range(M):
            alpha[t, i] = (alpha[t - 1] @ A[:,i]) * B[i, t]
        c_t[t] = 1.0 / np.sum(alpha[t,:])
        for i in range(M):
            alpha[t, i] *= c_t[t]

    return alpha, c_t

def backward(y, A, B, c_t):
    M = A.shape[0]
    T = len(y)

    beta = np.zeros((T, M))
    beta[T - 1] = np.ones((M))

    for t in range(T - 2, -1, -1):
        for i in range(M):
            beta[t, i] = (beta[t+1] * B[:, t+1]) @ A[i, :]
            beta[t,i] *=c_t[t]

    return beta


def baum_welch(y, A, mu, sd, Pi, n_iter):
    M = A.shape[0]
    T = len(y)

    B = get_emission_matrix_from_gaussian(mu,sd, M, y)
    np.savetxt("em/myem_"+".txt", B.T)

    for n in range(n_iter):
        Pi = get_stationary_distribution(A)[0]
        alpha, c_t = forward(y, A, B, Pi)
        beta = backward(y, A, B, c_t)

        theta = np.zeros((M, M, T-1))
        for t in range(T - 1):
            denominator = ((alpha[t, :].T @ A) * B[:, t + 1].T) @ beta[t + 1, :]
            for i in range(M):
                numerator = alpha[t, i] * A[i, :] * B[:, t + 1].T * beta[t + 1, :].T
                theta[i, :, t] = numerator / denominator

        gamma = np.sum(theta, axis=1)
        _c_t = np.tile(c_t[:-1],(2,1))
        _g = gamma
        A = np.sum(theta, axis=2) / np.sum(_g, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(theta[:, :, T - 2], axis=0).reshape((-1, 1))))

        mu = (gamma @ y) / np.sum(gamma, axis=1)

        jab = alpha.T * beta.T
        for col in range(T):
            jab[:, col] /= jab[:, col].sum()
        for i in range(M):
            diff = y - mu[i]
            temp = np.sum(jab[i] * diff * diff) / np.sum(jab[i])
            sd[i] = np.sqrt(temp)
        np.savetxt("em/mysd"+str(n)+".txt", sd)

        B = get_emission_matrix_from_gaussian(mu,sd, M, y)
        np.savetxt("em/myem"+str(n)+".txt", B.T)

    return A, mu, sd, Pi


def save_veterbi_results(y, Tm, Em, Pi, fn):
    S_opt = viterbi(y, Tm, Em, Pi).astype(str)
    S_opt[S_opt == '0'] = "El Nino"
    S_opt[S_opt == '1'] = "La Nina"
    np.savetxt(fn, S_opt, fmt="\"%s\"")




y = load_data("data/input/data.txt")
S, Tm, mu, sd = load_param("data/input/parameters.txt")

Pi = get_stationary_distribution(Tm, 50)
Pi = Pi[0]

Em = get_emission_matrix_from_gaussian(mu,sd, S, y)

save_veterbi_results(y, Tm, Em, Pi, "data/output/states_wo_learning.txt")


A, mu, sd, Pi = baum_welch(y, Tm, mu, sd, Pi, 20)
Em = get_emission_matrix_from_gaussian(mu,sd, S, y)

save_veterbi_results(y, Tm, Em, Pi, "data/output/states_after_learning.txt")

# print(A, mu, np.square(sd), Pi, sep="\n")

fnparam = 'data/output/parameters.txt'
fparam = open(fnparam, "w")

fparam.write(str(S) + '\n')

for i in range(S):
    for j in range(S):
        fparam.write(str(round(A[i][j], 7)) + "\t")
    fparam.write("\n")

for i in range(S):
    fparam.write(str(round(mu[i], 4)) + "\t")
fparam.write("\n")

for i in range(S):
    fparam.write(str(round(np.square(sd)[i], 7)) + "\t")
fparam.write("\n")

for i in range(S):
    fparam.write(str(round(Pi[i], 7)) + "\t")
fparam.write("\n")

fparam.close()
