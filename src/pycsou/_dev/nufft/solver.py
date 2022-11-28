import math

import numpy as np
import scipy.optimize as sopt

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    D, M, N = 3, 5001, 2000
    x = rng.normal(size=(M, D), scale=(1, 2, 500))
    z = rng.normal(size=(N, D), scale=(30, 3, 1))

    T_tot = x.ptp(axis=0)
    B_tot = z.ptp(axis=0)
    sigma = np.r_[1.5, 1.25, 2]
    itemsize = np.dtype(np.cdouble).itemsize
    fft_bytes = 100 * 2**20
    n_trans = 1
    n_trans *= 2  # seperate fw/bw FFT memory
    alpha = 10

    c = -np.ones(2 * D)
    _k, _q = np.triu_indices(n=D, k=1)
    M, Z = np.zeros((2, math.comb(D, 2), D))
    for i, (__k, __q) in enumerate(zip(_k, _q)):
        M[i, __k] = 1
        M[i, __q] = -1
    R = np.arange(D)
    _l, _m = np.kron(R, np.ones(D, dtype=int)), np.tile(R, D)
    M2 = np.zeros((D**2, 2 * D))
    for i, (__l, __m) in enumerate(zip(_l, _m)):
        M2[i, __l] = 1
        M2[i, D + __m] = -1
    A = np.block(
        [
            [-c],
            [-c],
            [M, Z],
            [-M, Z],
            [Z, M],
            [Z, -M],
            [M2],
            [-M2],
        ]
    )
    b = np.r_[
        np.log(fft_bytes / (itemsize * n_trans)) + np.log(2 * np.pi / sigma).sum(),
        np.log(T_tot).sum() + np.log(B_tot).sum(),
        np.log(alpha) + np.log(T_tot)[_k] - np.log(T_tot)[_q],
        np.log(alpha) - np.log(T_tot)[_k] + np.log(T_tot)[_q],
        np.log(alpha) + np.log(B_tot)[_k] - np.log(B_tot)[_q],
        np.log(alpha) - np.log(B_tot)[_k] + np.log(B_tot)[_q],
        np.log(alpha) + np.log(T_tot)[_l] - np.log(B_tot)[_m],
        np.log(alpha) - np.log(T_tot)[_l] + np.log(B_tot)[_m],
    ]
    lb = -np.inf * np.ones(2 * D)
    ub = np.r_[np.log(T_tot), np.log(B_tot)]

    res = sopt.linprog(
        c=c,
        A_ub=A,
        b_ub=b,
        bounds=np.stack([lb, ub], axis=1),
        method="highs",
    )
    if res.success:
        T = np.exp(res.x[:D])
        B = np.exp(res.x[D:])
        print(f"T-ratio: {T / T_tot}")
        print(f"B-ratio: {B / B_tot}")
        print(f"N_nufft: {(T_tot / T).prod() * (B_tot / B).prod()}")
    else:
        print("No solution found.")
