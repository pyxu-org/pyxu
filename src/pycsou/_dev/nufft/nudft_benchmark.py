# Goal: compare performance of different Numba kernels to evaluate NUFFT FW-model
# Experiments portray the type-3 case, but results transfer to type-1/2 case.

import timeit

import numba
import numpy as np

import pycsou.util.ptype as pyct


@numba.njit(parallel=True, fastmath=True, nogil=True)
def _nudft(
    weights: pyct.NDArray,  # (..., M) [float32/64, complex64/128] (2d or nd, depending on `version`)
    source: pyct.NDArray,  # (M, D) sample points [float32/64]
    target: pyct.NDArray,  # (N, D) query points [float32/64]
    *,
    isign: int,
    dtype: pyct.DType,  # complex64/128,
    version: int,
) -> pyct.NDArray:  # complex64/128
    if version == 1:  # nd weights -> nd output, prange x2
        M = source.shape[0]
        N = target.shape[0]
        out = np.zeros(
            shape=weights.shape[:-1] + (N,),
            dtype=dtype,
        )
        for n in numba.prange(N):
            for m in numba.prange(M):
                scale = np.exp(isign * 1j * np.dot(source[m], target[n]))
                out[..., n] += weights[..., m] * scale
    elif version == 2:  # nd weights -> nd output, prange x1
        M = source.shape[0]
        N = target.shape[0]
        out = np.zeros(
            shape=weights.shape[:-1] + (N,),
            dtype=dtype,
        )
        for n in numba.prange(N):
            for m in range(M):
                scale = np.exp(isign * 1j * np.dot(source[m], target[n]))
                out[..., n] += weights[..., m] * scale
    elif version == 3:  # 2d weights -> 2d output, loop assignment
        M = source.shape[0]
        N = target.shape[0]
        Q = weights.shape[0]
        out = np.zeros((Q, N), dtype=dtype)
        for n in numba.prange(N):
            for m in numba.prange(M):
                scale = np.exp(isign * 1j * np.dot(source[m], target[n]))
                for q in range(Q):
                    out[q, n] += weights[q, m] * scale
    elif version == 4:  # 2d weights -> 2d output, vector assignment
        M = source.shape[0]
        N = target.shape[0]
        Q = weights.shape[0]
        out = np.zeros(shape=(Q, N), dtype=dtype)
        for n in numba.prange(N):
            for m in range(M):
                scale = np.exp(isign * 1j * np.dot(source[m], target[n]))
                out[:, n] += weights[:, m] * scale
    elif version == 5:  # 2d weights -> 2d output, vector assignment + permutation
        M = source.shape[0]
        N = target.shape[0]
        Q = weights.shape[0]
        out = np.zeros(shape=(N, Q), dtype=dtype)
        for n in numba.prange(N):
            for m in range(M):
                scale = np.exp(isign * 1j * np.dot(source[m], target[n]))
                out[n, :] += weights[:, m] * scale
    elif version == 6:  # 2d weights -> 2d output, loop assignment + permutation
        M = source.shape[0]
        N = target.shape[0]
        Q = weights.shape[0]
        out = np.zeros(shape=(N, Q), dtype=dtype)
        for n in numba.prange(N):
            for m in numba.prange(M):
                scale = np.exp(isign * 1j * np.dot(source[m], target[n]))
                for q in range(Q):
                    out[n, q] += weights[q, m] * scale
    return out


rng = np.random.default_rng(0)
M, N, D = 500, 2000, 3
dtype = np.dtype(np.cdouble)
isign = 1

x = rng.normal(size=(M, D))
z = rng.normal(size=(N, D))
w = rng.normal(size=(5 * 3 * 4, M))
w = w + 1j * rng.normal(size=w.shape)
w = w.astype(dtype)


def run(version: int, N_iter: int):
    f = lambda: _nudft(
        weights=w,
        source=x,
        target=z,
        isign=isign,
        dtype=dtype,
        version=version,
    )
    timer = timeit.Timer(stmt=f, setup=f)
    t_wall = timer.timeit(number=N_iter) / N_iter
    print(f"Version {version}: {t_wall} [s/call]")


N_iter = 10
run(1, N_iter)
run(2, N_iter)
run(3, N_iter)
run(4, N_iter)
run(5, N_iter)
run(6, N_iter)
