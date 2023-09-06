import dask.array as da
import distributed  # noqa: F401
import numpy as np

import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu.util as pxu

# client = distributed.Client(processes=False)
use_dask = True


def NUFFT1_array(x, N, isign) -> np.ndarray:
    D = x.shape[-1]
    if isinstance(N, int):
        N = (N,) * D
    A = np.meshgrid(
        *[np.arange(-(n // 2), (n - 1) // 2 + 1) for n in N],
        indexing="ij",
    )
    B = np.stack(A, axis=0).reshape((D, -1)).T
    return np.exp(1j * np.sign(isign) * B @ x.T)


rng = np.random.default_rng(0)
D, M, N = 3, 200, (5, 3, 4)
x = rng.normal(size=(M, D))

with pxrt.Precision(pxrt.Width.DOUBLE):
    N_trans, isign, real = 10, -1, True
    A = pxo.NUFFT.type1(
        x=x,
        N=N,
        n_trans=N_trans,
        isign=isign,
        eps=1e-6,
        real=real,
        debug=2,
    )
    B = NUFFT1_array(x, N, isign)

    arr = rng.normal(size=(2, 3, 4, M))
    if not real:
        arr = arr + 1j * rng.normal(size=arr.shape)
    if use_dask:
        arr = da.array(arr)

    A_out_fw = pxu.view_as_complex(A.apply(pxu.view_as_real(arr)))
    B_out_fw = np.tensordot(arr.compute() if use_dask else arr, B, axes=[[-1], [-1]])

    A_out_bw = A.adjoint(pxu.view_as_real(A_out_fw))
    if not real:
        A_out_bw = pxu.view_as_complex(A_out_bw)
    B_out_bw = np.tensordot(B_out_fw, B.conj().T, axes=[[-1], [-1]])
    if real:
        B_out_bw = B_out_bw.real

    res_fw = (np.linalg.norm(A_out_fw - B_out_fw, axis=-1) / np.linalg.norm(B_out_fw, axis=-1)).max()
    res_bw = (np.linalg.norm(A_out_bw - B_out_bw, axis=-1) / np.linalg.norm(B_out_bw, axis=-1)).max()
    print(float(res_fw))
    print(float(res_bw))
