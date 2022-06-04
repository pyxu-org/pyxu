import dask.array as da
import numpy as np

import pycsou.operator.linop.nufft as nufft

importlib.reload(nufft)
import pycsou.runtime as pycrt
import pycsou.util as pycu


def NUFFT1_array(t, M, isign) -> np.ndarray:
    D = t.shape[-1]
    if isinstance(M, int):
        M = (M,) * D
    A = np.meshgrid(*[np.arange(-(m // 2), (m - 1) // 2 + 1) for m in M], indexing="ij")
    B = np.stack(A, axis=0).reshape((D, -1)).T
    return np.exp(1j * isign * B @ t.T)


use_dask = False

rng = np.random.default_rng(0)
D, J, M = 2, 200, 5
t = rng.normal(size=(J, D)) - 50
if use_dask:
    t = da.array(t)

with pycrt.Precision(pycrt.Width.DOUBLE):
    N_trans, isign = 5, -1
    A = nufft.NUFFT.rtype1(t, M, n_trans=N_trans, isign=isign)
    B = NUFFT1_array(t, M, isign)

    arr = rng.normal(size=(N_trans, J))
    if use_dask:
        arr = da.array(arr)

    A_out_fw = pycu.view_as_complex(A.apply(arr))
    B_out_fw = np.tensordot(B, arr, axes=[[1], [1]]).T

    A_out_bw = A.adjoint(pycu.view_as_real(A_out_fw))
    B_out_bw = np.tensordot(B.conj().T, B_out_fw, axes=[[1], [1]]).T

    res_fw = (np.linalg.norm(A_out_fw - B_out_fw, axis=-1) / np.linalg.norm(B_out_fw, axis=-1)).max()
    res_bw = (np.linalg.norm(A_out_bw - B_out_bw.real, axis=-1) / np.linalg.norm(B_out_bw.real, axis=-1)).max()
    if use_dask:
        res_fw, res_bw = pycu.compute(res_fw, res_bw)
    print(res_fw)
    print(res_bw)
