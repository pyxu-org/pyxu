import dask.array as da
import numpy as np

import pycsou.operator.linop.nufft as nufft
import pycsou.runtime as pycrt
import pycsou.util as pycu


def NUFFT3_array(x, z, isign) -> np.ndarray:
    return np.exp(1j * np.sign(isign) * z @ x.T)


use_dask = False

rng = np.random.default_rng(0)
D, M, N = 3, 200, 50
x = rng.normal(size=(M, D)) + 2
z = rng.normal(size=(N, D)) - 3
if use_dask:
    x = da.array(x)
    z = da.array(z)

with pycrt.Precision(pycrt.Width.DOUBLE):
    N_trans, isign = 20, -1
    A = nufft.NUFFT.type3(x, z, n_trans=N_trans, isign=isign, eps=1e-6, center="x")
    B = NUFFT3_array(x, z, isign)

    arr = rng.normal(size=(3, N_trans, M))
    arr = arr + 1j * rng.normal(size=arr.shape)
    if use_dask:
        arr = da.array(arr)

    A_out_fw = pycu.view_as_complex(A.apply(pycu.view_as_real(arr)))
    B_out_fw = np.tensordot(arr, B, axes=[[2], [1]])

    A_out_bw = pycu.view_as_complex(A.adjoint(pycu.view_as_real(A_out_fw)))
    B_out_bw = np.tensordot(B_out_fw, B.conj().T, axes=[[2], [1]])

    res_fw = (np.linalg.norm(A_out_fw - B_out_fw, axis=-1) / np.linalg.norm(B_out_fw, axis=-1)).max()
    res_bw = (np.linalg.norm(A_out_bw - B_out_bw, axis=-1) / np.linalg.norm(B_out_bw, axis=-1)).max()
    if use_dask:
        res_fw, res_bw = pycu.compute(res_fw, res_bw)
    print(res_fw)
    print(res_bw)
