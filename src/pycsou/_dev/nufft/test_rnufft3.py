import dask.array as da
import numpy as np

import pycsou.operator.linop.nufft as nufft

importlib.reload(nufft)
import pycsou.runtime as pycrt
import pycsou.util as pycu


def NUFFT3_array(t, f, isign) -> np.ndarray:
    return np.exp(1j * isign * f @ t.T)


use_dask = False

rng = np.random.default_rng(0)
D, J, K = 3, 200, 50
t = rng.normal(size=(J, D))
f = rng.normal(size=(K, D))
if use_dask:
    t = da.array(t)
    f = da.array(f)

with pycrt.Precision(pycrt.Width.DOUBLE):
    N_trans, isign = 20, -1
    A = nufft.NUFFT.rtype3(t, f, n_trans=N_trans, isign=isign)
    B = NUFFT3_array(t, f, isign)

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
