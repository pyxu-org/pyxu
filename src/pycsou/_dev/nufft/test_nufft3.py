import dask.array as da
import numpy as np

import pycsou.operator.linop.nufft as nufft

importlib.reload(nufft)
import pycsou.runtime as pycrt
import pycsou.util as pycu


def NUFFT3_array(t, f, isign) -> np.ndarray:
    return np.exp(1j * isign * f @ t.T)


rng = np.random.default_rng(0)
D, J, K = 3, 200, 50
t = rng.normal(size=(J, D))
f = rng.normal(size=(K, D))

with pycrt.Precision(pycrt.Width.DOUBLE):
    N_trans, isign = 20, -1
    A = nufft.NUFFT.type3(t, f, n_trans=N_trans, isign=isign)
    B = NUFFT3_array(t, f, isign)

    arr = rng.normal(size=(N_trans, J))
    arr = arr + 1j * rng.normal(size=arr.shape)
    # arr = da.array(arr)

    A_out_fw = pycu.view_as_complex(A.apply(pycu.view_as_real(arr)))
    B_out_fw = np.tensordot(B, arr, axes=[[1], [1]]).T

    A_out_bw = pycu.view_as_complex(A.adjoint(pycu.view_as_real(A_out_fw)))
    B_out_bw = np.tensordot(B.conj().T, B_out_fw, axes=[[1], [1]]).T

    print(np.allclose(A_out_fw, B_out_fw))
    print(np.allclose(A_out_bw, B_out_bw))
    print((np.linalg.norm(A_out_fw - B_out_fw, axis=-1) / np.linalg.norm(B_out_fw, axis=-1)).max())
    print((np.linalg.norm(A_out_bw - B_out_bw, axis=-1) / np.linalg.norm(B_out_bw, axis=-1)).max())
