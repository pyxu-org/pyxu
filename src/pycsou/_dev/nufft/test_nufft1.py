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


rng = np.random.default_rng(0)
D, J, M = 2, 200, 5
t = rng.normal(size=(J, D)) - 50

with pycrt.Precision(pycrt.Width.DOUBLE):
    N_trans, isign = 5, -1
    A = nufft.NUFFT.type1(t, M, n_trans=N_trans, isign=isign)
    B = NUFFT1_array(t, M, isign)

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
