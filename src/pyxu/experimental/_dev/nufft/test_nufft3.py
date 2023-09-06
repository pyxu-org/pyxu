import dask.array as da
import distributed  # noqa: F401
import numpy as np

import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu.util as pxu

# client = distributed.Client(processes=False)
use_dask = True


def NUFFT3_array(x, z, isign) -> np.ndarray:
    return np.exp(1j * np.sign(isign) * z @ x.T)


rng = np.random.default_rng(0)
D, M, N = 3, 200, 50
x = rng.normal(size=(M, D))
z = rng.normal(size=(N, D))

with pxrt.Precision(pxrt.Width.DOUBLE):
    N_trans, isign = 10, -1
    A = pxo.NUFFT.type3(
        x=x,
        z=z,
        n_trans=N_trans,
        isign=isign,
        eps=1e-6,
        real=False,
        debug=2,
    )
    B = NUFFT3_array(x, z, isign)

    arr = rng.normal(size=(2, 3, 4, M))
    arr = arr + 1j * rng.normal(size=arr.shape)
    if use_dask:
        arr = da.array(arr)

    A_out_fw = pxu.view_as_complex(A.apply(pxu.view_as_real(arr)))
    B_out_fw = np.tensordot(arr.compute() if use_dask else arr, B, axes=[[-1], [-1]])

    A_out_bw = pxu.view_as_complex(A.adjoint(pxu.view_as_real(A_out_fw)))
    B_out_bw = np.tensordot(B_out_fw, B.conj().T, axes=[[-1], [-1]])

    res_fw = (np.linalg.norm(A_out_fw - B_out_fw, axis=-1) / np.linalg.norm(B_out_fw, axis=-1)).max()
    res_bw = (np.linalg.norm(A_out_bw - B_out_bw, axis=-1) / np.linalg.norm(B_out_bw, axis=-1)).max()
    print(float(res_fw))
    print(float(res_bw))
