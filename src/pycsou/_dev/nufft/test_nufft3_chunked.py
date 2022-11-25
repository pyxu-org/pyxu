import pdb
import time

import dask.array as da
import distributed
import numpy as np

import pycsou.operator.linop.nufft as nufft
import pycsou.runtime as pycrt
import pycsou.util as pycu

if __name__ == "__main__":
    client = distributed.Client(processes=False)
    use_dask = True

    rng = np.random.default_rng(0)
    D, M, N = 3, 5000, 2000
    x = np.sort(rng.normal(size=(M, D), scale=3))
    z = np.sort(rng.normal(size=(N, D), scale=5))

    with pycrt.Precision(pycrt.Width.SINGLE):
        kwargs = dict(
            x=x,
            z=z,
            n_trans=10,
            isign=-1,
            eps=1e-3,
            real=False,
            nthreads=1,
        )
        A = nufft.NUFFT.type3(**kwargs, chunked=True)
        # Manual chunking (for now) ------------------------------------
        x_chunks = np.array_split(np.arange(len(kwargs.get("x"))), 5)
        z_chunks = np.array_split(np.arange(len(kwargs.get("z"))), 6)
        A.allocate(x_chunks, z_chunks)
        # --------------------------------------------------------------
        B = nufft.NUFFT.type3(**kwargs)

        arr = rng.normal(size=(kwargs.get("n_trans") * 3, M))
        if not kwargs.get("real"):
            arr = arr + 1j * rng.normal(size=arr.shape)
        if use_dask:
            arr = da.array(arr)

        t = time.time()
        A_out_fw = A.apply(pycu.view_as_real(arr))
        t_A = time.time() - t
        t = time.time()
        B_out_fw = B.apply(pycu.view_as_real(arr))
        t_B = time.time() - t
        print(t_A)
        print(t_B)

        t = time.time()
        A_out_bw = A.adjoint(A_out_fw)
        t_A = time.time() - t
        t = time.time()
        B_out_bw = B.adjoint(B_out_fw)
        t_B = time.time() - t
        print(t_A)
        print(t_B)

        res_fw = (np.linalg.norm(A_out_fw - B_out_fw, axis=-1) / np.linalg.norm(B_out_fw, axis=-1)).max()
        res_bw = (np.linalg.norm(A_out_bw - B_out_bw, axis=-1) / np.linalg.norm(B_out_bw, axis=-1)).max()
        if use_dask:
            res_fw, res_bw = pycu.compute(res_fw, res_bw)
        print()
        print(res_fw)
        print(res_bw)
