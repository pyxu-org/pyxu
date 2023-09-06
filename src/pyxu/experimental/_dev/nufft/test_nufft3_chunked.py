import importlib
import time

import dask.array as da
import numpy as np

import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu.util as pxu

importlib.reload(pxo)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    D, M, N = 3, 500_1, 203_0
    rnd_points = lambda _: rng.normal(scale=rng.uniform(0.25, 0.5, size=(D,)), size=(_, D))
    rnd_offset = lambda: rng.uniform(-1, 1, size=(D,))
    scale = 20
    x = np.concatenate(
        [
            rnd_points(M) + rnd_offset() * scale,
            rnd_points(M) + rnd_offset() * scale,
            rnd_points(M) + rnd_offset() * scale,
            rnd_points(M) + rnd_offset() * scale,
            rnd_points(M) + rnd_offset() * scale,
        ],
        axis=0,
    )
    z = np.concatenate(
        [
            rnd_points(N) + rnd_offset() * scale,
            rnd_points(N) + rnd_offset() * scale,
            rnd_points(N) + rnd_offset() * scale,
            rnd_points(N) + rnd_offset() * scale,
            rnd_points(N) + rnd_offset() * scale,
        ],
        axis=0,
    )
    M, N = map(len, [x, z])

    with pxrt.Precision(pxrt.Width.SINGLE):
        kwargs = dict(
            x=x,
            z=z,
            n_trans=5,
            isign=-1,
            eps=1e-3,
            real=False,
            nthreads=1,
        )
        # Chunked NUFFT ===================================
        A = pxo.NUFFT.type3(**kwargs, chunked=True, parallel=False)
        t = time.time()
        x_chunks, z_chunks = A.auto_chunk(
            max_mem=100,
            max_anisotropy=1,
        )
        t_chunk = time.time() - t
        print("t_chunk:", t_chunk)
        t = time.time()
        A.allocate(x_chunks, z_chunks, direct_eval_threshold=50**2)
        t_alloc = time.time() - t
        print("t_alloc:", t_alloc)
        # Regular NUFFT ===================================
        kwargs.update(nthreads=0, eps=0)  # to avoid memory issues
        B = pxo.NUFFT.type3(**kwargs)

        use_dask = True
        arr = rng.normal(size=(kwargs.get("n_trans") * 3, M))
        if not kwargs.get("real"):
            arr = arr + 1j * rng.normal(size=arr.shape)
        if use_dask:
            arr = da.array(arr, dtype=arr.dtype)

    with pxrt.Precision(pxrt.Width.SINGLE):
        t = time.time()
        A_out_fw = A.apply(pxu.view_as_real(arr))
        t_A = time.time() - t
        print("t_A_fw:", t_A)
        t = time.time()
        B_out_fw = B.apply(pxu.view_as_real(arr))
        t_B = time.time() - t
        print("t_B_fw:", t_B)

        t = time.time()
        A_out_bw = A.adjoint(A_out_fw)
        t_A = time.time() - t
        print("t_A_bw:", t_A)
        t = time.time()
        B_out_bw = B.adjoint(B_out_fw)
        t_B = time.time() - t
        print("t_B_bw:", t_B)

        t = time.time()
        res_fw = (np.linalg.norm(A_out_fw - B_out_fw, axis=-1) / np.linalg.norm(B_out_fw, axis=-1)).max()
        t_fw = time.time() - t
        print("res_fw, t_fw:", float(res_fw), t_fw)
        t = time.time()
        res_bw = (np.linalg.norm(A_out_bw - B_out_bw, axis=-1) / np.linalg.norm(B_out_bw, axis=-1)).max()
        t_bw = time.time() - t
        print("res_bw, t_bw:", float(res_bw), t_bw)
