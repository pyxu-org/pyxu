import importlib
import time

import numpy as np

import pycsou.operator.linop.nufft as nufft
import pycsou.runtime as pycrt
import pycsou.util as pycu

importlib.reload(nufft)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    D, M, N = 2, 501, 2030
    rnd_points = lambda: rng.normal(scale=rng.uniform(0.25, 0.5, size=(D,)), size=(M, D))
    radius = 100
    x = np.concatenate(
        [
            rnd_points() + np.r_[0, 0],
            rnd_points() + np.r_[-radius, 0],
            rnd_points() + np.r_[radius, 0],
            rnd_points() + np.r_[0, radius],
            rnd_points() + np.r_[0, -radius],
        ],
        axis=0,
    )
    M = len(x)
    phase = np.linspace(0, 2 * np.pi, N)
    circle = np.stack([np.cos(phase), np.sin(phase)], axis=-1)
    z = np.concatenate([circle * 1, circle * radius], axis=0)

    with pycrt.Precision(pycrt.Width.SINGLE):
        kwargs = dict(
            x=x,
            z=z,
            n_trans=3,
            isign=-1,
            eps=1e-3,
            real=False,
            nthreads=1,
        )
        # Chunked NUFFT ===================================
        A = nufft.NUFFT.type3(**kwargs, chunked=True)
        t = time.time()
        x_chunks, z_chunks = A.auto_chunk(
            max_mem=100,
            max_anisotropy=1,
            # method="border_search",
            # diagnostic_plot=True,
        )
        t_chunk = time.time() - t
        print(t_chunk)
        A.allocate(x_chunks, z_chunks, direct_eval_threshold=50)
        # Regular NUFFT ===================================
        kwargs.update(nthreads=0)
        B = nufft.NUFFT.type3(**kwargs)

        # Compare memory consumption
        A_mem = A.stats().blk_fft_mem["total"]
        B_mem = np.prod(B.params().fft_shape) * (2 * B._x.dtype.itemsize) / 2**20
        print(f"chunked/full: {A_mem / B_mem}")

        arr = rng.normal(size=(kwargs.get("n_trans") * 3, M))
        if not kwargs.get("real"):
            arr = arr + 1j * rng.normal(size=arr.shape)

    with pycrt.Precision(pycrt.Width.SINGLE):
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
        print(res_fw)
        print(res_bw)
