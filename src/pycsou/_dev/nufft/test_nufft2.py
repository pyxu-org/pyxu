import dask.array as da
import dask.distributed as dad
import numpy as np

import pycsou.operator.linop.nufft as nufft
import pycsou.runtime as pycrt
import pycsou.util as pycu

if __name__ == "__main__":
    use_dask = False
    nufft_kwargs = dict(real=False, eps=1e-3, isign=1, n_trans=6, nthreads=0, modeord=1)

    rng = np.random.default_rng(0)
    D, M, N = 2, 200, 5
    N_full = (N,) * D if isinstance(N, int) else N
    x = 2 * np.pi * rng.random(size=(M, D)) - np.pi
    if use_dask:
        client = dad.Client(processes=False)  # processes=True yields a serialization error.
        x = da.from_array(x)

    xp = pycu.get_array_module(x)
    with pycrt.Precision(pycrt.Width.DOUBLE):
        A = nufft.NUFFT.type2(x, N, **nufft_kwargs)
        cB = A.complex_matrix(xp)
        rB = pycu.view_as_real_mat(cB, real_output=nufft_kwargs["real"])

        arr = rng.normal(size=(13, nufft_kwargs["n_trans"], *N_full))
        if not nufft_kwargs["real"]:
            arr = arr + 1j * rng.normal(size=arr.shape)
        arr = arr.reshape((13, nufft_kwargs["n_trans"], -1))
        if use_dask:
            arr = da.from_array(arr)

        rA_out_fw = A.apply(pycu.view_as_real(arr))
        cA_out_fw = pycu.view_as_complex(rA_out_fw)
        rB_out_fw = xp.tensordot(pycu.view_as_real(arr), rB, axes=[[2], [1]])
        cB_out_fw = xp.tensordot(arr, cB, axes=[[2], [1]])

        rA_out_bw = A.adjoint(pycu.view_as_real(cA_out_fw))
        if not nufft_kwargs["real"]:
            cA_out_bw = pycu.view_as_complex(rA_out_bw)
        rB_out_bw = xp.tensordot(rB_out_fw, rB.T, axes=[[2], [1]])
        cB_out_bw = xp.tensordot(cB_out_fw, cB.conj().T, axes=[[2], [1]])
        if nufft_kwargs["real"]:
            cB_out_bw = cB_out_bw.real

        res_fw_r = (xp.linalg.norm(rA_out_fw - rB_out_fw, axis=-1) / xp.linalg.norm(rB_out_fw, axis=-1)).max()
        res_fw_c = (xp.linalg.norm(cA_out_fw - cB_out_fw, axis=-1) / xp.linalg.norm(cB_out_fw, axis=-1)).max()
        res_bw_r = (xp.linalg.norm(rA_out_bw - rB_out_bw, axis=-1) / xp.linalg.norm(rB_out_bw, axis=-1)).max()
        if not nufft_kwargs["real"]:
            res_bw_c = (xp.linalg.norm(cA_out_bw - cB_out_bw, axis=-1) / xp.linalg.norm(cB_out_bw, axis=-1)).max()
        else:
            res_bw_c = res_bw_r

        res = dict(
            zip(
                ["res_fw_r", "res_fw_c", "res_bw_r", "res_bw_c", "eps"],
                pycu.compute(res_fw_r, res_fw_c, res_bw_r, res_bw_c, nufft_kwargs["eps"]),
            )
        )
        print(res)
