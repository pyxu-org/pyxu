import dask.array as da
import numpy as np

import pycsou._dev as dev
import pycsou.opt.stochastic as pystoc

if __name__ == "__main__":

    data = np.arange(2 * 50 * 50).reshape(2, 50 * 50)
    arg_shape = (50, 50)
    chunks = (9, 9)

    cdataset = pystoc.ChunkDataloader(data, arg_shape=arg_shape, chunks=chunks)
    cdataset.communicate()

    Cop = dev.Convolve(arg_shape=arg_shape, filter=np.arange(11 * 11).reshape(11, 11), mode="reflect")
    chunkOp = pystoc.ChunkOp(Cop, depth={0: 5, 1: 5}, boundary={0: "reflect", 1: "reflect"})

    chunkOp.startup(**cdataset.communicate())

    c0 = chunkOp[5]

    x0 = da.arange(2 * 50 * 50).reshape(2, 50 * 50)

    x0_apply = c0.apply(x0)
    val = x0_apply.reshape(2, 9, 5).compute()

    x0_adjoint = c0.adjoint(x0_apply)

    out = x0_adjoint.reshape(2, 50, 50).compute()
