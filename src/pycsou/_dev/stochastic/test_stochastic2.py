import dask.array as da
import numpy as np

import pycsou._dev as dev
import pycsou.opt.stochastic as pystoc

data = np.arange(2 * 50 * 50).reshape(2, 50 * 50)
data_shape = (50, 50)
chunks = (9, 9)

cdataset = pystoc.ChunkDataset(data, data_shape=data_shape, chunks=chunks)
cdataset.communicate()

Cop = dev.Convolve(data_shape=data_shape, filter=np.arange(49).reshape(7, 7), mode="reflect")
chunkOp = pystoc.ChunkOp(Cop, depth={0: 3, 1: 3}, boundary={0: "reflect", 1: "reflect"})

chunkOp.startup(**cdataset.communicate())

c0 = chunkOp[0]

x0 = da.arange(2 * 50 * 50).reshape(2, 50 * 50)

x0_apply = c0.apply(x0)

x0_adjoint = c0.adjoint(x0_apply)

out = x0_adjoint.reshape(50, 50).compute()
a
