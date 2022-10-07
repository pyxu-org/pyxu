import time as t
import typing as typ
from functools import partial

import dask.array as da
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


class Spreading(pyco.LinOp):
    @pycrt.enforce_precision(i=["meshgrid", "knots"], o=False)
    def __init__(
        self,
        kernel: typ.Callable,
        scope: float,
        meshgrid: pyct.NDArray,
        knots: pyct.NDArray,
        chunks: typ.Union[tuple[int], int, typ.Literal["auto"]] = "auto",
        boundaries: typ.Literal["none", "periodic"] = "none",  # Periodic does not work must change distance
        ord: float = 2,
    ):
        assert scope > 0, f"Parameter scope must be positive, got {scope}."
        assert (
            meshgrid.shape[-1] == knots.shape[-1]
        ), f"Dimensionality mismatch between grid points and knots: {meshgrid.shape[-1]} != {knots.shape[-1]}."
        super(Spreading, self).__init__(shape=(knots.shape[0], np.prod(meshgrid.shape[:-1])))
        self._kernel = kernel
        self._scope = scope
        self._chunks = chunks
        self._meshgrid = meshgrid
        if not isinstance(chunks, tuple):
            self._meshgrid_da = da.asarray(
                meshgrid, chunks=tuple([self._chunks for _ in range(meshgrid.shape[-1])]) + (-1,)
            )
        else:
            self._meshgrid_da = da.asarray(meshgrid, chunks=self._chunks)
        self._domain_sizes = [self._meshgrid[..., dim].ptp() for dim in range(self._meshgrid.ndim - 1)]
        self._knots = knots
        self._blocked_knots, self._blocked_knots_indices = self._assign_knots_to_blocks()
        self._boundaries = boundaries
        self._ord = ord

    def _assign_knots_to_blocks(self) -> dict[tuple, pyct.NDArray]:
        xp = pycu.get_array_module(self._knots)
        step_sizes = xp.array(self._domain_sizes) / xp.array(self._meshgrid.shape[:-1])
        min_coords = xp.array([self._meshgrid[..., dim].min() for dim in range(self._meshgrid.shape[-1])])
        numblocks = self._meshgrid_da.numblocks[:-1]
        knots_arr_id = (self._knots - min_coords) // step_sizes
        step_sizes_da = xp.array(self._meshgrid.shape[:-1]) / xp.array(numblocks)
        knots_chunk_id = knots_arr_id // step_sizes_da
        blocked_knots = {}
        blocked_knots_indices = {}
        for ind in np.ndindex(*numblocks):
            blocked_knots[ind] = self._knots[xp.all(knots_chunk_id == xp.array(ind), axis=-1), :]
            blocked_knots_indices[ind] = xp.nonzero(xp.all(knots_chunk_id == xp.array(ind), axis=-1))
        return pycu.compute(blocked_knots, mode="persist"), pycu.compute(blocked_knots_indices, mode="persist")

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        # if arr.ndim == 1:
        #     weights = arr[None, :]
        # else:
        #     weights = arr.reshape(-1, arr.shape[-1])
        weights = arr
        xp = pycu.get_array_module(weights)
        partial_spreading = partial(self._spreading, weights=weights)
        out = self._meshgrid_da.map_blocks(
            partial_spreading,
            dtype=pycrt.getPrecision().value,
            meta=xp.array((), dtype=pycrt.getPrecision().value),
            drop_axis=self._meshgrid_da.ndim - 1,
        )
        # depth={i: self._scope for i in
        #       range(self._meshgrid_da.ndim - 1)},
        # boundary={i: self._boundaries for i in
        #          range(self._meshgrid_da.ndim - 1)})
        return out

    def _spreading(self, mesh_block: pyct.NDArray, weights: pyct.NDArray, block_info=None):
        xp = pycu.get_array_module(weights)
        chunki, chunkj = block_info[0]["chunk-location"][:-1][::-1]
        sub_knots = []
        sub_weights = []
        for i in range(chunki - 1, chunki + 2):
            for j in range(chunkj - 1, chunkj + 2):
                if self._boundaries == "periodic":
                    sub_knots.append(
                        self._blocked_knots[
                            (
                                np.mod(i, block_info[0]["num-chunks"][0]).astype(int),
                                np.mod(j, block_info[0]["num-chunks"][1]).astype(int),
                            )
                        ]
                    )
                    sub_weights.append(
                        weights[
                            self._blocked_knots_indices[
                                (
                                    np.mod(i, block_info[0]["num-chunks"][0]).astype(int),
                                    np.mod(j, block_info[0]["num-chunks"][1]).astype(int),
                                )
                            ]
                        ].ravel()
                    )
                else:
                    if i < 0 or i >= block_info[0]["num-chunks"][0]:
                        pass
                    elif j < 0 or j >= block_info[0]["num-chunks"][1]:
                        pass
                    else:
                        sub_knots.append(self._blocked_knots[(i, j)])
                        sub_weights.append(weights[self._blocked_knots_indices[(i, j)]].ravel())
        sub_knots = xp.concatenate(sub_knots, axis=0)
        sub_weights = xp.concatenate(sub_weights, axis=0)
        if sub_knots.size == 0:
            return xp.zeros_like(mesh_block[..., 0])
        else:
            if xp == da:
                pass
                # val = (sub_weights[..., None, None, :] * (
                #     xp.sqrt((mesh_block[..., None] - sub_knots.T[None, None, ...]) ** self._ord, axis=-2,
                #             order=self._ord)).map_block(
                #     self._kernel, dtype=pycrt.getPrecision().value)).sum(axis=-1)
            else:
                val = (
                    sub_weights
                    * self._kernel(
                        xp.linalg.norm(mesh_block[..., None] - sub_knots.T[None, None, ...], axis=-2, ord=self._ord)
                    )
                ).sum(axis=-1)
            return val


if __name__ == "__main__":
    # 2D meshgrid
    x = np.linspace(-1, 1, 1024)
    y = np.linspace(-1, 1, 1024)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X, Y], axis=-1)

    # Knots/weights chosen at random
    rng = np.random.default_rng(0)
    knots = 2 * rng.random(size=(64, 2)) - 1
    weights = rng.random(size=(64))

    # Kernel
    kernel_size = 8
    sigma = 0.02
    kernel = lambda r: np.exp(-((r) ** 2) / (2 * sigma**2))
    spreading_op = Spreading(
        kernel=kernel, scope=kernel_size, meshgrid=XY, knots=knots, chunks=128, boundaries="none", ord=2.0
    )
    t1 = t.time()
    x = spreading_op.apply(weights).compute()
    print(t.time() - t1)
    t1 = t.time()
    y = (weights * kernel(np.linalg.norm(XY[..., None] - knots.T[None, None, ...], axis=-2))).sum(axis=-1)
    print(t.time() - t1)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(x)
    plt.subplot(1, 2, 2)
    plt.imshow(y)

# plt.figure()
# blockx, blocky = np.linspace(-1, 1, XYda.numblocks[0] + 1), np.linspace(-1, 1, XYda.numblocks[1] + 1)
# block_colors = np.arange((blockx.size - 1) * (blocky.size - 1)).reshape((blockx.size - 1, blocky.size - 1)) % 20
# plt.pcolormesh(blockx, blocky, block_colors, alpha=0.5, cmap='tab20', shading='flat')
# for key, samp in sample_blocks.items():
#     if samp.size > 0:
#         plt.scatter(samp[:, 0], samp[:, 1], s=8, marker='s',
#                     c=[block_colors[key[1], key[0]] for _ in range(samp[:, 0].size)], cmap='tab20', vmin=0, vmax=19)
