import abc
import collections.abc as cabc
import functools
import itertools
import operator
import types
import typing as typ
import warnings

import dask.array as da
import numpy as np
import sparse
import toolz

import pycsou._dev as dev
import pycsou.abc.operator as pyco
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.operator as pycuo
import pycsou.util.ptype as pyct

if pycd.CUPY_ENABLED:
    import cupy as cp

__all__ = [
    "Dataset",
    "NpzDataset",
    "ChunkDataset",
    "ChunkOp",
    "Batch",
    "SGD",
    "Stochastic",
]


class Dataset(cabc.Sequence):
    r"""
    Base class for loading data.

    We use dask to batch the data, therefore we must provide .shape, .ndim, .dtype and support numpy-style slicing.

    **Initialization parameters of the class:**

    path : pyct.PathLike
        Filepath to load
    """

    def __init__(self, path: pyct.PathLike, gpu: bool):
        self.path = path
        self.gpu = gpu
        self.data, self._shape, self._ndim, self._dtype = self._load(self.path)
        self._size = np.prod(self._shape)

    def __getitem__(
        self,
        ind: typ.Union[
            int,
        ],
    ) -> pyct.NDArray:
        return self.data[ind]

    def __len__(self) -> int:
        return len(self.data)

    @abc.abstractmethod
    def _load(self, path: pyct.PathLike):
        raise NotImplementedError

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self):
        return self._size


class NpzDataset(Dataset):
    r"""
    NPZ file format loader.

    Uses mmap_mode which allows us to access small segments of large files on disk, without reading the entire file
    into memory.

    **Initialization parameters of the class:**

    path : pyct.PathLike
        Filepath to load
    """

    def __init__(self, path: pyct.PathLike, gpu: bool = False):
        super().__init__(path, gpu)

    def _load(self, path):
        if self.gpu:
            data = cp.load(path, mmap_mode="c")
        else:
            data = np.load(path, mmap_mode="c")
        return data, data.shape, data.ndim, data.dtype


# Reason to pass chunck_op in here is that I only want to have the ChunkDataset/ChunkOp to be
# specific to them. If I do it in BatchOp I have to write code specific to this chuncking that it makes
# batch op not general anymore.
class ChunkDataset(cabc.Sequence):
    r"""
    Base class that batches data into chunks using dask as the backend.

    **Initialization parameters of the class:**

    loader : Dataset-type
        Object that makes data available.
    data_shape: pyct.Shape
        how to reshape the data dimension
    chunks : int, tuple
        Shape of 1 batch of data.
    operator : pyco.LinOp
        Global operator that will be used to create batched operators.

    .. Important:: User must override create_op to provide logic for batching of a pycsou operator.

    **Remark 1:**

    Loader is assumed to provide data baseed on the pycsou standard of (stacking dimensions, 1 data dimension).
    For example for a color photograph this would be (C, H*W) with channels as the stacking dimension.

    **Remark 2:**

    Not all batches are guaranteed to be the same size. If the data shape is not divisible by chunks then dask
    will handle the edges by creating smaller chunks.

    """

    # TODO what if load is already a dask array -> redirect or if statement
    def __init__(self, load, data_shape, chunks: pyct.Shape):
        self.load = load
        self.data_shape = data_shape
        self.data_ndim = len(self.data_shape)

        self.chunks = chunks
        self.global_shape = self.load.shape
        self.stack_shape = self.global_shape[:-1]

        self.data = (
            da.from_array(self.load)
            .reshape(*self.stack_shape, *self.data_shape)
            .rechunk(chunks=(*self.stack_shape, *self.chunks))
        )

        # Went with name chunkLoader, but it doesn't correspond to what dask calls chunks. Changed name to blocks.
        self.blocks = self.data.chunks
        self.block_dim = [len(c) for c in self.blocks]
        self.indices = (
            da.arange(self.load.size)
            .reshape(*self.stack_shape, *self.data_shape)
            .rechunk(chunks=(*self.stack_shape, *self.chunks))
        )

    def __getitem__(self, b_index: int) -> tuple[pyct.NDArray, pyct.NDArray]:
        r"""
        Parameters
        ----------
        b_index : int
            index for 1 batch of data

        Returns
        -------
        NDArray
            flattened batch of data
        LinOp
            batched operator according to create_op
        np.array
            ND-indices flattened
        """
        i = np.unravel_index(b_index, self.block_dim)

        batch = self.data.blocks[i]
        ind = self.indices.blocks[i]

        return (batch.compute().reshape(*self.stack_shape, -1), ind.compute().flatten())

    def __len__(self):
        return self.data.npartitions

    def communicate(self):
        return {"blocks": self.blocks, "block_dim": self.block_dim, "stack_shape": self.stack_shape}


def depth_to_pad(depth: dict, ndims: int = 0) -> list[tuple]:
    r"""
    Converts from dask depth to numpy pad inputs.

    **Note**
    Depth is assumed to be coerced using dask.array.overlap.coerce_depth. This means every dimension will have a depth
    key, even if that key is zero.

    Examples:
    -------
    >>> depth_to_pad({0:1, 1:2, 2:0})
    [(1,1), (2,2), (0,0)]
    """
    initial = [(0, 0)] * ndims
    initial.extend([(v, v) for _, v in depth.items()])
    return initial


def _cumsum(seq, initial):
    r"""
    Modified from dask.utils._cumsum - https://github.com/dask/dask/blob/main/dask/utils.py
    Can take an initial value other than zero.
    """
    return tuple(toolz.accumulate(operator.add, seq, initial=initial))


def slices_from_chunks_with_overlap(chunks: tuple[tuple[int]], depth: dict):
    r"""
    Translates dask chunks tuples into a set of slices in product order. Takes into account padding and block overlaps.

    Modified from dask.array.core.slices_from_chunks - https://github.com/dask/dask/blob/main/dask/array/core.py

    **Remark 1:**
    The depth padding is assumed to be symmetric around each dimension. This is the convention dask uses, and we follow
    it.

    **Remark 2:**
    Depth is assumed to be coerced using dask.array.overlap.coerce_depth. This means every dimension will have a depth
    key, even if that key is zero.

    Examples:
    -------
    >>> slices_from_chunks_with_overlap(chunks=((2, 2), (3, 3, 3)), depth={0:1, 1:2})
     [(slice(0, 4, None), slice(0, 7, None)),
      (slice(0, 4, None), slice(3, 10, None)),
      (slice(0, 4, None), slice(6, 13, None)),
      (slice(2, 6, None), slice(0, 7, None)),
      (slice(2, 6, None), slice(3, 10, None)),
      (slice(2, 6, None), slice(6, 13, None))]
    """
    cumdims = [_cumsum(bds, initial=depth[i]) for i, bds in enumerate(chunks)]
    slices = [
        [slice(s - depth[i], s + dim + depth[i]) for s, dim in zip(starts, shapes)]
        for i, (starts, shapes) in enumerate(zip(cumdims, chunks))
    ]
    return list(itertools.product(*slices))


class ChunkOp(pyco.LinOp):
    def __init__(self, op: pyco.LinOp, depth=None, boundary=None):
        # TODO check that depth is not larger than any chunk (otherwise is rechunked which is not good)
        self.op = op
        self.data_shape = self.op.data_shape
        self.data_ndim = len(self.data_shape)
        super().__init__(shape=self.op.shape)
        self.depth = depth
        self.boundary = boundary
        self._diff_lipschitz = self.op.diff_lipschitz()
        self._lipschitz = self.op.lipschitz()

        # TODO create Stochasitc operator subclass to check against...
        # if not isinstance(op, dev.Convolve):
        #     raise ValueError("Operator must be a Convolve operator.")

    def startup(self, blocks, block_dim, stack_shape, *args, **kwargs):
        self.blocks = blocks
        self.block_dim = block_dim
        self.slices = da.core.slices_from_chunks(self.blocks)
        self.stack_shape = stack_shape
        stack_ndim = len(self.stack_shape)

        self.depth = self._coerce_condition(self.depth, self.data_ndim + stack_ndim, stack_ndim)
        self.boundary = self._coerce_condition(self.boundary, self.data_ndim + stack_ndim, stack_ndim, default=None)
        for i, (k, v) in enumerate(self.depth.items()):
            if v > 0:
                if not all([b >= v for b in self.blocks[i]]):
                    raise ValueError(
                        f"{self.blocks[i]} has a chunk/chunks smaller than depth: {v}. Provide a different chunking size."
                    )

    def __getitem__(self, b_index):
        """ """
        self.b_index = b_index
        self.index = np.unravel_index(self.b_index, self.block_dim)
        self.batch_slice = self.slices[self.b_index]
        self.overlap_shape = [s.stop - s.start + 2 * self.depth[i] for i, s in enumerate(self.batch_slice)][
            -self.data_ndim :
        ]
        self.batch_shape = [s.stop - s.start for s in self.batch_slice[-self.data_ndim :]]
        self.op.data_shape = self.overlap_shape
        return self

    def _coerce_condition(self, condition, ndim, num_stack, default=0):
        return dict(((i, default) if i < num_stack else (i, condition[i - num_stack]) for i in range(ndim)))

    def apply(self, arr):
        xp = pycu.get_array_module(arr)
        input_shape = arr.shape

        if xp != da:
            arr = da.from_array(arr)

        arr = arr.reshape(*input_shape[:-1], *self.data_shape).rechunk(self.blocks)

        out = da.map_overlap(
            lambda x: self.op.apply(x.reshape(*input_shape[:-1], -1)).reshape(x.shape),
            arr,
            depth=self.depth,
            boundary=self.boundary,
            # allow_rechunk=True, - released 2022.6.1 DASK
            meta=xp.array(()),
            dtype=arr.dtype,
        )

        out = out[self.batch_slice]

        if xp != da:
            return out.reshape(*input_shape[:-1], -1).compute()
        else:
            return out.reshape(*input_shape[:-1], -1)

    def adjoint(self, arr):
        input_shape = arr.shape
        xp = pycu.get_array_module(arr)

        if xp != da:
            out = da.from_array(xp.zeros((*input_shape[:-1], *self.data_shape)), chunks=self.blocks)
        else:
            out = da.zeros((*input_shape[:-1], *self.data_shape), chunks=self.blocks)

        out[self.batch_slice] = arr.reshape((*input_shape[:-1], *self.batch_shape))

        # TODO there is some problem in the smaller regions, that aren't the same size.
        # potentially has to do with
        n_map_overlap = functools.partial(
            neighbors_map_overlap,
            op=self.op,
            ind=self.index,
            stack_dims=input_shape[:-1],
            overlap=bool(self.depth),
        )
        out = da.map_overlap(
            n_map_overlap,
            out,
            depth=self.depth,
            boundary=None,
            # allow_rechunk=False,- released 2022.6.1 DASK
            meta=xp.array(()),
            dtype=arr.dtype,
        )

        if xp != da:
            return out.reshape(*input_shape[:-1], -1).compute()
        else:
            return out.reshape(*input_shape[:-1], -1)


def neighbors_map_overlap(x, op, ind, overlap, stack_dims, block_info=None):
    save_shape = x.shape
    block_id = block_info[0]["chunk-location"]
    array_location = block_info[0]["array-location"]

    if block_id:
        # subset to only be dimensions of interest (not stacking dimensions)
        block_id = block_id[len(stack_dims) :]
        array_location = array_location[len(stack_dims) :]
        ind = ind[len(stack_dims) :]

        # if dimensions of interest within +/-1, apply function
        if all([i - int(overlap) <= j <= i + int(overlap) for i, j in zip(ind, block_id)]):
            print(f"block_id: {block_id} array_location: {array_location} shape: {x.shape}")
            shape_overload = [s[1] - s[0] for s in array_location]
            op.data_shape = tuple(shape_overload)
            return op.adjoint(x.reshape(*stack_dims, -1)).reshape(save_shape)
    return x


class Batch:
    r"""
    Decides batching strategy and provides a generator that returns everything that dynamically changes for stochastic
    optimization.

    Keeps track of internal stochastic opt variables of iterations and epochs.

    **Initialization parameters of the class:**

    loader : ChunkDataset
        loader to get data and batched operator from
    shuffle : bool
        if batches are shuffled
    seed : int
        provides a seed to the shuffle generator

        * None - random shuffles every epoch
        * int - shuffle according to seed, and keep this shuffle order every epoch
    """

    def __init__(self, batch_dataset, batch_op, shuffle: bool = False, seed: int = None):
        self.batch_dataset = batch_dataset
        self.batch_op = batch_op
        self._communicate()

        self._epochs = 0
        self.counter = 0
        self.batch_counter = 0
        # TODO is this the right way to set num_batches if we have a stacking dimension?
        self.num_batches = len(self.batch_dataset)
        self.shuffle = shuffle
        self.seed = seed
        if self.shuffle:
            self.shuffled_batch = self._shuffle()

    @property
    def epochs(self) -> int:
        r"""
        Returns
        -------
        int
        """
        return self._epochs

    def _communicate(self):
        kwargs = self.batch_dataset.communicate()
        self.batch_op.startup(**kwargs)

    def _bookkeeping(self):
        if self.shuffle:
            self.batch_counter = self.shuffled_batch[self.counter]
        else:
            self.batch_counter = self.counter
        self.counter += 1
        if self.counter >= self.num_batches:
            self.counter = 0
            self._epochs += 1
            if (not self.seed) and self.shuffle:
                self.shuffled_batch = self._shuffle()

    def _shuffle(self):
        rng = np.random.default_rng(self.seed)
        return rng.permutation(np.arange(self.num_batches))

    def batches(self) -> typ.Tuple[pyct.NDArray, pyco.LinOp, np.array]:
        y, ind = self.batch_dataset[self.batch_counter]
        op = self.batch_op[self.batch_counter]
        self._bookkeeping()
        yield y, op, ind


class GradStrategy(abc.ABC):
    r"""
    Base class for gradient update strategy.
    """

    @abc.abstractmethod
    def apply(self, grad: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        grad : pyct.NDArray
            stochastic gradient computed by :py:func:`pycsou.opt.stochastic.Stochastic`

        Returns
        -------
        pyct.NDArray

        """
        raise NotImplementedError


class SGD(GradStrategy):
    r"""
    Supports either Stochastic Gradient Descent or Batch Gradient Descent.

    :math:`x_{t+1} = x_t - \eta_t \nabla f_i(x)`


    Allows the gradient to flow through unchanged as :py:func:`pycsou.opt.stochastic.Stochastic` takes care of computing
    the gradient stochastically.

    It is the base strategy for :py:func:`pycsou.opt.stochastic.Stochastic` if None is given.
    """

    def apply(self, grad: pyct.NDArray) -> pyct.NDArray:
        return grad


# TODO -> since the batches are the same everytime (assumption) we would only need to keep a vector per batch, not a vector for every item in the batch.
# so the grad book would be (batches, dim) and we would take batch_index instead of ind...
# wouldn't work if we had disjoint batching strategy, but would speed things a lot otherwise.
class SAGA(GradStrategy):
    r"""
    TODO
    https://www.di.ens.fr/~fbach/Defazio_NIPS2014.pdf
    """

    def __init__(self, n, dim):
        self.n = n
        self.dim = dim
        self.grad_book = sparse.DOK((self.n, self.dim))
        self.grad_book_sum = None
        self._ind = None

    @property
    def indices(self):
        return self._ind

    @indices.setter
    def indices(self, ind):
        self._ind = ind

    def apply(self, grad):
        xp = pycu.get_array_module(grad)
        # create internals with same type as grad (vs in __init__)
        if self.grad_book_sum is None:
            self.grad_book_sum = xp.zeros(self.dim)
        batch = len(self._ind)

        # TODO in jupyter this works quickly, but here it is super slow to subset grad_book.... has to do with updating each iteration...
        old_grad = self.grad_book[self._ind, :]
        # convert to coo, sum, make as numpy array
        old_grad = old_grad.to_coo().sum(axis=0).todense()
        grad_diff = grad - 1 / batch * old_grad
        saga_grad = grad_diff + 1 / self.n * self.grad_book_sum
        self._bookkeeping(grad, old_grad)
        return saga_grad

    def warm_up(self):
        pass

    def _bookkeeping(self, grad, old_grad):
        self.grad_book_sum -= old_grad
        self.grad_book_sum += grad * len(self._ind)

        # grad is sparse, convert to sparse array format
        sparse_grad = sparse.COO.from_numpy(grad)
        DENSITY_WARNING = 0.05
        if sparse_grad.density > DENSITY_WARNING:
            warnings.warn(
                f"grad density larger than {DENSITY_WARNING}, SAGA will perform sub-optimally.",
                UserWarning,
            )
        # get coordinates of data
        grad_cord = sparse_grad.coords.reshape(-1)
        y_cord = tuple(np.tile(grad_cord, self._ind.size))
        x_cord = tuple(np.repeat(self._ind, grad_cord.size))

        self.grad_book[x_cord, y_cord] = np.tile(sparse_grad.data, self._ind.size)


class Stochastic(pyco.DiffFunc):
    r"""
    Creates batched differentiable objective functions and computes the gradient.
    Enables computing stochastic gradients within the Pycsou framework.
    Takes the place of a DiffFunc as a data fidelity term.

    **Initialization parameters of the class:**

    f : pyco.DiffFunc
        data fidelity term
    batch : Batch
        generator to query and get batched data and a batched operator
    strategy : GradStrategy = None
        which gradient strategy to use. Default is :py:func:`pycsou.opt.stochastic.SGD`

    """

    def __init__(self, f: pyco.DiffFunc, batch: Batch, strategy: GradStrategy = None):
        self._f = f
        super().__init__(self._f.shape)
        if strategy:
            self.strategy = strategy
        else:
            self.strategy = SGD()
        self.batch = batch
        self.global_op = self.batch.batch_op.op
        n = self.global_op.shape[0]
        self._diff_lipschitz = (1 / n * self._f * self.global_op).diff_lipschitz()

    def apply(self, x):
        raise NotImplementedError

    def grad(self, x: pyct.NDArray) -> pyct.NDArray:
        r"""
        Computes :math:`\nabla f_i(x)` and potentially applies a gradient strategy.

        Parameters
        ----------
        x : pyct.NDArray
            variable to optimize

        Returns
        -------
        pyct.NDArray
            gradient of objective function with respect to x
        """
        y, op, ind = next(self.batch.batches())
        batch_size = y.shape[-1]
        f_hat = 1 / batch_size * self._f.asloss(y)
        stoc_loss = f_hat * op
        grad = stoc_loss.grad(x)
        if hasattr(self.strategy, "indices"):
            setattr(self.strategy, "indices", ind)
        full_grad = self.strategy.apply(grad)
        return full_grad
