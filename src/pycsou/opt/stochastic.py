import abc
import collections.abc as cabc
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
    "ConvolveLoader",
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

    def __init__(self, path: pyct.PathLike):
        super().__init__(path)

    def _load(self, path):
        if self.gpu:
            data = cp.load(path, mmap_mode="c")
        else:
            data = np.load(path, mmap_mode="c")
        return data, data.shape, data.ndim, data.dtype


class ChunkDataset(cabc.Sequence):
    r"""
    Base class that batches data into blocks using dask as the backend.

    **Initialization parameters of the class:**

    loader : Dataset-type
        Object that makes data available.
    data_shape: pyct.Shape
        how to reshape the data dimension
    blocks : int, tuple
        Shape of 1 batch of data.
    operator : pyco.LinOp
        Global operator that will be used to create batched operators.

    .. Important:: User must override create_op to provide logic for batching of a pycsou operator.

    **Remark 1:**

    Loader is assumed to provide data baseed on the pycsou standard of (stacking dimensions, 1 data dimension).
    For example for a color photograph this would be (C, H*W) with channels as the stacking dimension.

    **Remark 2:**

    Not all batches are guaranteed to be the same size. If the data shape is not divisible by blocks then dask
    will handle the edges by creating smaller blocks.

    """

    def __init__(self, load, data_shape: pyct.Shape, blocks: pyct.Shape, operator: pyco.LinOp):
        self.load = load
        self.blocks = blocks
        self.data_shape = data_shape
        self.global_op = operator
        self.global_shape = self.load.shape
        self._stack_dims = self.global_shape[:-1]

        self.data = (
            da.from_array(self.load)
            .reshape(*self._stack_dims, *data_shape)
            .rechunk(chunks=(*self._stack_dims, *self.blocks))
        )

        self.chunks = self.data.chunks
        self.chunk_dim = [len(c) for c in self.chunks]
        self.indices = (
            da.arange(self.load.size)
            .reshape(*self._stack_dims, *data_shape)
            .rechunk(chunks=(*self._stack_dims, *self.blocks))
        )

    def __getitem__(self, b_index: int) -> tuple[pyct.NDArray, pyco.LinOp, np.array]:
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
        i = np.unravel_index(b_index, self.chunk_dim)

        batch = self.data.blocks[i]
        ind = self.indices.blocks[i]

        op = self.create_op(b_index, batch.shape)

        return (
            batch.compute().reshape(*self.global_shape[:-1], -1),
            op,
            ind.compute().flatten(),
        )

    def __len__(self):
        return self.data.npartitions

    @abc.abstractmethod
    def create_op(self, b_index: int, batch_shape: pyct.Shape) -> pyco.LinOp:
        r"""
        Create an operator that works on a batch of data.

        Parameters
        ----------
        b_index : int
            index for 1 batch of data
        batch_shape : pyct.Shape
            shape of this batch of data

        Returns
        -------
        LinOp
            batched operator


        .. Important:

            This method should abide by the rules described in :ref:`developer-notes`.
        """
        raise NotImplementedError


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


class ConvolveLoader(ChunkDataset):
    r"""
    Batches data and a Convolution Operator.

    **Initialization parameters of the class:**

    loader : Dataset-type
        Object that makes data available.
    data_shape: pyct.Shape
        how to reshape the data dimension
    blocks : int, tuple
        Shape of 1 batch of data.
    operator : pyco.LinOp
        Global operator that will be used to create batched operators.
    depth : dict
        dictionary with the size of padding for each dimension. Padding is symmetric.
    mode : str
        how to pad the edges of the data, see np.pad - https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    """

    def __init__(
        self,
        load,
        data_shape: pyct.Shape,
        blocks: pyct.Shape,
        operator: pyco.LinOp,
        depth: tuple[int, int],
        mode: str = "reflect",
    ):
        super().__init__(load, data_shape, blocks, operator)
        if not isinstance(operator, dev.Convolve):
            raise ValueError("Operator must be a Convolve operator.")
        self.depth = da.overlap.coerce_depth(
            len(depth) + len(self._stack_dims), (*[(0)] * len(self._stack_dims), *depth)
        )
        self.overlap_ind = slices_from_chunks_with_overlap(self.chunks, self.depth)
        self.mode = mode
        self.stacking_shape = self.global_shape[:-1]
        self.pad_data_shape = tuple(
            [d + 2 * self.depth.get(i, 0) for i, d in enumerate((*self.stacking_shape, *self.data_shape))]
        )

    def create_op(self, b_index: int, batch_shape: pyct.Shape) -> pyco.LinOp:
        overlap_slice = self.overlap_ind[b_index]
        overlap_shape = tuple([s.stop - s.start for s in overlap_slice[len(self._stack_dims) :]])

        # TODO how to copy global convolve into local convolve, this is a cheat but
        kernel = self.global_op.filter

        # for dev.Convolve would prefer mode='valid' to save computation as we are handling padding ourselves
        # https://github.com/scipy/scipy/issues/12997
        convolve = dev.Convolve(data_shape=overlap_shape, filter=kernel, mode="constant")

        def apply(arr):
            xp = pycu.get_array_module(arr)
            input_shape = arr.shape
            padding = depth_to_pad(self.depth)

            arr = arr.reshape(*input_shape[:-1], *self.data_shape)
            p_arr = xp.pad(arr, padding, mode=self.mode)

            # grab indices with overlap
            sub_arr = p_arr[tuple([Ellipsis, *overlap_slice])]

            out = convolve.apply(sub_arr.reshape(*input_shape[:-1], -1))

            # trim overlap
            out = out.reshape(*input_shape[:-1], *overlap_shape)
            out = pycuo.unpad(out, padding)
            return out.reshape(*input_shape[:-1], -1)

        def adjoint(arr):
            input_shape = arr.shape
            xp = pycu.get_array_module(arr)
            padding = depth_to_pad(self.depth)

            # pad with zeros, we need to keep information in padded region when the region is shared inside an image
            arr = xp.pad(arr.reshape(*batch_shape), padding)

            out = convolve.adjoint(arr.reshape(*input_shape[:-1], -1))
            out = out.reshape(*input_shape[:-1], *overlap_shape)

            # create array size of gradient to place our block gradient into (plus padding to then trim edges, this
            # also trims the edges of our block)
            new_arr = xp.zeros(self.pad_data_shape)
            new_arr[tuple([Ellipsis, *overlap_slice])] = out
            new_arr = pycuo.unpad(new_arr, padding)

            return new_arr.reshape(*input_shape[:-1], -1)

        op = pyco.LinOp(shape=(np.prod(overlap_shape), np.prod(overlap_shape)))
        op.apply = types.MethodType(lambda _, arr: apply(arr), op)
        op.adjoint = types.MethodType(lambda _, arr: adjoint(arr), op)

        return op


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

    def __init__(self, loader, shuffle: bool = False, seed: int = None):
        self.loader = loader
        self._epochs = 0
        self.counter = 0
        self.batch_counter = 0
        self.num_batches = len(self.loader)
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
        y, op, ind = self.loader[self.batch_counter]
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
        self.global_op = self.batch.loader.global_op
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
