import abc
import collections.abc as cabc
import types
import typing as typ
import warnings

import dask.array as da
import numpy as np
import sparse

import pycsou._dev as dev
import pycsou.abc.operator as pyco
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou.opt as pyo


class Load(cabc.Sequence):
    r"""
    Base class for loading data.

    We use dask to batch the data, therefore we must provide .shape, .ndim, .dtype and support numpy-style slicing.
    """
    def __init__(self, path: pyct.PathLike):
        self.path = path
        self.data = self._load(self.path)

    def __getitem__(self, ind: typ.Union[int, ]) -> pyct.NDArray:
        return self.data[ind]

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _load(self, path: pyct.PathLike):
        raise NotImplementedError


class NpzLoad(Load):
    r"""
    NPZ file format loader.

    Uses mmap_mode which allows us to access small segments of large files on disk, without reading the entire file
    into memory.
    """
    def __init__(self, path: pyct.PathLike):
        super().__init__(path)
        self.size = self.data.size
        self.shape = self.data.shape
        self.ndim = len(self.shape)
        self.dtype = self.data.dtype

    def _load(self, path):
        return np.load(path, mmap_mode='c')


class Hdf5Load(Load):
    def __init__(self, path):
        super().__init__(path)

    def _load(self, path):
        # TODO
        raise NotImplementedError


class BlockLoader(cabc.Sequence):
    r"""
    Base class that batches data into blocks using dask as the backend.

    .. Important:: User must override create_op to provide logic for batching of a pycsou operator.

    Parameters
    ----------
    loader : Load-type
        Object that makes data available.
    blocks : int, tuple
        Shape of 1 batch of data.
    operator : pyco.LinOp
        Global operator that will be used to create batched operators.

    Notes
    -----
    Loader is assumed to provide data baseed on the pycsou standard of (stacking dimensions, 1 data dimension).
    For example for a color photograph this would be (C, H*W) with channels as the stacking dimension.

    Not all batches are guaranteed to be the same size. If the data shape is not divisible by blocks then dask
    will handle the edges by creating smaller blocks.

    """
    def __init__(self, loader, data_shape: pyct.Shape, blocks: pyct.Shape, operator: pyco.LinOp):
        self.loader = loader
        self.blocks = blocks
        self.data_shape = data_shape
        self.global_op = operator
        self.global_shape = self.loader.shape
        self._stack_dims = self.global_shape[:-1]

        self.data = da.from_array(
            self.loader
        ).reshape(
            *self._stack_dims, *data_shape
        ).rechunk(
            chunks=(*self._stack_dims, *self.blocks)
        ).persist()

        self.chunks = self.data.chunks
        self.chunk_dim = [len(c) for c in self.chunks]
        self.indices = da.arange(
            self.loader.size
        ).reshape(
            *self._stack_dims, *data_shape
        ).rechunk(
            chunks=(*self._stack_dims, *self.blocks)
        ).persist()

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

        return batch.compute().reshape(*self.global_shape[:-1], -1), op, ind.compute().flatten()

    def __len__(self):
        return self.data.npartitions

    @staticmethod
    def create_op(b_index: int, batch_shape: pyct.Shape) -> pyco.LinOp:
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

        .. Important::
            This method should abide by the rules described in :ref:`developer-notes`.

        """
        raise NotImplementedError


class ConvolveLoader(BlockLoader):
    r"""
    Batches data and a Convolution Operator.

    Parameters
    ----------
    loader : Load-type
        Object that makes data available.
    blocks : int, tuple
        Shape of 1 batch of data.
    operator : pyco.LinOp
        Global operator that will be used to create batched operators.
    depth : dict
        dictionary with the size of padding for each dimension
    mode : str
        how to pad the edges of the data
        see np.pad - https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    """
    def __init__(self, loader, data_shape: pyct.Shape, blocks: pyct.Shape, operator: pyco.LinOp, depth: tuple[int, int], mode: str = 'reflect'):
        super().__init__(loader, data_shape, blocks, operator)
        if not isinstance(operator, dev.Convolve):
            raise ValueError("Operator must be a Convolve operator.")
        self.depth = da.overlap.coerce_depth(len(depth) + len(self._stack_dims), (*[(0)]*len(self._stack_dims), *depth))
        self.overlap_ind = pyo.slices_from_chunks_with_overlap(self.chunks, self.depth)
        self.mode = mode
        self.stacking_shape = self.global_shape[:-1]
        self.pad_data_shape = tuple([d + 2 * self.depth.get(i, 0) for i, d in enumerate((*self.stacking_shape, *self.data_shape))])
        self.monkey_shape = None
    def create_op(self, b_index: int, batch_shape: pyct.Shape) -> pyco.LinOp:
        overlap_slice = self.overlap_ind[b_index]
        overlap_shape = tuple([s.stop - s.start for s in overlap_slice[len(self._stack_dims):]])

        # TODO how to copy global convolve into local convolve
        kernel = self.global_op.filter
        # would prefer mode='valid' to save computation as we are handling padding ourselves
        # https://github.com/scipy/scipy/issues/12997
        convolve = dev.Convolve(data_shape=overlap_shape, filter=kernel, mode='constant')
        def apply(arr):
            xp = pycu.get_array_module(arr)
            input_shape = arr.shape
            padding = pyo.depth_to_pad(self.depth)

            arr = arr.reshape(*input_shape[:-1], *self.data_shape)
            p_arr = xp.pad(arr, padding, mode=self.mode)

            # grab indices with overlap
            sub_arr = p_arr[tuple([Ellipsis, *overlap_slice])]

            out = convolve.apply(sub_arr.reshape(*input_shape[:-1], -1))

            # trim overlap
            out = out.reshape(*input_shape[:-1], *overlap_shape)
            out = pyo.unpad(out, padding)
            self.monkey_shape = out.reshape(*input_shape[:-1], -1).shape
            return out.reshape(*input_shape[:-1], -1)

        def adjoint(arr):
            #TODO monkey patch because input shape is wrong
            #arr = arr.reshape(self.monkey_shape)
            input_shape = arr.shape
            xp = pycu.get_array_module(arr)
            padding = pyo.depth_to_pad(self.depth)

            # pad with zeros, we need to keep information in padded region when the region is shared inside an image
            arr = xp.pad(arr.reshape(*batch_shape), padding)

            out = convolve.adjoint(arr.reshape(*input_shape[:-1], -1))
            out = out.reshape(*input_shape[:-1], *overlap_shape)

            # create array size of gradient to place our block gradient into (plus padding to then trim edges, this
            # also trims the edges of our block)
            new_arr = xp.zeros(self.pad_data_shape)
            new_arr[tuple([Ellipsis, *overlap_slice])] = out
            new_arr = pyo.unpad(new_arr, padding)

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

    Parameters
    ----------
    loader : BlockLoader
        loader to get data and batched operator from
    shuffle : bool
        if batches are shuffled

    """
    def __init__(self, loader, shuffle: bool = False):
        self.loader = loader
        self._epochs = 0
        self.counter = 0
        self.batch_counter = 0
        self.num_batches = len(self.loader)
        self.shuffle = shuffle
        if self.shuffle:
            rng = np.random.default_rng()
            self.shuffled_batch = rng.permutation(np.arange(self.num_batches))

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

    def batches(self) -> typ.Tuple[pyct.NDArray, pyco.LinOp, np.array]:
        y, op, ind = self.loader[self.batch_counter]
        self._bookkeeping()
        yield y, op, ind


class Strategy(abc.ABC):
    r"""
    Base class for gradient update strategy.
    """
    @staticmethod
    def apply(grad):
        raise NotImplementedError


class SGD(Strategy):
    r"""

    ..math SGD is an update of the form

    """
    def apply(self, grad):
        return grad

# TODO -> since the batches are the same everytime (assumption) we would only need to keep a vector per batch, not a vector for every item in the batch.
# so the grad book would be (batches, dim) and we would take batch_index instead of ind...
# wouldn't work if we had disjoint batching strategy, but would speed things a lot otherwise.
class SAGA(Strategy):
    r"""

    ..math

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

        #TODO in jupyter this works quickly, but here it is super slow to subset grad_book....
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
            warnings.warn(f"grad density larger than {DENSITY_WARNING}, SAGA will perform sub-optimally.", UserWarning)
        # get coordinates of data
        grad_cord = sparse_grad.coords.reshape(-1)
        y_cord = tuple(np.tile(grad_cord, self._ind.size))
        x_cord = tuple(np.repeat(self._ind, grad_cord.size))

        self.grad_book[x_cord, y_cord] = np.tile(sparse_grad.data, self._ind.size)


class Stochastic(pyco.DiffFunc):
    f"""
    
    
    """

    def __init__(self, f, batch, strategy=None):
        self._f = f
        super().__init__(self._f.shape)
        if strategy:
            self.strategy = strategy
        else:
            self.strategy = SGD()
        self.batch = batch
        self.global_op = self.batch.loader.global_op
        n = self.global_op.shape[0]
        self._diff_lipschitz = (1/n * self._f * self.global_op).diff_lipschitz()

    def apply(self, x):
        raise NotImplementedError

    def grad(self, x):
        y, op, ind = next(self.batch.batches())
        batch_size = y.shape[-1]
        f_hat = 1 / batch_size * self._f.asloss(y)
        stoc_loss = f_hat * op
        grad = stoc_loss.grad(x)
        if hasattr(self.strategy, 'indices'):
            setattr(self.strategy, 'indices', ind)
        full_grad = self.strategy.apply(grad)
        return full_grad
