import abc
import collections.abc as cabc
import functools
import typing as typ
import warnings

import dask.array as da
import numpy as np
import sparse

import pycsou.abc.operator as pyco
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

if pycd.CUPY_ENABLED:
    import cupy as cp

__all__ = [
    "Dataset",
    "NpzDataset",
    "DataLoader",
    "ChunkDataloader",
    "BatchOp",
    "ChunkOp",
    "Batch",
    "GradStrategy",
    "SGD",
    "Stochastic",
]


# TODO refacotor base classes into abc Dataset,
# what is wrong - how to get the class attributes necessary (shape, ndim, dtype) without this weird _load() thing
class Dataset(cabc.Sequence):
    r"""
    An abstract base class representing :py::class:`Dataset`.

    All subclasses should overwrite :py:meth:`~pycsou.opt.stochastic.Dataset._load` which should output an
    object holding the data and the relevant attributes.

    Parameters
    ----------
    path : pyct.Path
        Filepath to load

    gpu : bool
        load data in gpu or cpu format

    Notes
    -----
    If the dataset is very large, the data object can be constructed to not load the entire dataset in memory.
    Rather the dataset can be read piece by piece from disk and combined with a stochastic methodology linear inverse
    problems can be solved even if the data is too large to fit into memory.

    We use dask to batch the data, therefore we must provide `.shape, .ndim, .dtype and support numpy-style slicing <https://docs.dask.org/en/stable/generated/dask.array.from_array.html>`_.
    """

    def __init__(self, path: pyct.Path, gpu: bool):
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
    def _load(self, path: pyct.Path) -> tuple[cabc.Sequence, pyct.Shape, pyct.Integer, pyct.DType]:
        r"""
        Parameters
        ----------
        path : pyct.Path
            Filepath to load

        Returns
        -------
        data : cabc.Sequence
            object that has a :py:meth:`__getitem__`
        _shape : pyct.Shape
            shape of stored dataset
        _ndim : pyct.Integer
            number of dimensions of data
        _dtype : pyct.DType

        Note
        -------
        shape should abide by pycsou rules for NDArray shape
        """
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


Loader = typ.Union[cabc.Sequence, Dataset]  # TODO after refactor point Dataset to correct spot


class NpzDataset(Dataset):
    r"""
    NPZ file format loader.

    Uses mmap_mode which allows us to access small segments of large files on disk, without reading the entire file
    into memory.
    """

    def __init__(self, path: pyct.Path, gpu: bool = False):
        super().__init__(path, gpu)

    def _load(self, path):
        if self.gpu:
            data = cp.load(path, mmap_mode="c")
        else:
            data = np.load(path, mmap_mode="c")
        return data, data.shape, data.ndim, data.dtype


class DataLoader(cabc.Sequence):
    r"""
    Abstract class DataLoader. Given a dataset, gives batches of the data.

    All subclasses should overwrite:
    :py:meth:`~pycsou.opt.stochastic.DataLoader.__getitem__` which takes an index and returns a batch of the dataset and potentially other meta-information about that batch of data.
    :py:meth:`~pycsou.opt.stochastic.DataLoader.len` which outputs the number of batches in the dataset.
    :py:meth:`~pycsou.opt.stochastic.DataLoader.communicate` which communicates meta-information about the dataset and batching scheme in :py:class:`Batch`. This is information that needs to be communicated only once before the optimization process begins.
    so the user doesn't have to provide this information twice.

    See Also
    -------
    :py:class:`~pycsou.opt.stochastic.Batch`
    """

    def __init__(self, load: Loader, arg_shape: pyct.Shape):
        self.load = load
        self.arg_shape = arg_shape

    @abc.abstractmethod
    def __getitem__(self, index: pyct.Integer):
        r"""
        Should provide data itself and any information that is necessary on a per-batch basis.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        r"""
        Number of batches in the dataset
        """
        raise NotImplementedError

    @abc.abstractmethod
    def communicate(self):
        r"""
        Communicate meta information about the dataset and batching.
        """
        raise NotImplementedError


class ChunkDataloader(DataLoader):
    r"""
    Batches data following a chunking strategy using dask in the internals.
    Should be used in conjunction with `~pycsou.opt.stochastic.ChunkOp` as the DataLoading strategy must work in
    conjunction with the Operator batching strategy.

    We use dask under the hood to create `chunks <https://docs.dask.org/en/stable/array-chunks.html>_`.

    Note
    ----------
    Not all batches are guaranteed to be the same size. If the data shape is not divisible by chunks then dask
    will handle the edges by creating smaller chunks. This can cause potential issues surrounding the step size hyperparameters.
    We advise to make your data divisible into evenly spaced chunks. This could mean potentially padding or cutting your data.

    See Also
    ----------
    :py:class:`~pycsou.opt.stochastic.ChunkOp`
    """

    # TODO what if load is already a dask array -> redirect or if statement
    def __init__(self, load: Loader, arg_shape: pyct.Shape, chunks: pyct.Shape):
        r"""
        Parameters
        ----------
        load : Loader
            object to query data from
        arg_shape : pyct.Shape
            This should represent the actual shape of your data. This should not abide by rules for pycsou ndarrays.
            For example a 2d image would be represented by (H,W).
        chunks
            the size of one chunk of data.

        Note
        ----------
        The number of dimensions of chunks should match number of dimensions of arg_shape.
        """
        super().__init__(load, arg_shape)

        self.data_ndim = len(self.arg_shape)
        self.chunks = chunks
        if self.data_ndim != len(self.chunks):
            msg = f"chunk dim of {len(self.chunks)} doesn't match data dim of {self.data_ndim}"
            raise ValueError(msg)

        self.global_shape = self.load.shape
        self.stack_shape = self.global_shape[:-1]

        # TODO think we can just say da.array() but need to look into that
        self.data = (
            da.from_array(self.load)
            .reshape(*self.stack_shape, *self.arg_shape)
            .rechunk(chunks=(*self.stack_shape, *self.chunks))
        )

        self.blocks = self.data.chunks
        self.block_dim = [len(c) for c in self.blocks]
        self.indices = (
            da.arange(self.load.size)
            .reshape(*self.stack_shape, *self.arg_shape)
            .rechunk(chunks=(*self.stack_shape, *self.chunks))
        )

    # TODO we don't actually use the indices rn. Also think indices may be weird especially for stacking dimenions.
    # need to think further about the indices.
    def __getitem__(self, b_index: pyct.Integer) -> tuple[pyct.NDArray, pyct.NDArray]:
        r"""
        Parameters
        ----------
        b_index : pyct.Integer
            index for 1 batch of data

        Returns
        -------
        batch : pyct.NDArray
            flattened batch of data
        ind : pyct.NDArray
            indices corresponding to the data in the array
        """
        i = np.unravel_index(b_index, self.block_dim)

        batch = self.data.blocks[i]
        ind = self.indices.blocks[i]

        return batch.compute().reshape(*self.stack_shape, -1), ind.compute().flatten()

    def __len__(self):
        return self.data.npartitions

    def communicate(self):
        return {"blocks": self.blocks, "block_dim": self.block_dim, "stack_shape": self.stack_shape}


# TODO think about name at some point
class BatchOp(pyco.LinOp):
    r"""
    Abstract class for BatchOp, ie Batched Operators. Creates an operator that mimics a global pyco.LinOp but
    operates on a batch of the data.

    Works in conjunction with a ':py:class:`~pycsou.opt.stochastic.DataLoader`' class. The Dataloader provides the data,
    and this class provides the operator.

    All subclasses should overwrite:
    :py:meth:`~pycsou.opt.stochastic.BatchOp.startup`
    :py:meth:`~pycsou.opt.stochastic.BatchOp.__getitem__`

    Along with the subclasses for pyco.LinOp:
    :py:meth:`~pycsou.abc.operator.LinOp.apply`
    :py:meth:`~pycsou.abc.operator.LinOp.adjoint`

    See Also
    -------
    :py:class:`~pycsou.opt.stochastic.DataLoader`
    :py:class:`~pycsou.opt.stochastic.ChunkOp`
    """

    def __init__(self, op: pyco.LinOp):
        self.op = op
        self.arg_shape = self.op.arg_shape
        super().__init__(shape=self.op.shape)

    @abc.abstractmethod
    def startup(self, *args, **kwargs) -> None:
        r"""
        Startup is a secondary constructor that instantiates attributes from information provided in the DataLoader.
        :py:meth:`~pycsou.opt.stochastic.DataLoader.communicate` is called and passed to
        :py:meth:`~pycsou.opt.stochastic.BatchOp.startup`.

        This process is managed through :py:meth:`~pycsou.opt.stochastic.Batch._communicate`. This is to avoid the user
        having to duplicate passing in the same information twice.

        Parameters
        ----------
        args
        kwargs
            attributes that comes from a :py:meth:`~pycsou.opt.stochastic.DataLoader.communicate` method.

        See Also
        -------
        :py:class:`~pycsou.opt.stochastic.DataLoader`
        :py:class:`~pycsou.opt.stochastic.Batch`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, item: pyct.Integer) -> pyco.LinOp:
        r"""
        Return a new BatchOp that is prepared to operate on a particular subset of data.
        """
        raise NotImplementedError


class ChunkOp(BatchOp):
    r"""
    Batches certain kinds of Linear Operators to operate on Chunks of data efficiently.

    The Linear Operators that can be used with ChunkOp are:
        TODO
        - list of LinOps

    See Also
    -------
    TODO
    Convolve
    MaxPool
    AvgPool
    """

    def __init__(
        self, op: pyco.LinOp, depth: dict[int:int] = None, boundary: dict[int : typ.Union[str, pyct.Real]] = None
    ):
        r"""
        The parameters depth and boundary are passed along to
        `dask.map_overlap<https://docs.dask.org/en/stable/array-overlap.html>_`

        Parameters
        ----------
        op: pyco.LinOp
            operator to be batched. Only certain types of operators can be batched with a Chunking strategy.
        depth: dict[int: int]
            How much chunking overlap there should be. See example1_
            Current supported structure for depth is as a dict with the axis as the key, and number of values to overlap as value.
        boundary: dict[int: typ.Union[str, pyct.Real]]
            The boundary condition for overlap at the edges of the data. See example2_
            Current supported structure for boundary is the same as depth, with the axis as the key and as the value:

            - `periodic` - wrap borders around to the other side
            - `reflect` - reflect each border outwards
            - 'nearest' - take closest value and pad border
            - `any-constant` - pad the border with this value

        .. _example1:
            For a convolution this will depend on your kernel size.
            If you have a 5x5 kernel, you will want on overlap of 2 for each axis - {0: 2, 1: 2}
            If you have a 7x7x7 kernel, you will want an overlap of 3 for each axis - {0: 3, 1: 3, 2: 3}
            If you have a 5x7 kernel, you will want a varying level of overlap - {0: 2, 1: 3}

        .. _example2:
            Only if you have depth, will you need to define boundary. You can define a different boundary for each
            axis.
            If you have a depth of {0: 2, 1: 2} you can define a boundary of:

                - {0: 'reflect', 1: 'periodic'}
                - {0: 'reflect', 1: 'reflect'}
                - {0: 0, 1: np.nan}
        """
        # TODO create Stochastic operator subclass to check against...
        # if not isinstance(op, StochasticOp):
        #     raise ValueError("Operator must be a StochasticOp operator.")
        super().__init__(op=op)
        self.data_ndim = len(self.arg_shape)
        self.depth = depth
        self.boundary = boundary

        # use global op lipschitz constants
        self._diff_lipschitz = self.op.diff_lipschitz()
        self._lipschitz = self.op.lipschitz()

        # initialized in startup from the dataloader
        self.blocks = None
        self.block_dim = None
        self.slices = None
        self.stack_shape = None

    @staticmethod
    def _coerce_condition(condition: dict, ndim: pyct.Integer, num_stack: pyct.Integer, default: pyct.Real = 0) -> dict:
        r"""
        Add a number of default values to a dictionary equivalent to num_stack, and re-order the existing values to be
        after the default values. The spacing between values should stay equivalent.

        Parameters
        ----------
        condition: dict
            dictionary containing existing keys and values that needs to be re-ordered
        ndim:
            total number of keys in output dict
        num_stack: pyct.Integer
            number of keys to have default value in output dict
        default: pyct.Real
            the default value

        Returns
        -------
            dict
                dict with default values added to first num_stack keys, and the other keys re-ordered.
        """

        return dict(
            ((i, default) if i < num_stack else (i, condition.get(i - num_stack, default)) for i in range(ndim))
        )

    def startup(self, blocks, block_dim, stack_shape, *args, **kwargs):
        self.blocks = blocks
        self.block_dim = block_dim
        self.slices = da.core.slices_from_chunks(self.blocks)
        self.stack_shape = stack_shape
        stack_ndim = len(self.stack_shape)

        # TODO check what happens if None is put...is it what we want to happen?
        self.depth = self._coerce_condition(self.depth, self.data_ndim + stack_ndim, stack_ndim, default=0)
        self.boundary = self._coerce_condition(self.boundary, self.data_ndim + stack_ndim, stack_ndim, default=None)
        for i, (k, v) in enumerate(self.depth.items()):
            if v > 0:
                if not all([b >= v for b in self.blocks[i]]):
                    msg = (
                        f"{self.blocks[i]} has a chunk/chunks smaller than depth: {v}. Provide a larger chunking size."
                    )
                    raise ValueError(msg)

    def __getitem__(self, b_index):
        """ """
        self.b_index = b_index
        self.index = np.unravel_index(self.b_index, self.block_dim)
        self.batch_slice = self.slices[self.b_index]
        self.overlap_shape = [s.stop - s.start + 2 * self.depth[i] for i, s in enumerate(self.batch_slice)][
            -self.data_ndim :
        ]
        self.batch_shape = [s.stop - s.start for s in self.batch_slice[-self.data_ndim :]]
        self.op.arg_shape = self.overlap_shape
        self._shape = (np.prod(self.batch_shape), np.prod(self.batch_shape))
        return self

    def apply(self, arr):
        xp = pycu.get_array_module(arr)
        input_shape = arr.shape

        if xp != da:
            arr = da.from_array(arr)

        arr = arr.reshape(*input_shape[:-1], *self.arg_shape).rechunk(self.blocks)

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
            out = da.from_array(xp.zeros((*input_shape[:-1], *self.arg_shape)), chunks=self.blocks)
        else:
            out = da.zeros((*input_shape[:-1], *self.arg_shape), chunks=self.blocks)

        out[self.batch_slice] = arr.reshape((*input_shape[:-1], *self.batch_shape))

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
            boundary=0,
            # allow_rechunk=False,- released 2022.6.1 DASK
            meta=xp.array(()),
            dtype=arr.dtype,
        )

        if xp != da:
            return out.reshape(*input_shape[:-1], -1).compute()
        else:
            return out.reshape(*input_shape[:-1], -1)


def neighbors_map_overlap(
    x: pyct.NDArray, op: pyco.LinOp, ind: tuple[int], overlap: bool, stack_dims: pyct.Shape, block_info: dict = None
) -> pyct.NDArray:
    r"""
    Block function that works with `da.map_overlap`.
    Applies pycsou LinOp operator op.adjoint over 1 block and conditionally over its direct neighbors, leaving the
    other blocks untouched.

    **Remark 1:**

    Based on `Map Blocks`_ documentation.

    .. _Map Blocks: https://docs.dask.org/en/stable/generated/dask.array.map_blocks.html

    **Remark 2:**

     op must be a LinOp with attribute arg_shape and an adjoint method.
     The arg_shape is dynamically adjusted to allow for different batch_sizes.

     **Remark 3:**

    TODO - do we want this?
     If there is any overlap at all, every neighbor in every dimension will be calculated, even if there is not an
     overlap in that specific dimension.

    Parameters
    ----------
    x : da.array
        the function `da.map_overlap()` will provide chunks of data. x will be a chunk of data.
    op : pyco.LinOp
        operator with which to call adjoint over the data
    ind : tuple
        indices of block to apply op to
    overlap : bool
        whether to apply op to the neighbors of ind
    stack_dims : pyct.Shape
        stacking dimensions of x
    block_info : dict
        special keyword argument provided by `da.map_overlap()` to get information about where in array we are

    Returns
    -------
    x : da.array
        the chunk of data with op conditionally applied to it

    """
    block_id = block_info[0]["chunk-location"]
    array_location = block_info[0]["array-location"]

    if block_id:
        # subset to only be dimensions of interest (not stacking dimensions)
        block_id = block_id[len(stack_dims) :]
        array_location = array_location[len(stack_dims) :]
        ind = ind[len(stack_dims) :]

        # if dimensions of interest within +/-1, apply function
        if all([i - int(overlap) <= j <= i + int(overlap) for i, j in zip(ind, block_id)]):
            shape_overload = [s[1] - s[0] for s in array_location]
            op.arg_shape = tuple(shape_overload)
            return op.adjoint(x.reshape(*stack_dims, -1)).reshape(x.shape)
    return x


class Batch:
    r"""
    Batch is an orchestrating class.
    It controls batching through a batching strategy, currently random or in-order batching.
    It instantiates the given Dataset and a BatchOp in :py:meth:`~pycsou.opt.stochastic.Batch._communicate` and is in charge of generating a batch of data and the
    corresponding batched operator.
    It keeps track of batching iterations and epochs.

    See Also
    -------
    :py:class:`~pycsou.opt.stochastic.DataLoader`
    :py:class:`~pycsou.opt.stochastic.BatchOp`
    :py:class:`~pycsou.opt.stochastic.Stochastic`
    """

    def __init__(self, dataloader: DataLoader, batch_op: BatchOp, shuffle: bool = False, seed: pyct.Integer = None):
        r"""
        Parameters
        ----------
        dataloader: DataLoader
            dataloader to query for batches of data
        batch_op: BatchOp
            batched operator to query for an operator that operates on a batch of data
        shuffle: bool
            whether to ask for batches in order or randomly
        seed: pyct.Integer
            if shuffle=True, which seed to use for randomness
        """
        self.dataloader = dataloader
        self.batch_op = batch_op
        self._communicate()

        self._epochs = 0
        self.counter = 0
        self.batch_counter = 0

        self.num_batches = len(self.dataloader)
        self.shuffle = shuffle
        self.seed = seed
        if self.shuffle:
            self.shuffled_batch = self._shuffle()

    @property
    def epochs(self) -> int:
        r"""
        Each epoch denotes 1 pass through all batches of the data.

        Returns
        -------
        int
            number of epochs
        """
        return self._epochs

    def _communicate(self):
        r"""
        Communicates meta-information between the dataloader and the batch_op that is necessary upon intialization
        of the optimization problem.

        This is necessary so the user doesn't have to include information twice in the __init__ of both dataloader
        and batch_op. Also, if there is important meta-information computed in dataloader it only needs to be computed
        once and then shared.
        """
        kwargs = self.dataloader.communicate()
        self.batch_op.startup(**kwargs)

    def _bookkeeping(self):
        r"""
        Keeps track of which batch to query and when to start a new epoch.
        """
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

    def batches(self) -> typ.Tuple[pyct.NDArray, pyco.LinOp, pyct.NDArray]:
        r"""
        Generator which queries the dataloader and batch_op and yields these.

        Returns
        -------
        typ.Tuple[pyct.NDArray, pyco.LinOp, pyct.NDArray]
            data, operator, indices of data
        """
        y, ind = self.dataloader[self.batch_counter]
        op = self.batch_op[self.batch_counter]
        self._bookkeeping()
        yield y, op, ind


class GradStrategy:
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

    Allows the gradient to flow through unchanged as :py:class:`pycsou.opt.stochastic.Stochastic` takes care of computing
    the gradient stochastically.

    It is the base strategy for :py:class:`pycsou.opt.stochastic.Stochastic` if None is given.
    """

    def apply(self, grad: pyct.NDArray) -> pyct.NDArray:
        return grad


# TODO - refactor this out to another branch
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
    """

    def __init__(self, f: pyco.DiffFunc, batch: Batch, strategy: GradStrategy = None):
        r"""
        Parameters
        ----------
        f : pyco.DiffFunc
            data fidelity term
        batch : Batch
            generator to query and get batched data and a batched operator
        strategy : GradStrategy = None
            which gradient strategy to use. Default is :py:func:`pycsou.opt.stochastic.SGD`
        """
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
