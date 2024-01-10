import collections.abc as cabc
import concurrent.futures as cf
import copy
import functools
import inspect
import itertools

import dask
import dask.graph_manipulation as dgm

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.util.misc as pxm

__all__ = [
    "as_canonical_shape",
    "vectorize",
]


def as_canonical_shape(x: pxt.NDArrayShape) -> pxt.NDArrayShape:
    """
    Transform a lone integer into a valid tuple-based shape specifier.
    """
    if isinstance(x, cabc.Iterable):
        x = tuple(x)
    else:
        x = (x,)
    sh = tuple(map(int, x))
    return sh


def vectorize(
    i: pxt.VarName,
    dim_shape: pxt.NDArrayShape,
    codim_shape: pxt.NDArrayShape,
) -> cabc.Callable:
    r"""
    Decorator to auto-vectorize a function :math:`\mathbf{f}: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to
    \mathbb{R}^{N_{1} \times\cdots\times N_{K}}` to accept stacking dimensions.

    Parameters
    ----------
    i: VarName
        Function/method parameter to vectorize. This variable must hold an object with a NumPy API.
    dim_shape: NDArrayShape
        (M1,...,MD) shape of operator's domain.
    codim_shape: NDArrayShape
        (N1,...,NK) shape of operator's co-domain.

    Returns
    -------
    g: ~collections.abc.Callable
        Function/Method with signature ``(..., M1,...,MD) -> (..., N1,...,NK)`` in parameter `i`.

    Example
    -------
    .. code-block:: python3

       import pyxu.util as pxu

       def f(x):
           return x.sum(keepdims=True)

       N = 5
       g = pxu.vectorize("x", N, 1)(f)

       x = np.arange(2*N).reshape((2, N))
       g(x[0]), g(x[1])  #  [10], [35]
       g(x)              #  [[10],
                         #   [35]]

    Notes
    -----
    * :py:func:`~pyxu.util.vectorize` assumes the function being vectorized is **thread-safe** and can be evaluated in
      parallel. Using it on thread-unsafe code may lead to incorrect outputs.
    * As predicted by Pyxu's :py:class:`~pyxu.abc.Operator` API:

      - The dtype of the vectorized function is assumed to match `x.dtype`.
      - The array backend of the vectorized function is assumed to match that of `x`.
    """
    N = pxd.NDArrayInfo  # short-hand
    dim_shape = as_canonical_shape(dim_shape)
    dim_rank = len(dim_shape)
    codim_shape = as_canonical_shape(codim_shape)

    def decorator(func: cabc.Callable) -> cabc.Callable:
        sig = inspect.Signature.from_callable(func)
        if i not in sig.parameters:
            error_msg = f"Parameter[{i}] not part of {func.__qualname__}() parameter list."
            raise ValueError(error_msg)

        @functools.wraps(func)
        def wrapper(*ARGS, **KWARGS):
            func_args = pxm.parse_params(func, *ARGS, **KWARGS)

            x = func_args.pop(i)
            ndi = N.from_obj(x)
            xp = ndi.module()

            sh_stack = x.shape[:-dim_rank]
            if ndi in [N.NUMPY, N.CUPY]:
                task_kwargs = []
                for idx in itertools.product(*map(range, sh_stack)):
                    kwargs = copy.deepcopy(func_args)
                    kwargs[i] = x[idx]
                    task_kwargs.append(kwargs)

                with cf.ThreadPoolExecutor() as executor:
                    res = executor.map(lambda _: func(**_), task_kwargs)
                y = xp.stack(list(res), axis=0).reshape((*sh_stack, *codim_shape))
            elif ndi == N.DASK:
                # Find out codim chunk structure ...
                idx = (0,) * len(sh_stack)
                func_args[i] = x[idx]
                codim_chunks = func(**func_args).chunks  # no compute; only extract chunk info

                # ... then process all inputs.
                y = xp.zeros(
                    (*sh_stack, *codim_shape),
                    dtype=x.dtype,
                    chunks=x.chunks[:-dim_rank] + codim_chunks,
                )
                for idx in itertools.product(*map(range, sh_stack)):
                    func_args[i] = x[idx]
                    y[idx] = func(**func_args)
            else:
                # Define custom behavior
                raise ValueError("Unknown NDArray category.")

            return y

        return wrapper

    return decorator


def _dask_zip(
    func: list[cabc.Callable],
    data: list[pxt.NDArray],
    out_shape: list[pxt.NDArrayShape],
    out_dtype: list[pxt.DType],
    parallel: bool,
) -> list[pxt.NDArray]:
    """
    Map functions in parallel via Dask.

    Computes the equivalent of ``out = [f(x) for (f, x) in zip(func, data)]``, with the following semantics:

    * If `data` contains only NUMPY/CUPY arrays, then ``out`` is computed as above.
    * If `data` contains only DASK arrays, then entries of ``out`` are computed:

      * if `parallel` enabled  -> dask-delay each `func`, then evaluate in parallel.
      * if `parallel` disabled -> dask-delay each `func`, then evaluate in sequence.
        (This is useful if `func` s share a common resource, thus not thread-safe.)

    For Dask-array inputs, this amounts to creating a task graph with virtual dependencies between successive `func`
    calls. In other words, the task graph looks like::

        _dask_zip(func, data, parallel=True) -> out

                      +----+              +----+
            data[0]-->|func|-->blk[0]-+-->|list|-->out
                .     +----+          |   +----+
                .                     |
                .     +----+          |
            data[n]-->|func|-->blk[n]-+
                      +----+

        ==========================================================================================================
        _dask_zip(func, data, parallel=False) -> out
                                                                                               +----+
                                +-------------------+----------------+--------------------+----+list|-->out
                                |                   |                |                    |    +----+
                                |                   |                |                    |
                      +----+    |        +----+     |      +---+     |         +----+     |
            data[0]-->|func|-->out[0]-+->|func|-->out[1]-->|...|-->out[n-1]-+->|func|-->out[n]
                      +----+          |  +----+            +---+            |  +----+
                                      |                                     |
            data[1]-------------------+                                     |
                .                                                           |
                .                                                           |
                .                                                           |
            data[n]---------------------------------------------------------+

    Parameters
    ----------
    func: list
        Functions to apply to each element of `data`.

        Function signatures are ``Callable[[NDArray], NDArray]``.
    data: list
        (N_data,) arrays to act on.
    out_shape: list
        Shapes of ``func[i](data[i])``.

        This parameter is only used if inputs are DASK arrays.  Its goal is to transform
        :py:class:`~dask.delayed.Delayed` objects back to array form.
    out_dtype: list
        Dtypes of ``func[i](data[i])``.

        This parameter is only used if inputs are DASK arrays.  Its goal is to transform
        :py:class:`~dask.delayed.Delayed` objects back to array form.

    Returns
    -------
    out: list
        (N_data,) objects acted upon.

        Outputs have the same backend/dtype as inputs, or as specified by `out_[shape,dtype]`.
    """
    assert all(len(_) == len(func) for _ in [data, out_shape, out_dtype])

    NDI = pxd.NDArrayInfo
    dask_input = lambda obj: NDI.from_obj(obj) == NDI.DASK
    if all(map(dask_input, data)):
        out = []
        for i in range(len(data)):
            if parallel:
                # If parallel=True, the user guarantees that functions CAN be executed in parallel,
                # i.e. no side-effects induced by calling the func[k] if arbitrary order.
                # The functions are hence PURE.
                _func = dask.delayed(func[i], pure=True)
            else:
                # If parallel=False, side-effects MAY influence func[k] outputs.
                # The functions are hence IMPURE.
                _func = dgm.bind(
                    children=dask.delayed(func[i], pure=False),
                    parents=out[i - 1] if (i > 0) else [],
                )
            blk = _array_ize(_func(data[i]), out_shape[i], out_dtype[i])
            out.append(blk)
    else:
        out = [_func(arr) for (_func, arr) in zip(func, data)]
    return out


def _array_ize(
    data,
    shape: pxt.NDArrayShape,
    dtype: pxt.DType,
) -> pxt.NDArray:
    """
    Transform a Dask-delayed object into its Dask-array counterpart.

    This function is a no-op if `data` is not a :py:class:`~dask.delayed.Delayed` object.

    Parameters
    ----------
    data: NDArray, :py:class:`~dask.delayed.Delayed`
        Object to act upon.
    shape: NDArrayShape
        Shape of `data`.

        This parameter is only used if `data` is a :py:class:`~dask.delayed.Delayed` object.  Its goal is to transform
        the former back to array form.
    dtype: DType
        Dtype of `data`.

        This parameter is only used if `data` is a :py:class:`~dask.delayed.Delayed` object.  Its goal is to transform
        the former back to array form.

    Returns
    -------
    arr: NDArray
        Dask-backed NDArray if `data` was a :py:class:`~dask.delayed.Delayed` object; no-op otherwise.
    """
    from dask.delayed import Delayed

    if isinstance(data, Delayed):
        xp = pxd.NDArrayInfo.DASK.module()
        arr = xp.from_delayed(
            data,
            shape=shape,
            dtype=dtype,
        )
    else:
        arr = data
    return arr
