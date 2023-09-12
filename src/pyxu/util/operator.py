import collections.abc as cabc
import functools
import inspect

import dask
import dask.graph_manipulation as dgm

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.util.array_module as pxa
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
    method: str = "scan",
    codim: pxt.Integer = None,
) -> cabc.Callable:
    """
    Decorator to auto-vectorize an array function to abide by :py:class:`~pyxu.abc.Property` API rules.

    Parameters
    ----------
    i: VarName
        Function parameter to vectorize. This variable must hold an object with a NumPy API.
    method: str
        Vectorization strategy:

        * `scan` computes outputs using a for-loop.
        * `parallel` passes inputs to DASK and evaluates them in parallel.
        * `scan_dask` passes inputs to DASK but evaluates inputs in sequence.
          This is useful if the function being vectorized has a shared resource, i.e. is not thread-safe.  It
          effectively gives a DASK-unaware function the ability to work with DASK inputs.
    codim: Integer
        Size of the function's core dimension output.

        This parameter is only required in "parallel" and "scan_dask" modes.

    Example
    -------
    .. code-block:: python3

       import pyxu.util as pxu

       @pxu.vectorize('x')
       def f(x):
           return x.sum(keepdims=True)

       x = np.arange(10).reshape((2, 5))
       f(x[0]), f(x[1])  #  [10], [35]
       f(x)              #  [10, 35] -> would have retured [45] if not decorated.
    """
    method = method.strip().lower()
    assert method in ("scan", "scan_dask", "parallel"), f"Unknown vectorization method '{method}'."
    if using_dask := (method in ("scan_dask", "parallel")):
        assert isinstance(codim, pxt.Integer), f"Parameter[codim] must be specified for DASK-backed '{method}'."

    def decorator(func: cabc.Callable) -> cabc.Callable:
        sig = inspect.Signature.from_callable(func)
        if i not in sig.parameters:
            error_msg = f"Parameter[{i}] not part of {func.__qualname__}() parameter list."
            raise ValueError(error_msg)

        @functools.wraps(func)
        def wrapper(*ARGS, **KWARGS):
            func_args = pxm.parse_params(func, *ARGS, **KWARGS)

            x = func_args.pop(i)
            *sh, dim = x.shape
            x = x.reshape((-1, dim))
            N, xp = len(x), pxa.get_array_module(x)

            if using_dask:
                f = lambda _: func(**{i: _, **func_args})
                blks = _dask_zip(
                    func=(f,) * N,
                    data=x,
                    out_shape=[(codim,)] * N,  # can't infer codim -> user-specified
                    out_dtype=(x.dtype,) * N,
                    parallel=method == "parallel",
                )
                y = xp.stack(blks, axis=0)  # (N, codim)
            else:  # method = "scan"
                # infer output dimensions + allocate
                func_args[i] = x[0]
                y0 = func(**func_args)
                y = xp.zeros((N, y0.size), dtype=y0.dtype)  # (N, codim)

                y[0] = y0
                for k in range(1, N):
                    func_args[i] = x[k]
                    y[k] = func(**func_args)

            y = y.reshape(*sh, -1)
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
