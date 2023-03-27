import collections.abc as cabc
import functools
import inspect

import dask
import dask.graph_manipulation as dgm
import numpy as np

import pycsou.util.array_module as pycua
import pycsou.util.deps as pycd
import pycsou.util.inspect as pycui
import pycsou.util.ptype as pyct

__all__ = [
    "infer_composition_shape",
    "infer_sum_shape",
    "vectorize",
]


def infer_sum_shape(sh1: pyct.OpShape, sh2: pyct.OpShape) -> pyct.OpShape:
    A, B, C, D = *sh1, *sh2
    if None in (A, C):
        raise ValueError("Addition of codomain-dimension-agnostic operators is not supported.")
    try:
        domain_None = (B is None, D is None)
        if all(domain_None):
            return np.broadcast_shapes((A,), (C,)) + (None,)
        elif any(domain_None):
            fill = lambda _: [1 if (k is None) else k for k in _]
            return np.broadcast_shapes(fill(sh1), fill(sh2))
        elif domain_match := (B == D):
            return np.broadcast_shapes((A,), (C,)) + (B,)
        else:
            raise
    except:
        raise ValueError(f"Addition of {sh1} and {sh2} operators forbidden.")


def infer_composition_shape(sh1: pyct.OpShape, sh2: pyct.OpShape) -> pyct.OpShape:
    A, B, C, D = *sh1, *sh2
    if None in (A, C):
        raise ValueError("Composition of codomain-dimension-agnostic operators is not supported.")
    elif (B == C) or (B is None):
        return (A, D)
    else:
        raise ValueError(f"Composition of {sh1} and {sh2} operators forbidden.")


def vectorize(i: pyct.VarName) -> cabc.Callable:
    """
    Decorator to auto-vectorize an array function to abide by
    :py:class:`~pycsou.abc.operator.Property` API rules.

    Parameters
    ----------
    i: VarName
        Function parameter to vectorize. This variable must hold an object with a NumPy API.

    Example
    -------

    >>> import pycsou.util as pycu
    >>> @pycu.vectorize('x')
    ... def f(x):
    ...     return x.sum(keepdims=True)
    ...
    >>> x = np.arange(10).reshape((2, 5))
    >>> f(x[0]), f(x[1])  #  [10], [35]
    >>> f(x)              #  [10, 35] -> would have retured [45] if not decorated.

    Notes
    -----
    See :ref:`developer-notes`
    """

    def decorator(func: cabc.Callable) -> cabc.Callable:
        sig = inspect.Signature.from_callable(func)
        if i not in sig.parameters:
            error_msg = f"Parameter[{i}] not part of {func.__qualname__}() parameter list."
            raise ValueError(error_msg)

        @functools.wraps(func)
        def wrapper(*ARGS, **KWARGS):
            func_args = pycui.parse_params(func, *ARGS, **KWARGS)

            x = func_args[i]
            if is_1d := x.ndim == 1:
                x = x.reshape((1, x.size))
            sh_x = x.shape  # (..., N)
            sh_xf = (np.prod(sh_x[:-1]), sh_x[-1])  # (..., N) -> (M, N)
            x = x.reshape(sh_xf)

            # infer output dimensions + allocate
            func_args[i] = x[0]
            y0 = func(**func_args)
            xp = pycua.get_array_module(y0)
            y = xp.zeros((*sh_xf[:-1], y0.size), dtype=y0.dtype)

            y[0] = y0
            for k in range(1, sh_xf[0]):
                func_args[i] = x[k]
                y[k] = func(**func_args)
            y = y.reshape((*sh_x[:-1], y.shape[-1]))
            if is_1d:
                y = y.reshape(-1)

            return y

        return wrapper

    return decorator


def _dask_zip(
    func: list[cabc.Callable],
    data: list[pyct.NDArray],
    out_shape: list[pyct.NDArrayShape],
    out_dtype: list[pyct.DType],
    parallel: bool,
) -> list[pyct.NDArray]:
    # (This is Low-Level function.)
    #
    # Computes the equivalent of ``out = [f(x) for (f, x) in zip(func, data)]``, with the following semantics:
    #
    # * If `data` contains only NUMPY/CUPY arrays, then ``out`` is computed as above.
    # * If `data` contains only DASK arrays, then entries of ``out`` are computed:
    #
    #   * if `parallel` enabled  -> dask-delay each `func`, then evaluate in parallel.
    #   * if `parallel` disabled -> dask-delay each `func`, then evaluate in sequence.
    #     (This is useful if `func`s share a common resource, thus not thread-safe.)
    #
    # For Dask-array inputs, this amounts to creating a task graph with virtual dependencies
    # between successive `func` calls. In other words, the task graph looks like:
    #
    #        _dask_zip(func, data, parallel=True) -> out
    #
    #                    +----+              +----+
    #          data[0]-->|func|-->blk[0]-+-->|list|-->out
    #             .      +----+          |   +----+
    #             .                      |
    #             .      +----+          |
    #          data[n]-->|func|-->blk[n]-+
    #                    +----+
    #
    # ==========================================================================================================
    #        _dask_zip(func, data, parallel=False) -> out
    #                                                                                             +----+
    #                              +-------------------+----------------+--------------------+----+list|-->out
    #                              |                   |                |                    |    +----+
    #                              |                   |                |                    |
    #                    +----+    |        +----+     |      +---+     |         +----+     |
    #          data[0]-->|func|-->out[0]-+->|func|-->out[1]-->|...|-->out[n-1]-+->|func|-->out[n]
    #                    +----+          |  +----+            +---+            |  +----+
    #                                    |                                     |
    #          data[1]-------------------+                                     |
    #             .                                                            |
    #             .                                                            |
    #             .                                                            |
    #          data[n]---------------------------------------------------------+
    #
    #
    # Parameters
    # ----------
    # func: list(callable)
    #     Functions to apply to each element of `data`.
    #
    #     Function signatures are ``Callable[[pyct.NDArray], pyct.NDArray]``.
    # data: list[pyct.NDArray]
    #     (N_data,) arrays to act on.
    # out_shape: list[pyct.NDArrayShape]
    #     Shapes of ``func[i](data[i])``.
    #
    #     This parameter is only used if inputs are DASK arrays.
    #     Its goal is to transform Delayed objects back to array form.
    # out_dtype: list[pyct.DType]
    #     Dtypes of ``func[i](data[i])``.
    #
    #     This parameter is only used if inputs are DASK arrays.
    #     Its goal is to transform Delayed objects back to array form.
    #
    # Returns
    # -------
    # out: list[pyct.NDArray]
    #     (N_data,) objects acted upon.
    #
    #     Outputs have the same backend/dtype as inputs.
    assert all(len(_) == len(func) for _ in [data, out_shape, out_dtype])

    NDI = pycd.NDArrayInfo
    dask_input = lambda obj: NDI.from_obj(obj) == NDI.DASK
    if all(map(dask_input, data)):
        xp = NDI.DASK.module()

        out = []
        for i in range(len(data)):
            f = dask.delayed(func[i], pure=True)
            if parallel:
                _func = f
            else:
                _func = dgm.bind(
                    children=f,
                    parents=out[i - 1] if (i > 0) else [],
                )
            blk = xp.from_delayed(
                _func(data[i]),
                shape=out_shape[i],
                dtype=out_dtype[i],
            )
            out.append(blk)
    else:
        out = [_func(arr) for (_func, arr) in zip(func, data)]
    return out
