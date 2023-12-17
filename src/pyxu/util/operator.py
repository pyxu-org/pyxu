import collections.abc as cabc
import concurrent.futures as cf
import copy
import functools
import inspect
import itertools

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.util.misc as pxm

__all__ = [
    "as_canonical_axes",
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
    assert all(isinstance(_x, pxt.Integer) for _x in x)

    shape = tuple(map(int, x))
    return shape


def as_canonical_axes(
    axes: pxt.NDArrayAxis,
    rank: pxt.Integer,
) -> pxt.NDArrayAxis:
    """
    Transform NDarray axes into tuple-form with positive indices.

    Parameters
    ----------
    rank: Integer
        Rank of the NDArray. (Required to make all entries positive.)
    """
    assert rank >= 1

    axes = as_canonical_shape(axes)
    assert all(-rank <= ax < rank for ax in axes)  # all axes in valid range
    axes = tuple((ax + rank) % rank for ax in axes)  # get rid of negative axes
    return axes


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
