import collections.abc as cabc
import functools

import dask

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.util.misc as pxm

__all__ = [
    "compute",
    "get_array_module",
    "redirect",
    "to_NUMPY",
]


def get_array_module(x, fallback: pxt.ArrayModule = None) -> pxt.ArrayModule:
    """
    Get the array namespace corresponding to a given object.

    Parameters
    ----------
    x: object
        Any object compatible with the interface of NumPy arrays.
    fallback: ArrayModule
        Fallback module if `x` is not a NumPy-like array.  Default behaviour: raise error if fallback used.

    Returns
    -------
    namespace: ArrayModule
        The namespace to use to manipulate `x`, or `fallback` if provided.
    """

    def infer_api(y):
        try:
            return pxd.NDArrayInfo.from_obj(y).module()
        except ValueError:
            return None

    if (xp := infer_api(x)) is not None:
        return xp
    elif fallback is not None:
        return fallback
    else:
        raise ValueError(f"Could not infer array module for {type(x)}.")


def redirect(
    i: pxt.VarName,
    **kwargs: cabc.Mapping[str, cabc.Callable],
) -> cabc.Callable:
    """
    Change codepath for supplied array backends.

    Some functions/methods cannot be written in module-agnostic fashion. The action of this decorator is summarized
    below:

    * Analyze an array-valued parameter (`x`) of the wrapped function/method (`f`).
    * If `x` lies in one of the supplied array namespaces: re-route execution to the specified function.
    * If `x` lies in none of the supplied array namespaces: execute `f`.

    Parameters
    ----------
    i: VarName
        name of the array-like variable in `f` to base dispatch on.
    kwargs: ~collections.abc.Mapping

        * key[:py:class:`str`]: array backend short-name as defined in :py:class:`~pyxu.info.deps.NDArrayInfo`.
        * value[:py:class:`collections.abc.Callable`]: function/method to dispatch to.

    Notes
    -----
    Auto-dispatch via :py:func:`redirect` assumes the dispatcher/dispatchee have the same parameterization, i.e.:

    * if `f` is a function -> dispatch possible to another callable with identical signature (i.e., function or
      staticmethod)
    * if `f` is a staticmethod -> dispatch possible to another callable with identical signature (i.e., function or
      staticmethod)
    * if `f` is an instance-method -> dispatch to another instance-method of the class with identical signature.

    Example
    -------
    .. code-block:: python3

       def f(x, y): return "f"

       @redirect('x', NUMPY=f)    # if 'x' is of type NDArrayInfo.NUMPY, i.e. has
       def g(x, y): return "g"    # short-name 'NUMPY' -> reroute execution to `f`

       x1 = np.arange(5)
       x2 = da.array(x1)
       y = 1
       g(x1, y), g(x2, y)  # 'f', 'g'
    """

    def decorator(func: cabc.Callable) -> cabc.Callable:
        @functools.wraps(func)
        def wrapper(*ARGS, **KWARGS):
            try:
                func_args = pxm.parse_params(func, *ARGS, **KWARGS)
            except Exception as e:
                error_msg = f"Could not parameterize {func}()."
                raise ValueError(error_msg) from e

            if i not in func_args:
                error_msg = f"Parameter[{i}] not part of {func.__qualname__}() parameter list."
                raise ValueError(error_msg)

            ndi = pxd.NDArrayInfo.from_obj(func_args[i])
            if (alt_func := kwargs.get(ndi.name)) is not None:
                out = alt_func(**func_args)
            else:
                out = func(**func_args)

            return out

        return wrapper

    return decorator


def compute(*args, mode: str = "compute", **kwargs):
    r"""
    Force computation of Dask collections.

    Parameters
    ----------
    \*args: object, list
        Any number of objects.  If it is a dask object, it is evaluated and the result is returned.  Non-dask arguments
        are passed through unchanged.  Python collections are traversed to find/evaluate dask objects within.  (Use
        `traverse` =False to disable this behavior.)
    mode: str
        Dask evaluation strategy: compute or persist.
    \*\*kwargs: dict
        Extra keyword parameters forwarded to :py:func:`dask.compute` or :py:func:`dask.persist`.

    Returns
    -------
    \*cargs: object, list
        Evaluated objects. Non-dask arguments are passed through unchanged.
    """
    try:
        mode = mode.strip().lower()
        func = dict(compute=dask.compute, persist=dask.persist)[mode]
    except Exception:
        raise ValueError(f"mode: expected compute/persist, got {mode}.")

    cargs = func(*args, **kwargs)
    if len(args) == 1:
        cargs = cargs[0]
    return cargs


def to_NUMPY(x: pxt.NDArray) -> pxt.NDArray:
    """
    Convert an array from a specific backend to NUMPY.

    Parameters
    ----------
    x: NDArray
        Array to be converted.

    Returns
    -------
    y: NDArray
        Array with NumPy backend.

    Notes
    -----
    This function is a no-op if the array is already a NumPy array.
    """
    N = pxd.NDArrayInfo
    ndi = N.from_obj(x)
    if ndi == N.NUMPY:
        y = x
    elif ndi == N.DASK:
        y = compute(x)
    elif ndi == N.CUPY:
        y = x.get()
    else:
        msg = f"Dev-action required: define behaviour for {ndi}."
        raise ValueError(msg)
    return y
