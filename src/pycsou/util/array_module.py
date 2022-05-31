import collections.abc as cabc
import functools

import dask

import pycsou.util.deps as pycd
import pycsou.util.inspect as pycui
import pycsou.util.ptype as pyct

__all__ = [
    "get_array_module",
    "compute",
    "redirect",
]


def get_array_module(x, fallback: pyct.ArrayModule = None) -> pyct.ArrayModule:
    """
    Get the array namespace corresponding to a given object.

    Parameters
    ----------
    x: object
        Any object compatible with the interface of NumPy arrays.
    fallback: pycsou.util.ptype.ArrayModule
        Fallback module if `x` is not a NumPy-like array.
        Default behaviour: raise error if fallback used.

    Returns
    -------
    namespace: pycsou.util.ptype.ArrayModule
        The namespace to use to manipulate `x`, or `fallback` if provided.
    """

    def infer_api(y):
        for array_t, api, _ in pycd.array_backend_info():
            if isinstance(y, array_t):
                return api
        return None

    if (xp := infer_api(x)) is not None:
        return xp
    elif fallback is not None:
        return fallback
    else:
        raise ValueError(f"Could not infer array module for {type(x)}.")


def compute(*args, mode: str = "compute", **kwargs):
    """
    Force computation of Dask collections.

    Parameters
    ----------
    *args: object | sequence(object)
        Any number of objects.
        If it is a dask object, it is evaluated and the result is returned.
        Non-dask arguments are passed through unchanged.
        Python collections are traversed to find/evaluate dask objects within.
        (Use traverse=False to disable this behavior.)
    mode: str
        Dask evaluation strategy: compute or persist.
    kwargs: dict
        Extra keyword parameters forwarded to `dask.[compute, persist]`.

    Returns
    -------
    *cargs: object | sequence(object)
        Evaluated objects. Non-dask arguments are passed through unchanged.
    """
    try:
        func = dict(compute=dask.compute, persist=dask.persist)[mode.lower()]
    except:
        raise ValueError(f"mode: expected compute/persist, got {mode}.")

    cargs = func(*args, **kwargs)
    if len(args) == 1:
        cargs = cargs[0]
    return cargs


def redirect(
    i: pyct.VarName,
    **kwargs: cabc.Mapping[str, cabc.Callable],
) -> cabc.Callable:
    """
    Change codepath for supplied array backends.

    Some functions/methods cannot be written in module-agnostic fashion. The action of this
    decorator is summarized below:

    * Analyze an array-valued parameter (`x`) of the wrapped function/method (`f`).
    * If `x` lies in one of the supplied array namespaces: re-route execution to the specified
      function.
    * If `x` lies in none of the supplied array namespaces: execute `f`.

    Parameters
    ----------
    i: str
        name of the array-like variable in `f` to base dispatch on.
    **kwargs: dict[str, callable]

        key: array backend short-name as defined in :py:func:`~pycsou.util.deps.array_backend_info`.

        value: function/method to dispatch to.

    Notes
    -----
    Auto-dispatch via :py:func:`redirect` assumes the
    dispatcher/dispatchee have the same parameterization, i.e.:

    * if `f` is a function -> dispatch possible to another callable with identical signature (i.e.,
      function or staticmethod)
    * if `f` is a staticmethod -> dispatch possible to another callable with identical signature
      (i.e., function or staticmethod)
    * if `f` is an instance-method -> dispatch to another instance-method of the class with
      identical signature.

    Example
    -------
    >>> def f(x, y): return "f"
    >>>
    >>> @redirect('x', NUMPY=f)    # if 'x' lies in the array namespace having
    >>> def g(x, y): return "g"    # short-name 'NUMPY' -> reroute execution to `f`
    >>>
    >>> x1 = np.arange(5)
    >>> x2 = da.array(x1)
    >>> y = 1
    >>> g(x1, y), g(x2, y)  # 'f', 'g'
    """

    def decorator(func: cabc.Callable) -> cabc.Callable:
        @functools.wraps(func)
        def wrapper(*ARGS, **KWARGS):
            try:
                func_args = pycui.parse_params(func, *ARGS, **KWARGS)
            except Exception as e:
                error_msg = f"Could not parameterize {func}()."
                raise ValueError(error_msg) from e

            if i not in func_args:
                error_msg = f"Parameter[{i}] not part of {func.__qualname__}() parameter list."
                raise ValueError(error_msg)

            xp = get_array_module(func_args[i])
            short_name = {xp_: short_name for (_, xp_, short_name) in pycd.array_backend_info()}.get(xp)

            if (alt_func := kwargs.get(short_name)) is not None:
                out = alt_func(**func_args)
            else:
                out = func(**func_args)

            return out

        return wrapper

    return decorator
