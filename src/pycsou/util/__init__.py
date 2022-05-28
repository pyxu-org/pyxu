import collections.abc as cabc
import functools
import inspect

import dask
import numpy as np

import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct


def infer_sum_shape(sh1: pyct.Shape, sh2: pyct.Shape) -> pyct.Shape:
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


def infer_composition_shape(sh1: pyct.Shape, sh2: pyct.Shape) -> pyct.Shape:
    A, B, C, D = *sh1, *sh2
    if None in (A, C):
        raise ValueError("Composition of codomain-dimension-agnostic operators is not supported.")
    elif (B == C) or (B is None):
        return (A, D)
    else:
        raise ValueError(f"Composition of {sh1} and {sh2} operators forbidden.")


def infer_stack_shape(*shapes, axis):
    dims = [shape[1] for shape in shapes]
    codims = [shape[0] for shape in shapes]
    if axis == 0:
        unique_dims = np.unique(np.array(dims).astype(float))
        try:
            assert unique_dims.size <= 2
        except:
            raise ValueError("Inconsistent map shapes for vertical stacking.")
        dim = np.nansum(unique_dims)
        dim = None if np.isnan(dim) else int(dim)
        return (int(np.sum(codims).astype(int)), dim)
    else:
        try:
            assert np.all(~np.isnan(np.array(dims).astype(float)))
        except:
            raise ValueError("Horizontal stackings of maps including domain-agnostic maps is ambiguous.")
        unique_codim = np.unique(np.array(codims).astype(float))
        try:
            assert unique_codim.size == 1
        except:
            raise ValueError("Inconsistent map shapes for horizontal stacking.")
        return (int(unique_codim), int(np.sum(dims).astype(int)))


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


def parse_params(func, *args, **kwargs) -> cabc.Mapping:
    """
    Get function parameterization.

    Returns
    -------
    params: dict
        (key, value) params as seen in body of `func` when called via `func(*args, **kwargs)`.
    """
    sig = inspect.Signature.from_callable(func)
    f_args = sig.bind(*args, **kwargs)
    f_args.apply_defaults()

    params = dict(
        zip(f_args.arguments.keys(), f_args.args),  # positional arguments
        **f_args.kwargs,
    )
    return params


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
            func_args = parse_params(func, *ARGS, **KWARGS)

            x = func_args[i]
            if is_1d := x.ndim == 1:
                x = x.reshape((1, x.size))
            sh_x = x.shape  # (..., N)
            sh_xf = (np.prod(sh_x[:-1]), sh_x[-1])  # (M, N): x flattened to 2D
            x = x.reshape(sh_xf)

            # infer output dimensions + allocate
            func_args[i] = x[0]
            y0 = func(**func_args)
            xp = get_array_module(y0)
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
                func_args = parse_params(func, *ARGS, **KWARGS)
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
