import collections.abc as cabc
import contextlib
import enum
import functools
import inspect

import numpy as np


@enum.unique
class Width(enum.Enum):
    """
    Machine-dependent floating-point types.
    """

    HALF = np.dtype(np.half)
    SINGLE = np.dtype(np.single)
    DOUBLE = np.dtype(np.double)
    QUAD = np.dtype(np.longdouble)


class Precision(contextlib.AbstractContextManager):
    """
    Context Manager to locally redefine floating-point precision.

    Use this object via a with-block.

    Example
    -------
    >>> import pycsou.runtime as pyrt
    >>> pyrt.getPrecision()                      # Width.DOUBLE
    ... with pyrt.Precision(pyrt.Width.HALF):
    ...     pyrt.getPrecision()                  # Width.HALF
    ... pyrt.getPrecision()                      # Width.DOUBLE
    """

    def __init__(self, width: Width):
        self._width = width
        self._width_prev = getPrecision()

    def __enter__(self) -> "Precision":
        _setPrecision(self._width)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        exc_raised = any(_ is not None for _ in [exc_type, exc_value, traceback])
        if exc_raised:
            pass

        _setPrecision(self._width_prev)
        return False if exc_raised else True


def enforce_precision(i: cabc.Collection[str] = frozenset(), o: bool = True) -> cabc.Callable:
    """
    Decorator to pre/post-process function parameters to enforce runtime FP-precision.

    Parameters
    ----------
    i: cabc.Collection[str]
        Function parameters for which precision must be enforced to runtime's FP-precision.
        Function parameter values must have a NumPy API.
    o: bool
        If True (default), ensure function's output has runtime's FP-precision.
        If function's output does not have a NumPy API, set `o` explicitly to False.

    Example
    -------
    >>> import pycsou.runtime as pyrt
    >>> @pyrt.enforce_precision(i='y', o=False)  # `i` can process multiple args: `i=('x','y')`.
    ... def f(x, y, z=1):
    ...     print(x.dtype, y.dtype)
    ...     return x + y + z
    ...
    ... x = np.arange(5)
    ... y = np.r_[0.5]
    ... print(x.dtype, y.dtype)                  # int64, float64
    ... with pyrt.Precision(pyrt.Width.SINGLE):
    ...     out = f(x,y)                         # int64, float32 (printed inside f-call.)
    ... print(out.dtype)                         # float64 (would have been float32 if `o=True`)
    """

    def decorator(func: cabc.Callable) -> cabc.Callable:
        @functools.wraps(func)
        def wrapper(*ARGS, **KWARGS):
            dtype = getPrecision().value

            sig = inspect.Signature.from_callable(func)
            func_args = sig.bind(*ARGS, **KWARGS)
            func_args.apply_defaults()
            func_args = func_args.arguments
            for k in i:
                if k not in func_args:
                    error_msg = f"Parameter[{k}] not part of {func.__qualname__}() parameter list."
                    raise ValueError(error_msg)
                else:  # change input precision
                    func_args[k] = func_args[k].astype(dtype, copy=False)

            out = func(**func_args)
            if o and (out is not None):
                out = out.astype(dtype, copy=False)
            return out

        return wrapper

    return decorator


def getPrecision() -> Width:
    state = globals()
    return state["__width"]


def _setPrecision(width: Width):
    # For internal use only. It is recommended to modify FP-precision locally using the `Precision`
    # context manager.
    state = globals()
    state["__width"] = width


__width = Width.DOUBLE  # default FP-precision.
