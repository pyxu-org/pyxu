import collections.abc as cabc
import contextlib
import enum
import functools
import inspect
import numbers as nb

import numpy as np

import pycsou.util as pycu
import pycsou.util.ptype as pyct


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
    >>> import pycsou.runtime as pycrt
    >>> pycrt.getPrecision()                      # Width.DOUBLE
    ... with pycrt.Precision(pycrt.Width.HALF):
    ...     pycrt.getPrecision()                  # Width.HALF
    ... pycrt.getPrecision()                      # Width.DOUBLE
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


def enforce_precision(
    i: pyct.VarName = frozenset(),
    o: bool = True,
    allow_None: bool = True,
) -> cabc.Callable:
    """
    Decorator to pre/post-process function parameters to enforce runtime FP-precision.

    Parameters
    ----------
    i: VarName
        Function parameters for which precision must be enforced to runtime's FP-precision.
        Function parameter values must have a NumPy API, or be scalars.
        None-valued parameters are allowed if `allow_None` is True (default).
    o: bool
        If True (default), ensure function's output (if any) has runtime's FP-precision.
        If function's output does not have a NumPy API or is not scalar-valued, set `o` explicitly
        to False.
    allow_None: bool

    Example
    -------
    >>> import pycsou.runtime as pycrt
    >>> @pycrt.enforce_precision(i='y', o=False)  # `i` can process multiple args: `i=('x','y')`.
    ... def f(x, y, z=1):
    ...     print(x.dtype, y.dtype)
    ...     return x + y + z
    >>> x, y = np.arange(5), np.r_[0.5]
    >>> print(x.dtype, y.dtype)
    int64 float64
    >>> with pycrt.Precision(pycrt.Width.SINGLE):
    ...     out = f(x,y)                         # int64, float32 (printed inside f-call.)
    int64 float32
    >>> print(out.dtype)                         # float64 (would have been float32 if `o=True`)
    float64
    """

    def decorator(func: cabc.Callable) -> cabc.Callable:
        @functools.wraps(func)
        def wrapper(*ARGS, **KWARGS):
            sig = inspect.Signature.from_callable(func)
            func_args = sig.bind(*ARGS, **KWARGS)
            func_args.apply_defaults()
            func_args = func_args.arguments

            for k in [i] if isinstance(i, str) else i:
                if k not in func_args:
                    error_msg = f"Parameter[{k}] not part of {func.__qualname__}() parameter list."
                    raise ValueError(error_msg)
                elif (func_args[k] is None) and (not allow_None):
                    raise ValueError(f"Parameter[{k}] cannot be None-valued.")
                else:
                    func_args[k] = coerce(func_args[k])

            out = func(**func_args)
            if o and (out is not None):
                out = coerce(out)
            return out

        return wrapper

    return decorator


def getPrecision() -> Width:
    state = globals()
    return state["__width"]


def coerce(x):
    """
    Transform input to match runtime FP-precision.

    Parameters
    ----------
    x: Real | NDArray

    Returns
    -------
    y: Real | NDArray
        Input cast to the runtime FP-precision.
        Fails if operation is impossible or unsafe. (I.e. casting complex-valued data.)
    """
    dtype = getPrecision().value

    try:
        if isinstance(x, pyct.Real):
            return np.array(x, dtype=dtype)[()]
        elif isinstance(x, nb.Number):
            raise  # other number categories cannot be converted.
        else:
            return x.astype(dtype, copy=False)
    except:
        raise TypeError(f"Cannot coerce {type(x)} to scalar/array of precision {dtype}.")


def _setPrecision(width: Width):
    # For internal use only. It is recommended to modify FP-precision locally using the `Precision`
    # context manager.
    state = globals()
    state["__width"] = width


__width = Width.DOUBLE  # default FP-precision.
