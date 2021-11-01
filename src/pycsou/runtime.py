import contextlib
import enum

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


def getPrecision() -> Width:
    state = globals()
    return state["__width"]


def _setPrecision(width: Width):
    # For internal use only. It is recommended to modify compute precision locally using
    # `Precision`.
    state = globals()
    state["__width"] = width


__width = Width.DOUBLE  # default FP-precision.
