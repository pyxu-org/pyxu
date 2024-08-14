import enum

import numpy as np

import pyxu.info.ptype as pxt

__all__ = [
    "Width",
    "CWidth",
]


@enum.unique
class Width(enum.Enum):
    """
    Machine-dependent floating-point types.
    """

    SINGLE = np.dtype(np.single)
    DOUBLE = np.dtype(np.double)

    def eps(self) -> pxt.Real:
        """
        Machine precision of a floating-point type.

        Returns the difference between 1 and the next smallest representable float larger than 1.
        """
        eps = np.finfo(self.value).eps
        return float(eps)

    @property
    def complex(self) -> "CWidth":
        """
        Returns precision-equivalent complex-valued type.
        """
        return CWidth[self.name]


@enum.unique
class CWidth(enum.Enum):
    """
    Machine-dependent complex-valued floating-point types.
    """

    SINGLE = np.dtype(np.csingle)
    DOUBLE = np.dtype(np.cdouble)

    @property
    def real(self) -> "Width":
        """
        Returns precision-equivalent real-valued type.
        """
        return Width[self.name]
