import collections.abc as cabc
import enum
import importlib.util
import types

import dask.array
import numpy
import packaging.version as pkgv
import scipy.sparse

#: Show if CuPy-based backends are available.
CUPY_ENABLED: bool = importlib.util.find_spec("cupy") is not None
if CUPY_ENABLED:
    try:
        import cupy
        import cupyx.scipy.sparse
        import cupyx.scipy.sparse.linalg

        cupy.is_available()  # will fail if hardware/drivers/runtime missing
    except Exception:
        CUPY_ENABLED = False


@enum.unique
class NDArrayInfo(enum.Enum):
    """
    Supported dense array backends.
    """

    NUMPY = enum.auto()
    DASK = enum.auto()
    CUPY = enum.auto()

    @classmethod
    def default(cls) -> "NDArrayInfo":
        """Default array backend to use."""
        return cls.NUMPY

    def type(self) -> type:
        """Array type associated to a backend."""
        if self.name == "NUMPY":
            return numpy.ndarray
        elif self.name == "DASK":
            return dask.array.core.Array
        elif self.name == "CUPY":
            return cupy.ndarray if CUPY_ENABLED else type(None)
        else:
            raise ValueError(f"No known array type for {self.name}.")

    @classmethod
    def from_obj(cls, obj) -> "NDArrayInfo":
        """Find array backend associated to `obj`."""
        if obj is not None:
            for ndi in cls:
                if isinstance(obj, ndi.type()):
                    return ndi
        raise ValueError(f"No known array type to match {obj}.")

    @classmethod
    def from_flag(cls, gpu: bool) -> "NDArrayInfo":
        """Find array backend suitable for in-memory CPU/GPU computing."""
        if gpu:
            return cls.CUPY
        else:
            return cls.NUMPY

    def module(self, linalg: bool = False) -> types.ModuleType:
        """
        Python module associated to an array backend.

        Parameters
        ----------
        linalg: bool
            Return the linear-algebra submodule with identical API to :py:mod:`numpy.linalg`.
        """
        if self.name == "NUMPY":
            xp = numpy
            xpl = xp.linalg
        elif self.name == "DASK":
            xp = dask.array
            xpl = xp.linalg
        elif self.name == "CUPY":
            xp = cupy if CUPY_ENABLED else None
            xpl = xp if (xp is None) else xp.linalg
        else:
            raise ValueError(f"No known module(s) for {self.name}.")
        return xpl if linalg else xp


@enum.unique
class SparseArrayInfo(enum.Enum):
    """
    Supported sparse array backends.
    """

    SCIPY_SPARSE = enum.auto()
    CUPY_SPARSE = enum.auto()

    @classmethod
    def default(cls) -> "SparseArrayInfo":
        """Default array backend to use."""
        return cls.SCIPY_SPARSE

    def type(self) -> type:
        """Array type associated to a backend."""
        if self.name == "SCIPY_SPARSE":
            # All `*matrix` classes descend from `spmatrix`.
            return scipy.sparse.spmatrix
        elif self.name == "CUPY_SPARSE":
            return cupyx.scipy.sparse.spmatrix if CUPY_ENABLED else type(None)
        else:
            raise ValueError(f"No known array type for {self.name}.")

    @classmethod
    def from_obj(cls, obj) -> "SparseArrayInfo":
        """Find array backend associated to `obj`."""
        if obj is not None:
            for sai in cls:
                if isinstance(obj, sai.type()):
                    return sai
        raise ValueError(f"No known array type to match {sai}.")

    def module(self, linalg: bool = False) -> types.ModuleType:
        """
        Python module associated to an array backend.

        Parameters
        ----------
        linalg: bool
            Return the linear-algebra submodule with identical API to :py:mod:`scipy.sparse.linalg`.
        """
        if self.name == "SCIPY_SPARSE":
            xp = scipy.sparse
            xpl = xp.linalg
        elif self.name == "CUPY_SPARSE":
            xp = cupyx.scipy.sparse if CUPY_ENABLED else None
            xpl = xp if (xp is None) else cupyx.scipy.sparse.linalg
        else:
            raise ValueError(f"No known array module for {self.name}.")
        return xpl if linalg else xp


def supported_array_types() -> cabc.Collection[type]:
    """List of all supported dense array types in current Pyxu install."""
    data = set()
    for ndi in NDArrayInfo:
        if (ndi != NDArrayInfo.CUPY) or CUPY_ENABLED:
            data.add(ndi.type())
    return tuple(data)


def supported_array_modules() -> cabc.Collection[types.ModuleType]:
    """List of all supported dense array modules in current Pyxu install."""
    data = set()
    for ndi in NDArrayInfo:
        if (ndi != NDArrayInfo.CUPY) or CUPY_ENABLED:
            data.add(ndi.module())
    return tuple(data)


def supported_sparse_types() -> cabc.Collection[type]:
    """List of all supported sparse array types in current Pyxu install."""
    data = set()
    for sai in SparseArrayInfo:
        if (sai != SparseArrayInfo.CUPY_SPARSE) or CUPY_ENABLED:
            data.add(sai.type())
    return tuple(data)


def supported_sparse_modules() -> cabc.Collection[types.ModuleType]:
    """List of all supported sparse array modules in current Pyxu install."""
    data = set()
    for sai in SparseArrayInfo:
        if (sai != SparseArrayInfo.CUPY_SPARSE) or CUPY_ENABLED:
            data.add(sai.module())
    return tuple(data)


JAX_SUPPORT = dict(
    min=pkgv.Version("0.4.8"),
    max=pkgv.Version("1.0"),
)
PYTORCH_SUPPORT = dict(
    min=pkgv.Version("2.0"),
    max=pkgv.Version("3.0"),
)

__all__ = [
    "CUPY_ENABLED",
    "NDArrayInfo",
    "SparseArrayInfo",
    "supported_array_types",
    "supported_array_modules",
    "supported_sparse_types",
    "supported_sparse_modules",
    "JAX_SUPPORT",
    "PYTORCH_SUPPORT",
]
