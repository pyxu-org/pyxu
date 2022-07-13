"""
The purpose of this module is to group all abstract types and helpers usefol Python's static
type-checker.
"""

import collections.abc as cabc
import numbers as nb
import pathlib as plib
import typing as typ

import numpy.typing as npt

import pycsou.util.deps as pycd

# supported dense arrays/modules
NDArray = typ.TypeVar("NDArray", *pycd.supported_array_types())
ArrayModule = typ.TypeVar(
    "ArrayModule",
    *[typ.Literal[_] for _ in pycd.supported_array_modules()],
)

# supported sparse arrays/modules
SparseArray = typ.TypeVar("SparseArray", *pycd.supported_sparse_types())
SparseModule = typ.TypeVar(
    "SparseModule",
    *[typ.Literal[_] for _ in pycd.supported_sparse_modules()],
)

# Top-level operator exposed to users
OpT = typ.Literal["Operator"]  # Operator instances
OpC = typ.Type[OpT]  # Operator classes
Property = typ.Literal["Property"]

# Other
Integer = nb.Integral
Real = nb.Real
DType = npt.DTypeLike
Shape = tuple[Integer, typ.Union[Integer, None]]
Path = typ.Union[str, plib.Path]
VarName = typ.Union[str, cabc.Collection[str]]
