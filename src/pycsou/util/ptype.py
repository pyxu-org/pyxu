"""
The purpose of this module is to group all abstract types and helpers usefol Python's static
type-checker.
"""

import collections.abc as cabc
import numbers as nb
import pathlib as plib
import typing as typ

import pycsou.util.deps as pycd

# supported array types
NDArray = typ.TypeVar("NDArray", *pycd.supported_array_types())

# supported array modules
ArrayModule = typ.TypeVar(
    "ArrayModule",
    *[typ.Literal[_] for _ in pycd.supported_array_modules()],
)

# supported sparse arrays
SparseArray = typ.TypeVar("SparseArray", *pycd.supported_sparse_types())

# supported sparse modules
SparseModule = typ.TypeVar(
    "SparseModule",
    *[typ.Literal[_] for _ in pycd.supported_sparse_modules()],
)

# non-complex numbers
Real = nb.Real

# map shapes
Shape = typ.Tuple[int, typ.Union[int, None]]
ShapeOrDim = typ.Union[typ.Union[int, None], Shape]
NonAgnosticShape = typ.Tuple[int, int]
SquareShape = typ.Union[int, typ.Tuple[int, int]]

# one or more variable names
VarName = typ.Union[str, cabc.Collection[str]]

PathLike = typ.Union[str, plib.Path]
