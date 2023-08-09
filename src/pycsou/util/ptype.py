"""
The purpose of this module is to group all abstract types and helpers useful to Python's static type-checker.
"""

import collections.abc as cabc
import numbers as nb
import pathlib as plib
import typing as typ

import numpy.typing as npt

import pycsou.util.deps as pycd

if typ.TYPE_CHECKING:
    import pycsou.abc.operator as pyco
    import pycsou.abc.solver as pycs

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

OpT = typ.TypeVar(
    # Top-level operator exposed to users.
    # This list should be kept in sync with all user-facing operators in `pyco`.
    "OpT",
    "pyco.Operator",
    "pyco.DiffFunc",
    "pyco.ProxDiffFunc",
    "pyco.NormalOp",
    "pyco.ProxFunc",
    "pyco.SquareOp",
    "pyco.QuadraticFunc",
    "pyco.ProjOp",
    "pyco.LinFunc",
    "pyco.PosDefOp",
    "pyco.Map",
    "pyco.Func",
    "pyco.OrthProjOp",
    "pyco.DiffMap",
    "pyco.UnitOp",
    "pyco.SelfAdjointOp",
    "pyco.LinOp",
)
OpC = typ.Type[OpT]  # Operator classes
Property = "pyco.Property"

# Top-level solver exposed to users
SolverT = typ.TypeVar("SolverT", bound="pycs.Solver")  # Solver instances
SolverC = typ.Type[SolverT]  # Solver classes
SolverM = typ.TypeVar("SolverM", bound="pycs.Mode")

# Other
Integer = int  # [Sepand] `nb.Integral` seems to not work will with PyCharm...
Real = nb.Real
DType = npt.DTypeLike
OpShape = tuple[Integer, typ.Union[Integer, None]]
NDArrayAxis = typ.Union[Integer, tuple[Integer, ...]]
NDArrayShape = typ.Union[Integer, tuple[Integer, ...]]
Path = typ.Union[str, plib.Path]
VarName = typ.Union[str, cabc.Collection[str]]
