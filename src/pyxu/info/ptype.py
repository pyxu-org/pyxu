"""
The purpose of this module is to group all abstract types and helpers useful to Python's static type-checker.
"""

import collections.abc as cabc
import numbers as nb
import pathlib as plib
import typing as typ

import numpy.typing as npt

import pyxu.info.deps as pxd

if typ.TYPE_CHECKING:
    import pyxu.abc.operator as pxo
    import pyxu.abc.solver as pxs

# supported dense arrays/modules
NDArray = typ.TypeVar("NDArray", *pxd.supported_array_types())
ArrayModule = typ.TypeVar(
    "ArrayModule",
    *[typ.Literal[_] for _ in pxd.supported_array_modules()],
)

# supported sparse arrays/modules
SparseArray = typ.TypeVar("SparseArray", *pxd.supported_sparse_types())
SparseModule = typ.TypeVar(
    "SparseModule",
    *[typ.Literal[_] for _ in pxd.supported_sparse_modules()],
)

OpT = typ.TypeVar(
    # Top-level operator exposed to users.
    # This list should be kept in sync with all user-facing operators in `pxo`.
    "OpT",
    "pxo.Operator",
    "pxo.DiffFunc",
    "pxo.ProxDiffFunc",
    "pxo.NormalOp",
    "pxo.ProxFunc",
    "pxo.SquareOp",
    "pxo.QuadraticFunc",
    "pxo.ProjOp",
    "pxo.LinFunc",
    "pxo.PosDefOp",
    "pxo.Map",
    "pxo.Func",
    "pxo.OrthProjOp",
    "pxo.DiffMap",
    "pxo.UnitOp",
    "pxo.SelfAdjointOp",
    "pxo.LinOp",
)
OpC = typ.Type[OpT]  # Operator classes
Property = "pxo.Property"

# Top-level solver exposed to users
SolverT = typ.TypeVar("SolverT", bound="pxs.Solver")  # Solver instances
SolverC = typ.Type[SolverT]  # Solver classes
SolverM = typ.TypeVar("SolverM", bound="pxs.Mode")

# Other
Integer = int  # [Sepand] `nb.Integral` seems to not work will with PyCharm...
Real = nb.Real
DType = npt.DTypeLike
OpShape = tuple[Integer, Integer]
NDArrayAxis = typ.Union[Integer, tuple[Integer, ...]]
NDArrayShape = typ.Union[Integer, tuple[Integer, ...]]
Path = typ.Union[str, plib.Path]
VarName = typ.Union[str, cabc.Collection[str]]
