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

#: Supported dense array types.
NDArray = typ.TypeVar("NDArray", *pxd.supported_array_types())

#: Supported dense array modules.
ArrayModule = typ.TypeVar(
    "ArrayModule",
    *[typ.Literal[_] for _ in pxd.supported_array_modules()],
)

#: Supported sparse array types.
SparseArray = typ.TypeVar("SparseArray", *pxd.supported_sparse_types())

#: Supported sparse array modules.
SparseModule = typ.TypeVar(
    "SparseModule",
    *[typ.Literal[_] for _ in pxd.supported_sparse_modules()],
)

#: Top-level abstract :py:class:`~pyxu.abc.operator.Operator` interface exposed to users.
OpT = typ.TypeVar(
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

#: :py:class:`~pyxu.abc.operator.Operator` hierarchy class type.
OpC = typ.Type[OpT]  # Operator classes

#: Mathematical properties attached to :py:class:`~pyxu.abc.operator.Operator` objects.
Property = "pxo.Property"

#: Top-level abstract :py:class:`~pyxu.abc.solver.Solver` interface exposed to users.
SolverT = typ.TypeVar("SolverT", bound="pxs.Solver")

#: :py:class:`~pyxu.abc.solver.Solver` hierarchy class type.
SolverC = typ.Type[SolverT]

#: Solver run-modes.
SolverM = typ.TypeVar("SolverM", bound="pxs.Mode")

Integer = nb.Integral
Real = nb.Real
DType = npt.DTypeLike  #: NDArray dtype specifier.
OpShape = tuple[Integer, Integer]  #: Operator shape specifier.
NDArrayAxis = typ.Union[Integer, tuple[Integer, ...]]  #: Axis/Axes specifier.
NDArrayShape = typ.Union[Integer, tuple[Integer, ...]]  #: NDArray shape specifier.
Path = typ.Union[str, plib.Path]  #: Path-like object.
VarName = typ.Union[str, cabc.Collection[str]]  #: Variable name(s).
