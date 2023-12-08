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
if len(sst := pxd.supported_sparse_types()) == 1:
    SparseArray = typ.TypeVar("SparseArray", bound=tuple(sst)[0])
else:
    SparseArray = typ.TypeVar("SparseArray", *sst)

#: Supported sparse array modules.
if len(ssm := pxd.supported_sparse_modules()) == 1:
    SparseModule = typ.TypeVar("SparseModule", bound=tuple(ssm)[0])
else:
    SparseModule = typ.TypeVar("SparseModule", *[typ.Literal[_] for _ in ssm])

#: Top-level abstract :py:class:`~pyxu.abc.Operator` interface exposed to users.
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

#: :py:class:`~pyxu.abc.Operator` hierarchy class type.
OpC = typ.Type[OpT]  # Operator classes

#: Mathematical properties attached to :py:class:`~pyxu.abc.Operator` objects.
Property = "pxo.Property"

#: Top-level abstract :py:class:`~pyxu.abc.Solver` interface exposed to users.
SolverT = typ.TypeVar("SolverT", bound="pxs.Solver")

#: :py:class:`~pyxu.abc.Solver` hierarchy class type.
SolverC = typ.Type[SolverT]

#: Solver run-modes.
SolverM = typ.TypeVar("SolverM", bound="pxs.SolverMode")

Integer = nb.Integral
Real = nb.Real  #: Alias of :py:class:`numbers.Real`.
DType = npt.DTypeLike  #: :py:attr:`~pyxu.info.ptype.NDArray` dtype specifier.
NDArrayAxis = typ.Union[Integer, tuple[Integer, ...]]  #: Axis/Axes specifier.
NDArrayShape = typ.Union[Integer, tuple[Integer, ...]]  #: :py:attr:`~pyxu.info.ptype.NDArray` shape specifier.
Path = typ.Union[str, plib.Path]  #: Path-like object.
VarName = typ.Union[str, cabc.Collection[str]]  #: Variable name(s).
