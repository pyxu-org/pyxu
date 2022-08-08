import collections.abc as cabc
import functools as ft
import types
import typing as typ
import warnings

import numpy as np
import scipy.sparse as sp
import sparse as ssp

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycuw


class IdentityOp(pyca.UnitOp):
    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set()
        for klass in cls.__bases__:
            p |= klass.properties()
        return frozenset(p)

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return pycu.read_only(arr)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return pycu.read_only(arr)

    def svdvals(self, **kwargs) -> pyct.NDArray:
        if kwargs.pop("gpu", False):
            import cupy as xp
        else:
            xp = np
        D = xp.ones(kwargs.pop("k"), dtype=pycrt.getPrecision().value)
        return D

    def eigvals(self, **kwargs) -> pyct.NDArray:
        return self.svdvals(**kwargs)

    def asarray(self, **kwargs) -> pyct.NDArray:
        dtype = kwargs.pop("dtype", pycrt.getPrecision().value)
        xp = kwargs.pop("xp", np)
        A = xp.eye(N=self.dim, dtype=dtype)
        return A

    def trace(self, **kwargs) -> pyct.Real:
        return float(self.dim)


class NullOp(pyca.LinOp):
    """
    Null operator.

    This operator maps any input vector on the null vector.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        self._lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.broadcast_to(
            xp.array(0, arr.dtype),
            (*arr.shape[:-1], self.codim),
        )

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.broadcast_to(
            xp.array(0, arr.dtype),
            (*arr.shape[:-1], self.dim),
        )

    def svdvals(self, **kwargs) -> pyct.NDArray:
        if kwargs.pop("gpu", False):
            import cupy as xp
        else:
            xp = np
        D = xp.zeros(kwargs.pop("k"), dtype=pycrt.getPrecision().value)
        return D

    def gram(self) -> pyct.OpT:
        return NullOp(shape=(self.dim, self.dim)).asop(pyca.PosDefOp)

    def cogram(self) -> pyct.OpT:
        return NullOp(shape=(self.codim, self.codim)).asop(pyca.PosDefOp)

    def asarray(self, **kwargs) -> pyct.NDArray:
        dtype = kwargs.pop("dtype", pycrt.getPrecision().value)
        xp = kwargs.pop("xp", np)
        A = xp.zeros(self.shape, dtype=dtype)
        return A

    def trace(self, **kwargs) -> pyct.Real:
        return float(0)


def NullFunc(dim: pyct.Integer) -> pyct.OpT:
    """
    Null functional.

    This functional maps any input vector on the null scalar.
    """
    op = NullOp(shape=(1, dim))._squeeze()
    return op


def HomothetyOp(cst: pyct.Real, dim: pyct.Integer) -> pyct.OpT:
    """
    Scaling operator.

    Parameters
    ----------
    cst: pyct.Real
        Scaling factor.
    dim: pyct.Integer
        Dimension of the domain.

    Returns
    -------
    op: pyct.OpT
        (dim, dim) scaling operator.

    Notes
    -----
    This operator is not defined in terms of :py:func:`~pycsou.operator.linop.DiagonalOp` since it
    is array-backend-agnostic.
    """
    assert isinstance(cst, pyct.Real), f"cst: expected real, got {cst}."

    if np.isclose(cst, 0):
        op = NullOp(shape=(dim, dim))
    elif np.isclose(cst, 1):
        op = IdentityOp(dim=dim)
    else:  # build PosDef or SelfAdjointOp

        @pycrt.enforce_precision(i="arr")
        def op_apply(_, arr):
            out = arr.copy()
            out *= _._cst
            return out

        def op_svdvals(_, **kwargs):
            if kwargs.pop("gpu", False):
                import cupy as xp
            else:
                xp = np
            D = xp.full(
                shape=kwargs.pop("k"),
                fill_value=abs(_._cst),
                dtype=pycrt.getPrecision().value,
            )
            return D

        def op_eigvals(_, **kwargs):
            D = _.svdvals(**kwargs)
            D *= np.sign(_._cst)
            return D

        def op_gram(_):
            return HomothetyOp(cst=_._cst**2, dim=_.dim)

        def op_trace(_, **kwargs):
            out = _._cst * _.codim
            return float(out)

        klass = pyca.PosDefOp if (cst > 0) else pyca.SelfAdjointOp
        op = klass(shape=(dim, dim))
        op._cst = cst
        op._lipschitz = abs(cst)
        op.apply = types.MethodType(op_apply, op)
        op.svdvals = types.MethodType(op_svdvals, op)
        op.eigvals = types.MethodType(op_eigvals, op)
        op.gram = types.MethodType(op_gram, op)
        op.cogram = op.gram
        op.trace = types.MethodType(op_trace, op)

    # IdentityOp(dim>1) cannot be squeezed since it doesn't fall into a single core-operator
    # category.
    return op._squeeze() if (op.codim == 1) else op


def DiagonalOp(
    vec: pyct.NDArray,
    enable_warnings: bool = True,
) -> pyct.OpT:
    r"""
    Diagonal linear operator :math:`L: \mathbf{x} \to \text{diag}(\mathbf{v}) \mathbf{x}`.

    Notes
    -----
    :py:func:`~pycsou.operator.linop.base.DiagonalOp` instances are **not arraymodule-agnostic**:
    they will only work with NDArrays belonging to the same array module as ``vec``.
    Moreover, inner computations may cast input arrays when the precision of ``vec`` does not match
    the user-requested precision.
    If such a situation occurs, a warning is raised.

    Parameters
    ----------
    vec: pyct.NDArray
        (N,) diagonal scale factors.
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mis-match issues.
    """
    assert len(vec) == np.prod(vec.shape), f"vec: {vec.shape} is not a DiagonalOp generator."
    if (dim := vec.size) == 1:  # Module-agnostic
        return HomothetyOp(cst=float(vec), dim=1)
    else:
        xp = pycu.get_array_module(vec)
        if pycu.compute(xp.allclose(vec, 0)):
            op = NullOp(shape=(dim, dim))
        elif pycu.compute(xp.allclose(vec, 1)):
            op = IdentityOp(dim=dim)
        else:  # build PosDef or SelfAdjointOp

            @pycrt.enforce_precision(i="arr")
            def op_apply(_, arr):
                if (_._vec.dtype != arr.dtype) and _._enable_warnings:
                    msg = "Computation may not be performed at the requested precision."
                    warnings.warn(msg, pycuw.PrecisionWarning)
                out = arr.copy()
                out *= _._vec
                return out

            def op_asarray(_, **kwargs) -> pyct.NDArray:
                dtype = kwargs.pop("dtype", pycrt.getPrecision().value)
                xp = kwargs.pop("xp", np)
                try:  # NumPy/Dask
                    v = xp.array(_._vec, dtype=dtype)
                except TypeError:  # CuPy
                    v = _._vec.get()
                A = xp.diag(v.astype(dtype=dtype, copy=False))
                return A

            def op_gram(_):
                return DiagonalOp(
                    vec=_._vec**2,
                    enable_warnings=_._enable_warnings,
                )

            def op_svdvals(_, **kwargs):
                k = kwargs.pop("k")
                which = kwargs.pop("which", "LM")
                if kwargs.pop("gpu", False):
                    import cupy as xp
                else:
                    xp = np
                D = xp.abs(pycu.compute(_._vec))
                D = D[xp.argsort(D)]
                D = D.astype(pycrt.getPrecision().value, copy=False)
                return D[:k] if (which == "SM") else D[-k:]

            def op_eigvals(_, **kwargs):
                k = kwargs.pop("k")
                which = kwargs.pop("which", "LM")
                if kwargs.pop("gpu", False):
                    import cupy as xp
                else:
                    xp = np
                D = pycu.compute(_._vec)
                D = D[xp.argsort(xp.abs(D))]
                D = D.astype(pycrt.getPrecision().value, copy=False)
                return D[:k] if (which == "SM") else D[-k:]

            @pycrt.enforce_precision(i="arr")
            def op_pinv(_, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
                damp = kwargs.pop("damp", 0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scale = _._vec / (_._vec**2 + damp)
                    scale[xp.isnan(scale)] = 0
                out = arr.copy()
                out *= scale
                return out

            def dagger(self, **kwargs) -> pyct.OpT:
                damp = kwargs.pop("damp", 0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scale = _._vec / (_._vec**2 + damp)
                    scale[xp.isnan(scale)] = 0
                return DiagonalOp(
                    vec=scale,
                    enable_warnings=_._enable_warnings,
                )

            def op_trace(_, **kwargs):
                return float(_._vec.sum())

            klass = pyca.PosDefOp if pycu.compute(xp.all(vec > 0)) else pyca.SelfAdjointOp
            op = klass(shape=(dim, dim))
            op._vec = vec
            op._enable_warnings = bool(enable_warnings)
            op._lipschitz = pycu.compute(xp.abs(vec).max())
            op.apply = types.MethodType(op_apply, op)
            op.asarray = types.MethodType(op_asarray, op)
            op.gram = types.MethodType(op_gram, op)
            op.cogram = op.gram
            op.svdvals = types.MethodType(op_svdvals, op)
            op.eigvals = types.MethodType(op_eigvals, op)
            op.pinv = types.MethodType(op_pinv, op)
            op.trace = types.MethodType(op_trace, op)

        # IdentityOp(dim>1) cannot be squeezed since it doesn't fall into a single core-operator
        # category.
        return op._squeeze() if (op.codim == 1) else op


def ExplicitLinOp(
    cls: pyct.OpC,
    mat: typ.Union[pyct.NDArray, pyct.SparseArray],
    enable_warnings: bool = True,
) -> pyct.OpT:
    r"""
    Build a linear operator from its matrix representation.

    Given a matrix :math:`\mathbf{A}\in\mathbb{R}^{M\times N}`, the *explicit linear operator*
    associated to :math:`\mathbf{A}` is defined as

    .. math::

       f_\mathbf{A}(\mathbf{x})
       =
       \mathbf{A}\mathbf{x},
       \qquad
       \forall \mathbf{x}\in\mathbb{R}^N,

    with adjoint given by:

    .. math::

       f^\ast_\mathbf{A}(\mathbf{z})
       =
       \mathbf{A}^T\mathbf{z},
       \qquad
       \forall \mathbf{z}\in\mathbb{R}^M.

    Parameters
    ----------
    cls: pyct.OpC
        LinOp sub-class to instantiate.
    mat: pyct.NDArray | pyct.SparseArray
        (M, N) matrix generator.
        The input array can be *dense* or *sparse*.
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mis-match issues.

    Notes
    -----
    :py:class:`~pycsou.operator.linop.base.ExplicitLinOp` instances are **not arraymodule-agnostic**:
    they will only work with NDArrays belonging to the same array module as ``mat``.
    Moreover, inner computations may cast input arrays when the precision of ``mat`` does not match
    the user-requested precision.
    If such a situation occurs, a warning is raised.
    """

    def _standard_form(A):
        fail_dense = False
        try:
            pycu.get_array_module(A)
            return A
        except:
            fail_dense = True

        fail_scipy_sparse = False
        try:
            return A.tocoo().tocsr()
        except:
            fail_scipy_sparse = True

        fail_pydata_sparse = False
        try:
            return ssp.as_coo(A).tocsr()
        except:
            fail_pydata_sparse = True

        if fail_dense and fail_scipy_sparse and fail_pydata_sparse:
            raise ValueError("mat: format could not be inferred.")
        return B

    def _matmat(A, b, warn: bool = True) -> pyct.NDArray:
        # A: (M, N) dense/sparse
        # b: (..., N) dense
        # out: (..., M) dense
        if (A.dtype != b.dtype) and warn:
            msg = "Computation may not be performed at the requested precision."
            warnings.warn(msg, pycuw.PrecisionWarning)

        M, N = A.shape
        sh_out = (*b.shape[:-1], M)
        b = b.reshape((-1, N)).T  # (N, (...).prod)
        out = A.dot(b)  # (M, (...).prod)
        return out.T.reshape(sh_out).astype(b.dtype, copy=False)

    @pycrt.enforce_precision(i="arr")
    def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
        return _matmat(_._mat, arr, warn=_._enable_warnings)

    @pycrt.enforce_precision(i="arr")
    def op_adjoint(_, arr: pyct.NDArray) -> pyct.NDArray:
        return _matmat(_._mat.T, arr, warn=_._enable_warnings)

    def op_asarray(
        _,
        xp: pyct.ArrayModule = np,
        dtype: pyct.DType = None,
    ) -> pyct.NDArray:
        if dtype is None:
            dtype = pycrt.getPrecision().value
        try:
            # dense arrays
            pycu.get_array_module(_._mat)
            A = _._mat
        except:
            # sparse arrays
            A = _._mat.toarray()
        return xp.array(A, dtype=dtype)

    def op_trace(_, **kwargs) -> pyct.Real:
        if _.dim != _.codim:
            raise NotImplementedError
        else:
            try:
                # dense arrays
                pycu.get_array_module(_._mat)
                return float(_._mat.trace())
            except:
                # sparse arrays
                return pyca.SquareOp.trace(_, **kwargs)

    def op_lipschitz(_, **kwargs) -> pyct.Real:
        # We want to piggy-back onto Lin[Op,Func].lipschitz() to compute the Lipschitz constant L.
        # Problem: LinOp.lipschitz() relies on svdvals() or hutchpp() to compute L, and they take
        # different parameters to do computations on the GPU.
        # Solution:
        # * we add the relevant kwargs before calling the LinOp.lipschitz() + drop all unrecognized
        #   kwargs there as needed.
        # * similarly for LinFunc.lipschitz().
        xp = pycu.get_array_module(_._mat)
        kwargs.update(xp=xp)
        if pycd.CUPY_ENABLED:
            import cupy as cp

            if xp == cp:
                kwargs.update(gpu=True)
        if _.codim == 1:
            L = pyca.LinFunc.lipschitz(_, xp=xp)
        else:
            L = _.__class__.lipschitz(_, **kwargs)
        return L

    op = cls(shape=mat.shape)
    op._mat = _standard_form(mat)
    op._enable_warnings = bool(enable_warnings)
    op.apply = types.MethodType(op_apply, op)
    op.adjoint = types.MethodType(op_adjoint, op)
    op.asarray = types.MethodType(op_asarray, op)
    op.lipschitz = types.MethodType(op_lipschitz, op)
    op.trace = types.MethodType(op_trace, op)
    return op
