import collections.abc as cabc
import functools as ft
import typing as typ
import warnings

import numpy as np

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


class IdentityOp(pyca.PosDefOp, pyca.UnitOp, pyca.OrthProjOp):
    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set()
        for klass in cls.__bases__:
            p |= klass.properties()
        return frozenset(p)

    def __init__(self, dim: pyct.Integer):
        pyca.PosDefOp.__init__(self, shape=(dim, dim))
        pyca.UnitOp.__init__(self, shape=(dim, dim))
        pyca.OrthProjOp.__init__(self, shape=(dim, dim))

        # Use methods from UnitOp/OrthProjOp as needed.
        # Others are delegated to PosDefOp automatically.
        self.pinv = ft.partial(pyca.UnitOp.pinv, self)
        self.dagger = ft.partial(pyca.UnitOp.dagger, self)
        self.lipschitz = ft.partial(pyca.OrthProjOp.lipschitz, self)
        self.gram = ft.partial(pyca.OrthProjOp.gram, self)
        self.cogram = ft.partial(pyca.OrthProjOp.cogram, self)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr

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
        def op_apply(cst, _, arr):
            out = arr.copy()
            out *= cst
            return out

        def op_trace(cst, _, **kwargs):
            out = cst * _.codim
            return out

        klass = pyca.PosDefOp if (cst > 0) else pyca.SelfAdjointOp
        op = klass(shape=(dim, dim))
        op._lipschitz = abs(cst)
        op.apply = ft.partial(op_apply, cst, op)
        op.trace = ft.partial(op_trace, cst, op)

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
        return HomothetyOp(cst=vec.item(), dim=1)
    else:
        xp = pycu.get_array_module(vec)
        if pycu.compute(xp.allclose(vec, 0)):
            op = NullOp(shape=(dim, dim))
        elif pycu.compute(xp.allclose(vec, 1)):
            op = IdentityOp(dim=dim)
        else:  # build PosDef or SelfAdjointOp
            enable_warnings = bool(enable_warnings)

            @pycrt.enforce_precision(i="arr")
            def op_apply(vec, _, arr):
                if (vec.dtype != arr.dtype) and enable_warnings:
                    msg = "Computation may not be performed at the requested precision."
                    warnings.warn(msg, UserWarning)
                out = arr.copy()
                out *= vec
                return out

            def op_asarray(vec, _, **kwargs) -> pyct.NDArray:
                dtype = kwargs.pop("dtype", pycrt.getPrecision().value)
                xp = kwargs.pop("xp", np)
                A = xp.diag(vec).astype(dtype, copy=False)
                return A

            def op_gram(vec, _):
                return DiagonalOp(vec=vec**2, enable_warnings=enable_warnings)

            def op_trace(vec, _, **kwargs):
                return vec.sum().item()

            klass = pyca.PosDefOp if pycu.compute(xp.all(vec > 0)) else pyca.SelfAdjointOp
            op = klass(shape=(dim, dim))
            op._lipschitz = pycu.compute(xp.abs(vec).max())
            op.apply = ft.partial(op_apply, vec, op)
            op.asarray = ft.partial(op_asarray, vec, op)
            op.gram = ft.partial(op_gram, vec, op)
            op.cogram = op.gram
            op.trace = ft.partial(op_trace, vec, op)

        # IdentityOp(dim>1) cannot be squeezed since it doesn't fall into a single core-operator
        # category.
        return op._squeeze() if (op.codim == 1) else op


class ExplicitLinOp(pyca.LinOp):
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

    Notes
    -----
    :py:class:`~pycsou.operator.linop.base.ExplicitLinOp` instances are **not arraymodule-agnostic**:
    they will only work with NDArrays belonging to the same array module as ``mat``.
    Moreover, inner computations may cast input arrays when the precision of ``mat`` does not match
    the user-requested precision.
    If such a situation occurs, a warning is raised.
    """

    def __init__(
        self,
        mat: typ.Union[pyct.NDArray, pyct.SparseArray],
        enable_warnings: bool = True,
    ):
        r"""
        Parameters
        ----------
        mat: pyct.NDArray | pyct.SparseArray
            (M, N) matrix generator.
            The input array can be *dense* or *sparse*.
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.
        """
        super().__init__(shape=mat.shape)
        self.mat = self._standard_form(mat)
        self._enable_warnings = bool(enable_warnings)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self._matmat(self.mat, arr.T).T

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self._matmat(self.mat.T, arr.T).T

    @property
    def T(self) -> pyct.OpT:
        return ExplicitLinOp(self.mat.T)

    def asarray(
        self,
        xp: pyct.ArrayModule = np,
        dtype: pyct.DType = None,
    ) -> pyct.NDArray:
        if dtype is None:
            dtype = pycrt.getPrecision().value
        return xp.array(self.mat, dtype=dtype)

    @staticmethod
    def _standard_form(A):
        fail_dense = False
        try:
            pycu.get_array_module(A)
            B = A
        except:
            fail_dense = True

        fail_sparse = False
        try:
            B = A.tocsr()
        except:
            fail_sparse = True

        if fail_dense and fail_sparse:
            raise ValueError("mat: format could not be inferred.")
        return B

    @staticmethod
    def _matmat(A, b, warn: bool = True) -> pyct.NDArray:
        # A: (M, N) dense/sparse
        # b: (N, [,Q]) dense
        # out: (M [,Q]) dense
        if (A.dtype != b.dtype) and warn:
            msg = "Computation may not be performed at the requested precision."
            warnings.warn(msg, UserWarning)
        return A.dot(b)


def ExplicitLinFunc(
    vec: pyct.NDArray,
    enable_warnings: bool = True,
) -> pyca.LinFunc:
    r"""
    Build a linear functional from its vectorial representation.

    Given a vector :math:`\mathbf{z}\in\mathbb{R}^N`, the *explicit linear functional* associated to
    :math:`\mathbf{z}` is defined as

    .. math::

       f_\mathbf{z}(\mathbf{x})
       =
       \mathbf{z}^T\mathbf{x},
       \qquad
       \forall \mathbf{x}\in\mathbb{R}^N,

    with adjoint given by:

    .. math::

       f^\ast_\mathbf{z}(\alpha)
       =
       \alpha\mathbf{z},
       \qquad
       \forall \alpha\in\mathbb{R}.

    The vector :math:`\mathbf{z}` is called the *vectorial representation* of the linear functional
    :math:`f_\mathbf{z}`.
    The lipschitz constant of explicit linear functionals is trivially given by
    :math:`\|\mathbf{z}\|_2`.

    Notes
    -----
    :py:func:`~pycsou.operator.linop.base.ExplicitLinFunc` instances are **not arraymodule-agnostic**:
    they will only work with NDArrays belonging to the same array module as ``vec``.
    Moreover, inner computations may cast input arrays when the precision of ``vec`` does not match
    the user-requested precision.
    If such a situation occurs, a warning is raised.

    Parameters
    ----------
    vec: pyct.NDArray
        (N,) generator.
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mis-match issues.
    """
    assert len(vec) == np.prod(vec.shape), f"vec: {vec.shape} is not a LinFunc generator."
    op = ExplicitLinOp(
        mat=vec.reshape((1, -1)),
        enable_warnings=enable_warnings,
    )._squeeze()
    return op
