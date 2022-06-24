import typing as typ
import warnings

import numpy as np

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct


class IdentityOp(pyca.PosDefOp, pyca.UnitOp):
    r"""
    Identity operator :math:`\mathrm{Id}`.
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr


class NullOp(pyca.LinOp):
    r"""
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


class NullFunc(NullOp, pyca.LinFunc):
    r"""
    Null functional.

    This functional maps any input vector on the null scalar.
    """

    def __init__(self):
        super().__init__(shape=(1, None))

    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    @pycrt.enforce_precision(i="arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        return arr


class HomothetyOp(pyca.SelfAdjointOp):
    r"""
    Scaling operator.
    """

    def __init__(self, cst: pyct.Real, dim: pyct.Integer):
        r"""
        Parameters
        ----------
        cst: pyct.Real
            Scaling factor.
        dim: pyct.Integer
            Dimension of the domain.
        """
        super().__init__(shape=(dim, dim))
        assert isinstance(cst, pyct.Real), f"cst: expected real, got {cst}."
        self._cst = cst
        self._lipschitz = abs(cst)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = arr.copy()
        out *= self._cst
        return out


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
            (M,N) matrix generator.
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
    def T(self) -> pyct.MapT:
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


class ExplicitLinFunc(pyca.LinFunc):
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
    :py:class:`~pycsou.operator.linop.base.ExplicitLinFunc` instances are **not arraymodule-agnostic**:
    they will only work with NDArrays belonging to the same array module as ``vec``.
    Moreover, inner computations may cast input arrays when the precision of ``vec`` does not match
    the user-requested precision.
    If such a situation occurs, a warning is raised.
    """

    def __init__(
        self,
        vec: pyct.NDArray,
        enable_warnings: bool = True,
    ):
        r"""
        Parameters
        ----------
        vec: pyct.NDArray
            (N,) generator.
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.
        """
        assert vec.size == np.prod(vec.shape), f"vec: {vec.shape} is not a LinFunc generator."
        self._op = ExplicitLinOp(
            mat=vec.reshape((1, -1)),
            enable_warnings=enable_warnings,
        )
        super().__init__(shape=self._op.shape)
        self.apply = self._op.apply

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        A = self._op.mat
        if arr.ndim == 1:
            A = A.squeeze(axis=0)

        xp = pycu.get_array_module(arr)
        return xp.broadcast_to(A, arr.shape)
