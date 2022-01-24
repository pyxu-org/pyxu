import types
import typing as typ
import warnings

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


class ExplicitLinFunc(pyco.LinFunc):
    r"""
    Build a linear functional from its vectorial representation.

    Given a vector :math:`\mathbf{z}\in\mathbb{R}^N`, the *explicit linear functional* associated to  :math:`\mathbf{z}` is defined as

    .. math::

        f_\mathbf{z}(\mathbf{x})=\mathbf{z}^T\mathbf{x}, \qquad \forall \mathbf{x}\in\mathbb{R}^N.

    The vector :math:`\mathbf{z}` is called the *vectorial representation* of the linear functional :math:`f_\mathbf{z}`.
    The lipschitz constant of explicit linear functionals is trivially given by :math:`\|\mathbf{z}\|_2`.

    Examples
    --------
    >>> from pycsou.linop.base import ExplicitLinFunc
    >>> import numpy as np
    >>> vec = np.ones(10)
    >>> sum_func = ExplicitLinFunc(vec)
    >>> sum_func.shape
    (1, 10)
    >>> np.allclose(sum_func(np.arange(10)), np.sum(np.arange(10)))
    True
    >>> np.allclose(sum_func.adjoint(3), 3 * vec)
    True

    Notes
    -----
    :py:class:`~pycsou.linop.base.ExplicitLinFunc` are **not array module agnostic**: they will only work with input arrays
    belonging to the same array module than the one of the array ``vec`` used to initialize the :py:class:`~pycsou.linop.base.ExplicitLinFunc` object.
    Moreover, while the input/output precisions of the callable methods of :py:class:`~pycsou.linop.base.ExplicitLinFunc` objects are
    guaranteed to match the user-requested precision, the inner computations may force a recast of the input arrays when
    the precision of ``vec`` does not match the user-requested precision. If such a situation occurs, a warning is raised.

    See Also
    --------
    :py:meth:`~pycsou.abc.operator.LinOp.asarray`
        Convert a matrix-free :py:class:`~pycsou.abc.operator.LinFunc` into an :py:class:`~pycsou.linop.base.ExplicitLinFunc`.
    """

    @pycrt.enforce_precision(i="vec")
    def __init__(self, vec: pyct.NDArray, enable_warnings: bool = True):
        r"""

        Parameters
        ----------
        vec: NDArray
            (N,) input. N-D input arrays are flattened. This is the vectorial representation of the linear functional.
        enable_warnings: bool
            If ``True``, the user will be warned in case of mismatching precision issues.

        Notes
        -----
        The input ``vec`` is automatically casted by the decorator :py:func:`~pycsou.runtime.enforce_precision` to the user-requested precision at initialization time.
        Explicit control over the precision of ``vec`` is hence only possible via the context manager :py:class:`~pycsou.runtime.Precision`.
        """
        super(ExplicitLinFunc, self).__init__(shape=(1, vec.size))
        xp = pycu.get_array_module(vec)
        self._vec = vec.copy().reshape(-1)
        self._lipschitz = xp.linalg.norm(vec)
        self._enable_warnings = bool(enable_warnings)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._vec.dtype != pycrt.getPrecision() and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return (self._vec * arr).sum(axis=-1)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._vec.dtype != pycrt.getPrecision() and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return arr * self._vec

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._vec.dtype != pycrt.getPrecision() and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        xp = pycu.get_array_module(arr)
        return xp.broadcast_to(self._vec, arr.shape)

    @pycrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        if self._vec.dtype != pycrt.getPrecision() and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return arr - tau * self._vec


class IdentityOp(pyco.PosDefOp, pyco.SelfAdjointOp, pyco.UnitOp):
    def __init__(self, shape: pyct.Shape):
        pyco.PosDefOp.__init__(self, shape)
        pyco.SelfAdjointOp.__init__(self, shape)
        pyco.UnitOp.__init__(self, shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr


class HomothetyOp(pyco.SelfAdjointOp):
    def __init__(self, cst: pyct.Real, dim: int):
        if not isinstance(cst, pyct.Real):
            raise ValueError(f"cst: expected real number, got {cst}.")
        super(HomothetyOp, self).__init__(shape=(dim, dim))
        self._cst = cst
        self._lipschitz = abs(cst)
        self._diff_lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = arr.copy()
        out *= self._cst
        return out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    def __mul__(self, other):
        op = pyco.Property.__mul__(self, other)
        if isinstance(other, pyco.ProxFunc):
            op.specialize(cast_to=pyco.ProxFunc)
            post_composition_prox = lambda obj, arr, tau: other.prox(arr, self._cst * tau)
            op.prox = types.MethodType(post_composition_prox, op)
            return op


class NullOp(pyco.LinOp):
    def __init__(self, shape: typ.Tuple[int, int]):
        super(NullOp, self).__init__(shape)
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
            (arr.shape[:-1], self.dim),
        )


class NullFunc(NullOp, pyco.LinFunc):
    def __init__(self, dim: typ.Optional[int] = None):
        pyco.LinFunc.__init__(self, shape=(1, dim))
        NullOp.__init__(self, shape=self.shape)

    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    @pycrt.enforce_precision(i="arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        return arr
