import types
import typing as typ
import warnings

import numpy as np
import scipy.sparse as scisp
import sparse as sp

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

if pycd.CUPY_ENABLED:
    import cupy as cp
    import cupyx.scipy.sparse as cusp


class ExplicitLinFunc(pyco.LinFunc):
    r"""
    Build a linear functional from its vectorial representation.

    Given a vector :math:`\mathbf{z}\in\mathbb{R}^N`, the *explicit linear functional* associated to  :math:`\mathbf{z}` is defined as

    .. math::

        f_\mathbf{z}(\mathbf{x})=\mathbf{z}^T\mathbf{x}, \qquad \forall \mathbf{x}\in\mathbb{R}^N,

    with adjoint given by:

    .. math::

        f^\ast_\mathbf{z}(\alpha)=\alpha\mathbf{z}, \qquad \forall \alpha\in\mathbb{R}.

    The vector :math:`\mathbf{z}` is called the *vectorial representation* of the linear functional :math:`f_\mathbf{z}`.
    The lipschitz constant of explicit linear functionals is trivially given by :math:`\|\mathbf{z}\|_2`.

    Examples
    --------
    >>> from pycsou.operator.linop.base import ExplicitLinFunc
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
    :py:class:`~pycsou.operator.linop.base.ExplicitLinFunc` instances are **not array module agnostic**: they will only work with input arrays
    belonging to the same array module than the one of the array ``vec`` used to initialize the :py:class:`~pycsou.operator.linop.base.ExplicitLinFunc` object.
    Moreover, while the input/output precisions of the callable methods of :py:class:`~pycsou.operator.linop.base.ExplicitLinFunc` objects are
    guaranteed to match the user-requested precision, the inner computations may force a recast of the input arrays when
    the precision of ``vec`` does not match the user-requested precision. If such a situation occurs, a warning is raised.

    See Also
    --------
    :py:meth:`~pycsou.abc.operator.LinOp.asarray`
        Convert a matrix-free :py:class:`~pycsou.abc.operator.LinFunc` into an :py:class:`~pycsou.operator.linop.base.ExplicitLinFunc`.
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
        Explicit control over the precision of ``vec`` is hence only possible via the context manager :py:class:`~pycsou.runtime.Precision`:

        >>> from pycsou.operator.linop.base import ExplicitLinFunc
        >>> import pycsou.runtime as pycrt
        >>> import numpy as np
        >>> vec = np.ones(10) # This array will be recasted to requested precision.
        >>> with pycrt.Precision(pycrt.Width.HALF):
        ...     sum_func = ExplicitLinFunc(vec) # The init function of ExplicitLinFunc stores ``vec`` at the requested precision.
        ...     # Further calculations with sum_func. Within this context mismatching precisions are avoided.

        """
        super(ExplicitLinFunc, self).__init__(shape=(1, vec.size))
        xp = pycu.get_array_module(vec)
        self.vec = vec.copy().reshape(-1)
        self._lipschitz = xp.linalg.norm(vec)
        self._enable_warnings = bool(enable_warnings)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self.vec.dtype != pycrt.getPrecision().value and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return (self.vec * arr).sum(axis=-1)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self.vec.dtype != pycrt.getPrecision().value and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return arr * self.vec

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        if self.vec.dtype != pycrt.getPrecision().value and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return xp.broadcast_to(self.vec, arr.shape)

    @pycrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        if self.vec.dtype != pycrt.getPrecision().value and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return arr - tau * self.vec


class IdentityOp(pyco.PosDefOp, pyco.UnitOp):
    r"""
    Identity operator :math:`\mathrm{Id}`.
    """

    def __init__(self, shape: pyct.Shape):
        pyco.PosDefOp.__init__(self, shape)
        pyco.UnitOp.__init__(self, shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr


class HomothetyOp(pyco.SelfAdjointOp):
    r"""
    Scaling operator.

    This operators performs a scaling by real factor ``cst``. Its Lipschitz constant is given by ``abs(cst)``.
    """

    def __init__(self, cst: pyct.Real, dim: int):
        r"""

        Parameters
        ----------
        cst: Real
            Scaling factor.
        dim: int
            Dimension of the domain.

        Raises
        ------
        ValueError
            If ``cst`` is not real.
        """
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
    r"""
    Null operator.

    This operator maps any input vector on the null vector. Its Lipschitz constant is zero.
    """

    def __init__(self, shape: typ.Tuple[int, int]):
        r"""

        Parameters
        ----------
        shape: tuple(int, int)
            Shape of the null operator.
        """
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
    r"""
    Null functional.

    This functional maps any input vector on the null scalar. Its Lipschitz constant is zero.
    """

    def __init__(self, dim: typ.Optional[int] = None):
        r"""

        Parameters
        ----------
        dim: Optional[int]
            Dimension of the domain. Set ``dim=None`` for making the functional domain-agnostic.
        """
        pyco.LinFunc.__init__(self, shape=(1, dim))
        NullOp.__init__(self, shape=self.shape)

    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    @pycrt.enforce_precision(i="arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        return arr


class ExplicitLinOp(pyco.LinOp):
    r"""
    Build a linear operator from its matrix representation.

    Given a matrix :math:`\mathbf{A}\in\mathbb{R}^{M\times N}`, the *explicit linear operator* associated to  :math:`\mathbf{A}` is defined as

    .. math::

        f_\mathbf{A}(\mathbf{x})=\mathbf{A}\mathbf{x}, \qquad \forall \mathbf{x}\in\mathbb{R}^N,

    with adjoint given by:

    .. math::

        f^\ast_\mathbf{A}(\mathbf{z})=\mathbf{A}^T\mathbf{z}, \qquad \forall \mathbf{z}\in\mathbb{R}^M.

    Examples
    --------
    >>> from pycsou.operator.linop.base import ExplicitLinOp
    >>> import numpy as np
    >>> mat = np.arange(10).reshape(2,5)
    >>> f = ExplicitLinOp(mat)
    >>> f.shape
    (2, 5)
    >>> np.allclose(f(np.arange(5)), mat @ np.arange(5))
    True
    >>> np.allclose(f.adjoint(np.arange(2)), mat.T @ np.arange(2))
    True

    Notes
    -----
    :py:class:`~pycsou.operator.linop.base.ExplicitLinOp` instances are **not array module agnostic**: they will only work with input arrays
    belonging to the same array module than the one of the array ``mat`` used to initialize the :py:class:`~pycsou.operator.linop.base.ExplicitLinOp` object.
    Moreover, while the input/output precisions of the callable methods of :py:class:`~pycsou.operator.linop.base.ExplicitLinOp` objects are
    guaranteed to match the user-requested precision, the inner computations may force a recast of the input arrays when
    the precision of ``mat`` does not match the user-requested precision. If such a situation occurs, a warning is raised.

    See Also
    --------
    :py:class:`~pycsou.operator.linop.base.ExplicitLinFunc`
    :py:meth:`~pycsou.abc.operator.LinOp.asarray`
        Convert a matrix-free :py:class:`~pycsou.abc.operator.LinOp` into an :py:class:`~pycsou.operator.linop.base.ExplicitLinOp`.
    """

    @pycrt.enforce_precision(i="mat")
    def __init__(self, mat: typ.Union[pyct.NDArray, pyct.SparseArray], enable_warnings: bool = True):
        r"""

        Parameters
        ----------
        mat: NDArray | SparseArray
            (M,N) input array. This is the matrix representation of the linear operator. The input array can be *dense* or *sparse*.
            In the latter case, it must be an instance of one of the following sparse array classes: :py:class:`sparse.SparseArray`,
            :py:class:`scipy.sparse.spmatrix`, :py:class:`cupyx.scipy.sparse.spmatrix`. Note that
        enable_warnings: bool
            If ``True``, the user will be warned in case of mismatching precision issues.

        Notes
        -----
        The input ``mat`` is automatically casted by the decorator :py:func:`~pycsou.runtime.enforce_precision` to the user-requested precision at initialization time.
        Explicit control over the precision of ``mat`` is hence only possible via the context manager :py:class:`~pycsou.runtime.Precision`:

        >>> from pycsou.operator.linop.base import ExplicitLinOp
        >>> import pycsou.runtime as pycrt
        >>> import numpy as np
        >>> mat = np.arange(10).reshape(2,5) # This array will be recasted to requested precision.
        >>> with pycrt.Precision(pycrt.Width.HALF):
        ...     f = ExplicitLinOp(mat) # The init function of ExplicitLinOp stores ``mat`` at the requested precision.
        ...     # Further calculations with f. Within this context mismatching precisions are avoided.

        Note moreover that sparse inputs with type :py:class:`scipy.sparse.spmatrix` are automatically casted as :py:class:`sparse.SparseArray` which should be
        the preferred class for representing sparse arrays. Finally, the default sparse storage format is ``'csr'`` (for fast matrix-vector multiplications).
        """
        super(ExplicitLinOp, self).__init__(shape=mat.shape)
        self.mat = self._coerce_input(mat)
        self._enable_warnings = bool(enable_warnings)
        self._module_name = self._get_module_name(mat)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self.mat.dtype != pycrt.getPrecision().value and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        if self._module_name == "cupyx":
            stack_shape = arr.shape[:-1]
            return cp.asarray(self.mat.dot(arr.reshape(-1, self.dim).transpose()).transpose()).reshape(
                *stack_shape, self.codim
            )
        else:
            return self.mat.__matmul__(arr[..., None]).squeeze(axis=-1)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self.mat.dtype != pycrt.getPrecision().value and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        if self._module_name == "cupyx":
            stack_shape = arr.shape[:-1]
            return cp.asarray(self.mat.T.dot(arr.reshape(-1, self.dim).transpose()).transpose()).reshape(
                *stack_shape, self.codim
            )
        else:
            return self.mat.transpose().__matmul__(arr[..., None]).squeeze(axis=-1)

    def lipschitz(self, recompute: bool = False, algo: str = "svds", **kwargs) -> float:
        r"""
        Same functionality as :py:meth:`~pycsou.abc.operator.LinOp.lipschitz` but the case ``algo='fro'`` is handled
        differently: the Frobenius norm of the operator is directly computed from its matrix representation rather than with the Hutch++ algorithm.
        """
        kwargs.pop("gpu", None)
        gpu = True if self._module_name in ["cupy", "cupyx"] else False
        if recompute or (self._lipschitz == np.inf):
            if algo == "fro":
                if self._module_name in ["sparse", "cupyx"]:
                    data = self.mat.asformat("coo").data.squeeze()
                    xp = pycu.get_array_module(data)
                    self._lipschitz = xp.linalg.norm(data, ord=algo)
                else:
                    xp = pycu.get_array_module(self.mat)
                    self._lipschitz = xp.linalg.norm(self.mat, ord=algo)
            else:
                self._lipschitz = pyco.LinOp.lipschitz(self, recompute=recompute, algo=algo, gpu=gpu, **kwargs)
        return self._lipschitz

    def svdvals(self, k: int, which="LM", **kwargs) -> pyct.NDArray:
        kwargs.pop("gpu", None)
        gpu = True if self._module_name in ["cupy", "cupyx"] else False
        return pyco.LinOp.svdvals(self, k=k, which=which, gpu=gpu, **kwargs)

    def _coerce_input(
        self, mat: typ.Union[pyct.NDArray, pyct.SparseArray]
    ) -> typ.Union[pyct.NDArray, pyct.SparseArray]:
        assert type(mat) in pycd.supported_array_types() + pycd.supported_sparse_types()
        if pycd.CUPY_ENABLED and isinstance(mat, cusp.spmatrix):
            out = mat.tocsr(copy=True)
        elif isinstance(mat, scisp.spmatrix):
            out = sp.GCXS.from_scipy_sparse(mat)
        elif isinstance(mat, sp.SparseArray):
            assert mat.ndim == 2
            out = mat.asformat("gcxs")
        else:
            assert mat.ndim == 2
            out = mat.copy()
        return out

    def _get_module_name(self, arr: typ.Union[pyct.NDArray, pyct.SparseArray]) -> str:
        if pycd.CUPY_ENABLED and isinstance(arr, cusp.spmatrix):
            return "cupyx"
        else:
            array_module = pycu.get_array_module(arr, fallback=sp)
            return array_module.__name__

    def trace(self, **kwargs) -> float:
        return self.mat.trace().item()
