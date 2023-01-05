import numpy as np

import pycsou.abc as pyca
import pycsou.math.linalg as pylinalg
import pycsou.operator.func.norm as pycofn
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "L1Ball",
    "L2Ball",
    "LInfinityBall",
    "PositiveOrthant",
    "HyperSlab",
    "RangeSet",
]


class _IndicatorFunction(pycofn.ShiftLossMixin, pyca.ProxFunc):
    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(1, dim))
        self._lipschitz = np.inf

    @staticmethod
    def _bool2indicator(x: pyct.NDArray, dtype: pyct.DType) -> pyct.NDArray:
        # x: NDarray[bool]
        # y: NDarray[(0, \inf), dtype]
        xp = pycu.get_array_module(x)
        cast = lambda _: np.array(_, dtype=dtype)[()]
        y = xp.where(x, cast(0), cast(np.inf))
        return y


class _NormBall(_IndicatorFunction):
    def __init__(
        self,
        dim: pyct.Integer,
        ord: pyct.Integer,
        radius: pyct.Real,
    ):
        super().__init__(dim=dim)
        self._ord = ord
        self._radius = float(radius)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        norm = pylinalg.norm(arr, ord=self._ord, axis=-1, keepdims=True)
        out = self._bool2indicator(norm <= self._radius, arr.dtype)
        return out

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        klass = {  # class of proximal operator to use
            1: pycofn.LInfinityNorm,
            2: pycofn.L2Norm,
            np.inf: pycofn.L1Norm,
        }[self._ord]
        op = klass()

        out = arr.copy()
        out -= op.prox(arr, tau=self._radius)
        return out


def L1Ball(dim: pyct.Integer = None, radius: pyct.Real = 1) -> pyct.OpT:
    r"""
    Indicator function of the :math:`\ell_1`-ball.

    .. math::

       \iota_{1}^{r}(\mathbf{x})
       :=
       \begin{cases}
           0 & \|\mathbf{x}\|_{1} \le r \\
           \infty & \text{otherwise}.
       \end{cases}

    .. math::

       \text{prox}_{\tau\, \iota_{1}^{r}}(\mathbf{x})
       :=
       \mathbf{x} - \text{prox}_{r\, \ell_{\infty}}(\mathbf{x})

    Parameters
    ----------
    dim: pyct.Integer
        Dimension size. (Default: domain-agnostic.)
    radius: pyct.Real
        Ball radius. (Default: unit ball.)

    Returns
    -------
    op: pyct.OpT
    """
    op = _NormBall(dim=dim, ord=1, radius=radius)
    op._name = "L1Ball"
    return op


def L2Ball(dim: pyct.Integer = None, radius: pyct.Real = 1) -> pyct.OpT:
    r"""
    Indicator function of the :math:`\ell_2`-ball.

    .. math::

       \iota_{2}^{r}(\mathbf{x})
       :=
       \begin{cases}
           0 & \|\mathbf{x}\|_{2} \le r \\
           \infty & \text{otherwise}.
       \end{cases}

    .. math::

       \text{prox}_{\tau\, \iota_{2}^{r}}(\mathbf{x})
       :=
       \mathbf{x} - \text{prox}_{r\, \ell_{2}}(\mathbf{x})

    Parameters
    ----------
    dim: pyct.Integer
        Dimension size. (Default: domain-agnostic.)
    radius: pyct.Real
        Ball radius. (Default: unit ball.)

    Returns
    -------
    op: pyct.OpT
    """
    op = _NormBall(dim=dim, ord=2, radius=radius)
    op._name = "L2Ball"
    return op


def LInfinityBall(dim: pyct.Integer = None, radius: pyct.Real = 1) -> pyct.OpT:
    r"""
    Indicator function of the :math:`\ell_\infty`-ball.

    .. math::

       \iota_{\infty}^{r}(\mathbf{x})
       :=
       \begin{cases}
           0 & \|\mathbf{x}\|_{\infty} \le r \\
           \infty & \text{otherwise}.
       \end{cases}

    .. math::

       \text{prox}_{\tau\, \iota_{\infty}^{r}}(\mathbf{x})
       :=
       \mathbf{x} - \text{prox}_{r\, \ell_{1}}(\mathbf{x})

    Parameters
    ----------
    dim: pyct.Integer
        Dimension size. (Default: domain-agnostic.)
    radius: pyct.Real
        Ball radius. (Default: unit ball.)

    Returns
    -------
    op: pyct.OpT
    """
    op = _NormBall(dim=dim, ord=np.inf, radius=radius)
    op._name = "LInfinityBall"
    return op


class PositiveOrthant(_IndicatorFunction):
    r"""
    Indicator function of the positive orthant.

    .. math::

       \iota_{+}(\mathbf{x})
       :=
       \begin{cases}
           0 & \min{\mathbf{x}} \ge 0
           \infty & \text{otherwise}.
       \end{cases}

    .. math::

       \text{prox}_{\tau\, \iota_{+}}(\mathbf{x})
       :=
       \max(\mathbf{x}, \mathbf{0})
    """

    def __init__(self, dim: pyct.Integer = None):
        """
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)
        """
        super().__init__(dim=dim)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        in_set = (arr >= 0).all(axis=-1, keepdims=True)
        out = self._bool2indicator(in_set, arr.dtype)
        return out

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        out = arr.clip(0, None)
        return out


class HyperSlab(_IndicatorFunction):
    r"""
    Indicator function of a hyperslab.

    .. math::

       \iota_{\mathbf{a}}^{l,u}(\mathbf{x})
       :=
       \begin{cases}
           0 & l \le \langle \mathbf{a}, \mathbf{x} \rangle \le u \\
           \infty & \text{otherwise}.
       \end{cases}

    .. math::

       \text{prox}_{\tau\, \iota_{\mathbf{a}}^{l,u}}(\mathbf{x})
       :=
       \begin{cases}
           \mathbf{x} + \frac{l - \langle \mathbf{a}, \mathbf{x} \rangle}{\|\mathbf{a}\|^{2}} \mathbf{a} & \langle \mathbf{a}, \mathbf{x} \rangle < l, \\
           \mathbf{x} + \frac{u - \langle \mathbf{a}, \mathbf{x} \rangle}{\|\mathbf{a}\|^{2}} \mathbf{a} & \langle \mathbf{a}, \mathbf{x} \rangle > u, \\
           \mathbf{x} & \text{otherwise}.
       \end{cases}
    """

    @pycrt.enforce_precision(i=("l", "u"))
    def __init__(self, a: pyca.LinFunc, l: pyct.Real, u: pyct.Real):
        """
        Parameters
        ----------
        A: pyca.LinFunc
            (N,) operator
        l: pyct.Real
            Lower bound
        u: pyct.Real
            Upper bound
        """
        assert l < u
        super().__init__(dim=a.dim)

        # Everything happens internally in normalized coordinates.
        _norm = a.lipschitz()  # \norm{a}{2}
        self._a = a / _norm
        self._l = l / _norm
        self._u = u / _norm

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = self._a.apply(arr)
        in_set = ((self._l <= y) & (y <= self._u)).all(axis=-1, keepdims=True)
        out = self._bool2indicator(in_set, arr.dtype)
        return out

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)

        a = self._a.adjoint(xp.ones(1, dtype=arr.dtype))  # slab direction
        y = self._a.apply(arr)
        out = arr.copy()

        l_corr = self._l - y
        l_corr[l_corr <= 0] = 0
        out += l_corr * a

        u_corr = self._u - y
        u_corr[u_corr >= 0] = 0
        out += u_corr * a

        return out


class RangeSet(_IndicatorFunction):
    r"""
    Indicator function of a range set.

    .. math::

       \iota_{\mathbf{A}}^{R}(\mathbf{x})
       :=
       \begin{cases}
           0 & \mathbf{x} \in \text{span}(\mathbf{A}) \\
           \infty & \text{otherwise}.
       \end{cases}

    .. math::

       \text{prox}_{\tau\, \iota_{\mathbf{A}}^{R}}(\mathbf{x})
       :=
       \mathbf{A} (\mathbf{A}^{T} \mathbf{A})^{-1} \mathbf{A}^{T} \mathbf{x}.
    """

    def __init__(self, A: pyca.LinOp):
        """
        Parameters
        ----------
        A: pyca.LinOp
            (M, N) operator
        """
        super().__init__(dim=A.codim)
        self._A = A

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        # I'm in range(A) if prox(x)==x.
        xp = pycu.get_array_module(arr)
        in_set = xp.isclose(self.prox(arr, tau=1), arr).all(axis=-1, keepdims=True)
        out = self._bool2indicator(in_set, arr.dtype)
        return out

    @pycrt.enforce_precision(i=("arr", "tau"))
    @pycu.vectorize(i="arr")  # see comment below
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        # [2023.01.03 Sepand]
        #
        # When more than one input is provided, `A.pinv(arr)` may sometimes return NaNs.
        # The problem is pinpointed to the instruction below from CG():
        #     alpha = rr / (p * Ap).sum(axis=-1, keepdims=True)
        #
        # Oddly the problem does not occur when `arr` is 1D.
        # Could not figure out why the CG line breaks down at times with multi-inputs.
        #
        # Temporary(/Permanent?) workaround: use @vectorize() to evaluate prox calls one at a time.
        y = self._A.pinv(arr)
        out = self._A.apply(y)
        return out
