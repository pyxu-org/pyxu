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
]


class _NormBall(pycofn.ShiftLossMixin, pyca.ProxFunc):
    def __init__(
        self,
        dim: pyct.Integer,
        ord: pyct.Integer,
        radius: pyct.Real,
    ):
        super().__init__(shape=(1, dim))
        self._ord = ord
        self._radius = float(radius)
        self._lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        norm = pylinalg.norm(arr, ord=self._ord, axis=-1, keepdims=True)
        cast = lambda _: np.array(_, dtype=arr.dtype)[()]

        xp = pycu.get_array_module(arr)
        out = xp.where(norm <= self._radius, cast(0), cast(np.inf))
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
