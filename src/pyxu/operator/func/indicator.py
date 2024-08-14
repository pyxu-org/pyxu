import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.operator.func.norm as pxf
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "L1Ball",
    "L2Ball",
    "LInfinityBall",
    "PositiveOrthant",
    "HyperSlab",
    "RangeSet",
]


class _IndicatorFunction(pxa.ProxFunc):
    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=1,
        )
        self.lipschitz = np.inf

    @staticmethod
    def _bool2indicator(x: pxt.NDArray, dtype: pxt.DType) -> pxt.NDArray:
        # x: NDarray[bool]
        # y: NDarray[(0, \inf), dtype]
        xp = pxu.get_array_module(x)
        cast = lambda _: np.array(_, dtype=dtype)[()]
        y = xp.where(x, cast(0), cast(np.inf))
        return y


class _NormBall(_IndicatorFunction):
    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        ord: pxt.Integer,
        radius: pxt.Real,
    ):
        super().__init__(dim_shape=dim_shape)
        self._ord = ord
        self._radius = float(radius)

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        from pyxu.opt.stop import _norm

        norm = _norm(arr, ord=self._ord, rank=self.dim_rank)  # (..., 1)
        out = self._bool2indicator(norm <= self._radius, arr.dtype)
        return out

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        klass = {  # class of proximal operator to use
            1: pxf.LInfinityNorm,
            2: pxf.L2Norm,
            np.inf: pxf.L1Norm,
        }[self._ord]
        op = klass(dim_shape=self.dim_shape)

        out = arr.copy()
        out -= op.prox(arr, tau=self._radius)
        return out


def L1Ball(dim_shape: pxt.NDArrayShape, radius: pxt.Real = 1) -> pxt.OpT:
    r"""
    Indicator function of the :math:`\ell_{1}`-ball.

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
    dim_shape: NDArrayShape
    radius: Real
        Ball radius. (Default: unit ball.)

    Returns
    -------
    op: OpT

    Note
    ----
    * Computing :py:meth:`~pyxu.abc.ProxFunc.prox` is unavailable with DASK inputs.
      (Inefficient exact solution at scale.)
    """
    op = _NormBall(dim_shape=dim_shape, ord=1, radius=radius)
    op._name = "L1Ball"
    return op


def L2Ball(dim_shape: pxt.NDArrayShape, radius: pxt.Real = 1) -> pxt.OpT:
    r"""
    Indicator function of the :math:`\ell_{2}`-ball.

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
    dim_shape: NDArrayShape
    radius: Real
        Ball radius. (Default: unit ball.)

    Returns
    -------
    op: OpT
    """
    op = _NormBall(dim_shape=dim_shape, ord=2, radius=radius)
    op._name = "L2Ball"
    return op


def LInfinityBall(dim_shape: pxt.NDArrayShape, radius: pxt.Real = 1) -> pxt.OpT:
    r"""
    Indicator function of the :math:`\ell_{\infty}`-ball.

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
    dim_shape: NDArrayShape
    radius: Real
        Ball radius. (Default: unit ball.)

    Returns
    -------
    op: OpT
    """
    op = _NormBall(dim_shape=dim_shape, ord=np.inf, radius=radius)
    op._name = "LInfinityBall"
    return op


class PositiveOrthant(_IndicatorFunction):
    r"""
    Indicator function of the positive orthant.

    .. math::

       \iota_{+}(\mathbf{x})
       :=
       \begin{cases}
           0 & \min{\mathbf{x}} \ge 0,\\
           \infty & \text{otherwise}.
       \end{cases}

    .. math::

       \text{prox}_{\tau\, \iota_{+}}(\mathbf{x})
       :=
       \max(\mathbf{x}, \mathbf{0})
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(dim_shape=dim_shape)

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        axis = tuple(range(-self.dim_rank, 0))
        in_set = (arr >= 0).all(axis=axis)[..., np.newaxis]
        out = self._bool2indicator(in_set, arr.dtype)
        return out

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
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

    def __init__(self, a: pxa.LinFunc, lb: pxt.Real, ub: pxt.Real):
        """
        Parameters
        ----------
        A: ~pyxu.abc.operator.LinFunc
            Linear functional with domain (M1,...,MD).
        lb: Real
            Lower bound.
        ub: Real
            Upper bound.
        """
        assert lb < ub
        super().__init__(dim_shape=a.dim_shape)

        # Everything happens internally in normalized coordinates.
        _norm = a.lipschitz  # \norm{a}{2}
        self._a = a / _norm
        self._l = lb / _norm
        self._u = ub / _norm

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        y = self._a.apply(arr)  # (..., 1)
        in_set = ((self._l <= y) & (y <= self._u)).all(axis=-1, keepdims=True)
        out = self._bool2indicator(in_set, arr.dtype)  # (..., 1)
        return out

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)

        a = self._a.adjoint(xp.ones(1, dtype=arr.dtype))  # (M1,...,MD)
        expand = (np.newaxis,) * (self.dim_rank - 1)
        y = self._a.apply(arr)[..., *expand]  # (..., 1,...,1)
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

    def __init__(self, A: pxa.LinOp):
        super().__init__(dim_shape=A.codim_shape)
        self._A = A

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        # I'm in range(A) if prox(x)==x.
        axis = tuple(range(-self.dim_rank, 0))
        y = self.prox(arr, tau=1)
        in_set = self.isclose(y, arr).all(axis=axis)  # (...,)
        out = self._bool2indicator(in_set[..., np.newaxis], arr.dtype)
        return out  # (..., 1)

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        y = self._A.pinv(arr, damp=0)
        out = self._A.apply(y)
        return out

    @staticmethod
    def isclose(a: pxt.NDArray, b: pxt.NDArray) -> pxt.NDArray:
        """
        Equivalent of `xp.isclose`, but where atol is automatically chosen based on input's `dtype`.
        """
        atol = {
            pxrt.Width.SINGLE.value: 2e-4,
            pxrt.Width.DOUBLE.value: 1e-8,
        }
        # Numbers obtained by:
        # * \sum_{k >= (p+1)//2} 2^{-k}, where p=<number of mantissa bits>; then
        # * round up value to 3 significant decimal digits.
        # N_mantissa = [23, 52] for [single, double] respectively.
        xp = pxu.get_array_module(a)
        prec = atol.get(a.dtype, pxrt.Width.DOUBLE.value)  # default only should occur for integer types
        eq = xp.isclose(a, b, atol=prec)
        return eq
