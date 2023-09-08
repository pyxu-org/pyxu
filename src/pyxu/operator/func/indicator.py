import collections.abc as cabc
import functools
import operator

import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.math as pxm
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
    "AffineSet",
    "ConvexSetIntersection",
]


class _IndicatorFunction(pxf._ShiftLossMixin, pxa.ProxFunc):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(1, dim))
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
        dim: pxt.Integer,
        ord: pxt.Integer,
        radius: pxt.Real,
    ):
        super().__init__(dim=dim)
        self._ord = ord
        self._radius = float(radius)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        norm = pxm.norm(arr, ord=self._ord, axis=-1, keepdims=True)
        out = self._bool2indicator(norm <= self._radius, arr.dtype)
        return out

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        klass = {  # class of proximal operator to use
            1: pxf.LInfinityNorm,
            2: pxf.L2Norm,
            np.inf: pxf.L1Norm,
        }[self._ord]
        op = klass(dim=self.dim)

        out = arr.copy()
        out -= op.prox(arr, tau=self._radius)
        return out


def L1Ball(dim: pxt.Integer, radius: pxt.Real = 1) -> pxt.OpT:
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
    dim: Integer
    radius: Real
        Ball radius. (Default: unit ball.)

    Returns
    -------
    op: OpT
    """
    op = _NormBall(dim=dim, ord=1, radius=radius)
    op._name = "L1Ball"
    return op


def L2Ball(dim: pxt.Integer, radius: pxt.Real = 1) -> pxt.OpT:
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
    dim: Integer
    radius: Real
        Ball radius. (Default: unit ball.)

    Returns
    -------
    op: OpT
    """
    op = _NormBall(dim=dim, ord=2, radius=radius)
    op._name = "L2Ball"
    return op


def LInfinityBall(dim: pxt.Integer, radius: pxt.Real = 1) -> pxt.OpT:
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
    dim: Integer
    radius: Real
        Ball radius. (Default: unit ball.)

    Returns
    -------
    op: OpT
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
           0 & \min{\mathbf{x}} \ge 0,\\
           \infty & \text{otherwise}.
       \end{cases}

    .. math::

       \text{prox}_{\tau\, \iota_{+}}(\mathbf{x})
       :=
       \max(\mathbf{x}, \mathbf{0})
    """

    def __init__(self, dim: pxt.Integer):
        super().__init__(dim=dim)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        in_set = (arr >= 0).all(axis=-1, keepdims=True)
        out = self._bool2indicator(in_set, arr.dtype)
        return out

    @pxrt.enforce_precision(i=("arr", "tau"))
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

    @pxrt.enforce_precision(i=("lb", "ub"))
    def __init__(self, a: pxa.LinFunc, lb: pxt.Real, ub: pxt.Real):
        """
        Parameters
        ----------
        A: ~pyxu.abc.operator.LinFunc
            (N,) operator
        lb: Real
            Lower bound
        ub: Real
            Upper bound
        """
        assert lb < ub
        super().__init__(dim=a.dim)

        # Everything happens internally in normalized coordinates.
        _norm = a.lipschitz  # \norm{a}{2}
        self._a = a / _norm
        self._l = lb / _norm
        self._u = ub / _norm

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        y = self._a.apply(arr)
        in_set = ((self._l <= y) & (y <= self._u)).all(axis=-1, keepdims=True)
        out = self._bool2indicator(in_set, arr.dtype)
        return out

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)

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

    def __init__(self, A: pxa.LinOp):
        """
        Parameters
        ----------
        A: ~pyxu.abc.operator.LinOp
            (M, N) operator
        """
        super().__init__(dim=A.codim)
        self._A = A

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        # I'm in range(A) if prox(x)==x.
        xp = pxu.get_array_module(arr)
        in_set = xp.isclose(self.prox(arr, tau=1), arr).all(axis=-1, keepdims=True)
        out = self._bool2indicator(in_set, arr.dtype)
        return out

    @pxrt.enforce_precision(i=("arr", "tau"))
    @pxu.vectorize(i="arr")  # see comment below
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
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
        y = self._A.pinv(arr, damp=0)
        out = self._A.apply(y)
        return out


class AffineSet(_IndicatorFunction):
    r"""
    Indicator function of an affine set.

    .. math::

       \iota_{\mathbf{A}}^{\mathbf{b}}(\mathbf{x})
       :=
       \begin{cases}
           0 & \mathbf{A} \mathbf{x} = \mathbf{b} \\
           \infty & \text{otherwise}.
       \end{cases}

    .. math::

       \text{prox}_{\tau\, \iota_{\mathbf{A}}^{\mathbf{b}}}(\mathbf{x})
       :=
       \mathbf{x} - \mathbf{A}^{T} (\mathbf{A}\mathbf{A}^{T})^{-1} (\mathbf{Ax - b})

    Notes
    -----
    * Assumptions on :math:`\mathbf{A}` and :math:`\mathbf{b}`:

      * :math:`\mathbf{b} \in \text{span}(\mathbf{A})`.
      * :math:`\mathbf{A}` has full row-rank, i.e. :math:`\mathbf{A}` is square or fat.

    * :py:class:`~pyxu.operator.AffineSet` instances are **not arraymodule-agnostic**:
      they will only work with NDArrays belonging to the same array module as `A` and `b`.
    """

    @pxrt.enforce_precision(i="b")
    def __init__(self, A: pxa.LinOp, b: pxt.NDArray):
        """
        Parameters
        ----------
        A: ~pyxu.abc.operator.LinOp
            (M, N) operator
        b: NDArray
            (M,)
        """
        assert A.codim <= A.dim, f"`A` must have full row-rank, but A.shape = {A.shape}."
        assert (b.ndim == 1) and (b.size == A.codim)
        super().__init__(dim=A.dim)

        # Some approximate solution to A x = b, i.e x0.
        # x0 must be a good estimate given assumptions on (A, b).
        self._x0 = A.pinv(b, damp=0)
        self._b = A.apply(self._x0)  # fuzz `b` slightly to avoid numerical issues in .apply()
        self._A = A

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        in_set = xp.isclose(self._A.apply(arr), self._b).all(axis=-1, keepdims=True)
        out = self._bool2indicator(in_set, arr.dtype)
        return out

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        op = RangeSet(self._A.T)

        out = arr.copy()
        out -= op.prox(arr - self._x0, tau=1)  # tau arbitrary
        return out


class ConvexSetIntersection(_IndicatorFunction):
    r"""
    Indicator function of an intersection of convex domains :math:`\mathcal{C}_{1} \cap \cdots \cap \mathcal{C}_{K}`.

    .. math::

       \iota(\mathbf{x})
       :=
       \iota_{1}(\mathbf{x}) + \cdots + \iota_{K}(\mathbf{x}).

    :math:`\text{prox}_{\tau\, \iota}(\mathbf{x})` is computed using the PoCS algorithm.  (Dykstra's variant
    [PoCS_Dykstra]_.)

    This function assumes :math:`\mathcal{C}_{1} \cap \cdots \cap \mathcal{C}_{K} \ne \emptyset`.
    :py:meth:`~pyxu.operator.ConvexSetIntersection.prox` will loop indefinitely if this condition is violated.

    Examples
    --------
    .. code-block:: python3

       import numpy as np
       import pyxu.operator as pxo

       op = pxo.ConvexSetIntersection(             # intersection of
           pxo.LInfinityBall(),                    #   L-\infty ball centered at (0,0)
           pxo.L2Ball().asloss(np.array([1, 0])),  #   L2 ball centered at (1,0)
       )

       x = np.array([
           [0, 0],
           [1, 0],
           [2, 0],  # not in LInfinity ball -> \infty
           [1, 1],
           [1, -1],
           [0, 1],  # not in L2 ball -> \infty
        ])
       op.apply(x)  # [0 0 inf 0 0 inf]
    """

    def __init__(self, *args: cabc.Sequence[pxa.ProxFunc]):
        """
        Parameters
        ----------
        args: :py:class:`list` ( :py:class:`~pyxu.abc.ProxFunc` )
            Sequence of indicator functions encoding convex domains.
        """
        # Create `op` to auto-compute best shape behind the scenes.
        if len(args) == 1:
            op = args[0]
        else:
            op = functools.reduce(operator.add, args)
        super().__init__(dim=op.dim)

        self._f = args

        # Initialize PoCS solver.
        # Useful to do so in __init__() to be able to examine solver stats() after prox() calls.
        self._slvr = ConvexSetIntersection._PoCS(
            f=self._f,
            show_progress=False,
            log_var="x",
        )

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = [None] * len(self._f)
        for i, f in enumerate(self._f):
            out[i] = f.apply(arr)
        out = sum(out)
        return out

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        if len(self._f) == 1:
            # One projection sufficient
            op = self._f[0]
            out = op.prox(arr, tau)  # tau can be arbitrary
        else:
            # Use Dykstra's PoCS algorithm until a point in the intersection is found
            from pyxu.opt.stop import AbsError

            self._slvr.fit(
                x0=arr,
                stop_crit=AbsError(
                    var="x",
                    eps=1,  # any point in intersection satisfies (eps < inf)
                    f=lambda x: self.apply(x),
                ),
            )

            data, _ = self._slvr.stats()
            out = data["x"]
        return out

    def _expr(self) -> tuple:
        # Show all cvx-domains involved
        return ("add", *self._f)

    class _PoCS(pxa.Solver):
        # Dykstra's PoCS algorithm [PoCS_Dykstra]_
        #
        # This is a bare-bones Solver sub-class since only used internally.

        def __init__(self, f: cabc.Sequence[pxa.ProxFunc], **kwargs):
            super().__init__(**kwargs)
            self._f = f

        @pxrt.enforce_precision(i="x0")
        def m_init(self, x0: pxt.NDArray):
            mst = self._mstate  # shorthand
            mst["x"] = x0.copy()

            xp = pxu.get_array_module(x0)
            K = len(self._f)
            *sh, N = x0.shape
            mst["residual"] = xp.zeros((K, *sh, N), dtype=x0.dtype)

        def m_step(self):
            mst = self._mstate  # shorthand
            x, r = mst["x"], mst["residual"]

            for i, f in enumerate(self._f):
                y = f.prox(x - r[i], tau=1)  # tau is arbitrary
                r[i] = y - x
                x[:] = y
