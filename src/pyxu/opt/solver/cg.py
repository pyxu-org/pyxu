import warnings

import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "CG",
]


class CG(pxa.Solver):
    r"""
    Conjugate Gradient Method.

    The Conjugate Gradient method solves the minimization problem

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}^{M_{1} \times\cdots\times M_{D}}}
       \frac{1}{2} \langle \mathbf{x}, \mathbf{A} \mathbf{x} \rangle - \langle \mathbf{x}, \mathbf{b} \rangle,

    where :math:`\mathbf{A}: \mathbb{R}^{{M_{1} \times\cdots\times M_{D}}} \to \mathbb{R}^{{M_{1} \times\cdots\times
    M_{D}}}` is a *symmetric* *positive definite* operator, and :math:`\mathbf{b} \in \mathbb{R}^{{M_{1}
    \times\cdots\times M_{D}}}`.

    The norm of the `explicit residual <https://www.wikiwand.com/en/Conjugate_gradient_method>`_ :math:`\mathbf
    {r}_{k+1}:=\mathbf{b}-\mathbf{Ax}_{k+1}` is used as the default stopping criterion.  This provides a guaranteed
    level of accuracy both in exact arithmetic and in the presence of round-off errors.  By default, the iterations stop
    when the norm of the explicit residual is smaller than 1e-4.

    Parameters (``__init__()``)
    ---------------------------
    * **A** (:py:class:`~pyxu.abc.PosDefOp`)
      --
      Positive-definite operator :math:`\mathbf{A}: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to \mathbb{R}^{M_{1}
      \times\cdots\times M_{D}}`.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.__init__`.

    Parameters (``fit()``)
    ----------------------
    * **b** (:py:attr:`~pyxu.info.ptype.NDArray`)
      --
      (..., M1,...,MD) :math:`\mathbf{b}` terms in the CG cost function.

      All problems are solved in parallel.
    * **x0** (:py:attr:`~pyxu.info.ptype.NDArray`, :py:obj:`None`)
      --
      (..., M1,...,MD) initial point(s).

      Must be broadcastable with `b` if provided.  Defaults to 0.
    * **restart_rate** (:py:attr:`~pyxu.info.ptype.Integer`)
      --
      Number of iterations after which restart is applied.

      By default, a restart is done after 'n' iterations, where 'n' corresponds to the dimension of :math:`\mathbf{A}`.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.fit`.
    """

    def __init__(self, A: pxa.PosDefOp, **kwargs):
        kwargs.update(
            log_var=kwargs.get("log_var", ("x",)),
        )
        super().__init__(**kwargs)

        self._A = A

    def m_init(
        self,
        b: pxt.NDArray,
        x0: pxt.NDArray = None,
        restart_rate: pxt.Integer = None,
    ):
        mst = self._mstate  # shorthand

        if restart_rate is not None:
            assert restart_rate >= 1
            mst["restart_rate"] = int(restart_rate)
        else:
            mst["restart_rate"] = self._A.dim_size

        xp = pxu.get_array_module(b)
        if x0 is None:
            mst["b"] = b
            mst["x"] = xp.zeros_like(b)
        elif b.shape == x0.shape:
            # No broadcasting involved
            mst["b"] = b
            mst["x"] = x0.copy()
        else:
            # In-place updates involving b/x won't work if shapes differ -> coerce to largest.
            mst["b"], x0 = xp.broadcast_arrays(b, x0)
            mst["x"] = x0.copy()

        mst["residual"] = mst["b"].copy()
        mst["residual"] -= self._A.apply(mst["x"])
        mst["conjugate_dir"] = mst["residual"].copy()

    def m_step(self):
        mst = self._mstate  # shorthand
        x, r, p = mst["x"], mst["residual"], mst["conjugate_dir"]
        xp = pxu.get_array_module(x)
        reduce = lambda x: x.sum(  # (..., M1,...,MD) -> (..., 1,...,1)
            axis=tuple(range(-self._A.dim_rank, 0)),
            keepdims=True,
        )

        Ap = self._A.apply(p)  # (..., M1,...,MD)
        rr = reduce(r**2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha = xp.nan_to_num(rr / reduce(p * Ap))  # (..., 1,...1)
        x += alpha * p

        if pxu.compute(xp.any(rr <= pxrt.Width(rr.dtype).eps())):  # explicit eval
            r[:] = mst["b"]
            r -= self._A.apply(x)
        else:  # implicit eval
            r -= alpha * Ap

        # Because CG can only generate N conjugate vectors in an N-dimensional space, it makes sense
        # to restart CG every N iterations.
        if self._astate["idx"] % mst["restart_rate"] == 0:  # explicit eval
            beta = 0
            r[:] = mst["b"]
            r -= self._A.apply(x)
        else:  # implicit eval
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                beta = xp.nan_to_num(reduce(r**2) / rr)
        p *= beta
        p += r

        # for homogenity with other solver code. Optional in CG due to in-place computations.
        mst["x"], mst["residual"], mst["conjugate_dir"] = x, r, p

    def default_stop_crit(self) -> pxa.StoppingCriterion:
        from pyxu.opt.stop import AbsError

        stop_crit = AbsError(
            eps=1e-4,
            var="residual",
            rank=self._A.dim_rank,
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit

    def objective_func(self) -> pxt.NDArray:
        x = self._mstate["x"]  # (..., M1,...,MD)
        b = self._mstate["b"]  # (..., M1,...,MD)

        f = self._A.apply(x)
        f = pxu.copy_if_unsafe(f)
        f /= 2
        f -= b
        f *= x

        axis = tuple(range(self._A.dim_rank, 0))
        y = f.sum(axis=axis)[..., np.newaxis]  # (..., 1)
        return y

    def solution(self) -> pxt.NDArray:
        """
        Returns
        -------
        x: NDArray
            (..., M1,...,MD) solution.
        """
        data, _ = self.stats()
        return data.get("x")
