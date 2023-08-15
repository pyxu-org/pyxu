import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.math.linalg as pxlg
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

        \min_{x\in\mathbb{R}^{N}} \frac{1}{2} \mathbf{x}^{T} \mathbf{A} \mathbf{x} - \mathbf{x}^{T} \mathbf{b},

     where :math:`\mathbf{A}: \mathbb{R}^{N} \to \mathbb{R}^{N}` is a *symmetric* *positive definite*
     operator, and :math:`\mathbf{b} \in \mathbb{R}^{N}`.

     The norm of the `explicit residual <https://www.wikiwand.com/en/Conjugate_gradient_method>`_
     :math:`\mathbf {r}_{k+1}:=\mathbf{b}-\mathbf{Ax}_{k+1}` is used as the default stopping criterion.
     This provides a guaranteed level of accuracy both in exact arithmetic and in the presence of
     round-off errors.
     By default, the iterations stop when the norm of the explicit residual is smaller than 1e-4.


     ``CG.fit()`` **Parameterization**

     b: pxt.NDArray
         (..., N) 'b' terms in the CG cost function.
         All problems are solved in parallel.
     x0: pxt.NDArray
        (..., N) initial point(s). Defaults to 0 if unspecified.
     restart_rate: pxt.Integer
        Number of iterations after which restart is applied.
        By default, restart is done after 'n' iterations, where 'n' corresponds to the dimension of
        the linear operator :math:`\mathbf{A}`.

    **Remark 1:**
    If provided, 'x0' must be broadcastable with 'b'.

    **Remark 2:**
    `Restarts <https://www.wikiwand.com/en/Conjugate_gradient_method>`_ may slow down convergence,
    but improve stability due to round-off error or ill-conditioning of the linear operator.
    If these issues are suspected, adjust 'restart_rate' accordingly.
    """

    def __init__(self, A: pxa.PosDefOp, **kwargs):
        kwargs.update(
            log_var=kwargs.get("log_var", ("x",)),
        )
        super().__init__(**kwargs)

        self._A = A

    @pxrt.enforce_precision(i=("b", "x0"))
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
            mst["restart_rate"] = self._A.dim

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
            mst["b"], mst["x"] = xp.broadcast_arrays(b, x0)
            mst["x"] = mst["x"].copy()

        mst["residual"] = mst["b"].copy()
        mst["residual"] -= self._A.apply(mst["x"])
        mst["conjugate_dir"] = mst["residual"].copy()

    def m_step(self):
        mst = self._mstate  # shorthand
        x, r, p = mst["x"], mst["residual"], mst["conjugate_dir"]
        xp = pxu.get_array_module(x)

        Ap = self._A.apply(p)
        rr = pxlg.norm(r, ord=2, axis=-1, keepdims=True) ** 2
        alpha = rr / (p * Ap).sum(axis=-1, keepdims=True)
        x += alpha * p

        if pxu.compute(xp.any(rr <= pxrt.Width(rr.dtype).eps())):  # explicit eval
            r[:] = mst["b"]
            r -= self._A.apply(x)
        else:  # implicit eval
            r -= alpha * Ap

        # Because CG can only generate n conjugate vectors in an n-dimensional space, it makes sense
        # to restart CG every n iterations.
        if self._astate["idx"] % mst["restart_rate"] == 0:  # explicit eval
            beta = 0
            r[:] = mst["b"]
            r -= self._A.apply(x)
        else:  # implicit eval
            beta = pxlg.norm(r, ord=2, axis=-1, keepdims=True) ** 2 / rr
        p *= beta
        p += r

        # for homogenity with other solver code. Optional in CG due to in-place computations.
        mst["x"], mst["residual"], mst["conjugate_dir"] = x, r, p

    def default_stop_crit(self) -> pxa.StoppingCriterion:
        from pyxu.opt.stop import AbsError

        stop_crit = AbsError(
            eps=1e-4,
            var="residual",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit

    def objective_func(self) -> pxt.NDArray:
        x = self._mstate["x"]
        b = self._mstate["b"]

        f = self._A.apply(x)
        f = pxu.copy_if_unsafe(f)
        f /= 2
        f -= b
        f *= x

        return f.sum(axis=-1, keepdims=True)

    def solution(self) -> pxt.NDArray:
        """
        Returns
        -------
        x: pxt.NDArray
            (..., N) solution.
        """
        data, _ = self.stats()
        return data.get("x")
