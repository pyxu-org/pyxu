import typing as typ

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.abc.solver as pycs
import pycsou.opt.stop as pycos
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


class CG(pycs.Solver):
    r"""
    Conjugate Gradient Method.

    The Conjugate Gradient method solves the minimization problem

    .. math::

       \min_{x\in\mathbb{R}^{N}} \frac{1}{2} \mathbf{x}^{T} \mathbf{A} \mathbf{x} - \mathbf{x}^{T} \mathbf{b},

    where :math:`\mathbf{A}: \mathbb{R}^{N} \to \mathbb{R}^{N}` is a *symmetric* *positive definite*
    operator, and :math:`\mathbf{b} \in \mathbb{R}^{N}`.
    """

    def __init__(
        self,
        A: pyco.PosDefOp,
        *,
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        log_var: pyct.VarName = ("primal",),
    ):
        super().__init__(
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            log_var=log_var,
        )

        self._A = A

    def fit(
        self,
        b: pyct.NDArray,
        primal_init: pyct.NDArray = None,
        stop_crit: typ.Optional[pycs.StoppingCriterion] = None,
        mode: pycs.Mode = pycs.Mode.BLOCK,
    ):
        r"""
        Solve the minimization problem defined in :py:meth:`CG.__init__`, with the provided
        run-specific parameters.

        Parameters
        ----------
        b: NDArray
            (..., N) 'b' terms in the CG cost function. All problems are solved in parallel.
        primal_init: NDArray
           (..., N) primal variable initial point(s). Defaults to 0 if unspecified.
        stop_crit: StoppingCriterion
            Stopping criterion to end solver iterations.
            Defaults to stopping if all residual abs-norms reach 1e-5.
        mode: Mode
           Execution mode. See :py:class:`Solver` for usage examples.
           Useful method pairs depending on the execution mode:
           * BLOCK: fit()
           * ASYNC: fit() + busy() + stop()
           * MANUAL: fit() + steps()
        """
        if stop_crit is None:
            stop_crit = pycos.AbsError(
                eps=1e-5,
                var="residual",
                f=None,
                norm=2,
                satisfy_all=True,
            )
        self._fit_init(mode, stop_crit)
        self.m_init(b=b, primal_init=primal_init)
        self._fit_run()

    def m_init(self, b: pyct.NDArray, primal_init: pyct.NDArray):
        mst = self._mstate  # shorthand

        b = pycrt.coerce(b)
        xp = pycu.get_array_module(b)
        if primal_init is None:
            mst["primal"] = xp.zeros_like(b)
        else:
            mst["primal"] = pycrt.coerce(primal_init)

        # 2-stage res-computation guarantees RT-precision in case apply() not
        # enforce_precision()-ed.
        mst["residual"] = xp.zeros_like(b)
        mst["residual"][:] = b - self._A.apply(mst["primal"])
        mst["conjugate_dir"] = mst["residual"].copy()

    def m_step(self):
        mst = self._mstate
        x = mst["primal"]
        r = mst["residual"]
        p = mst["conjugate_dir"]
        xp = pycu.get_array_module(x)
        ap = self._A.apply(p)
        rr = xp.linalg.norm(r, ord=2, axis=-1, keepdims=True) ** 2
        alpha = rr / (p * ap).sum(axis=-1, keepdims=True)
        x = x + alpha * p
        r = r - alpha * ap
        beta = xp.linalg.norm(r, ord=2, axis=-1, keepdims=True) ** 2 / rr
        p = r + beta * p
        mst["primal"], mst["residual"], mst["conjugate_dir"] = x, r, p

    def solution(self) -> pyct.NDArray:
        """
        Returns
        -------
        p: NDArray
            (..., N) primal solution.
        """
        _, data = self.stats()
        return data.get("primal")
