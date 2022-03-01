import warnings

import pycsou.abc.operator as pyco
import pycsou.abc.solver as pycs
import pycsou.opt.stop as pycos
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


def is_broadcastable(arr1, arr2):
    shp1, shp2 = arr1.shape, arr2.shape
    if shp1[-1] != shp2[-1]:
        return False
    for a, b in zip(shp1[:-1], shp2[:-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


class CG(pycs.Solver):
    r"""
    Conjugate Gradient Method.
    The Conjugate Gradient method solves the system of linear equations of the form
    .. math::
       {\mathbf{A}\mathbf{x} = \mathbf{b}},
    for the vector :math:`\mathcal{x}: \mathbb{x}^N`.
    where:
    * :math:`\mathcal{A}: \mathbb{R}^M\times^N` is a *symmetric* *positive definite* matrix.
    * :math:`\mathcal{b}: \mathbb{b}^M`.
    """

    def __init__(
        self,
        a: pyco.PosDefOp,
        b: pyct.NDArray,
        log_var: pyct.VarName = ("x",),
    ):
        super().__init__(
            log_var=log_var,
        )

        self._a = a
        self._b = pycrt.coerce(b)

    def fit(
        self,
        x0: pyct.NDArray = None,
        stop_crit: pycs.StoppingCriterion = None,
        mode: pycs.Mode = pycs.Mode.BLOCK,
        tol: float = 1e-5,
    ):
        r"""
        Solve the minimization problem defined in :py:meth:`CG.__init__`, with the provided
        run-specific parameters.
        Parameters
        ----------
        x0: NDArray
           (..., N) Starting guess(es) for the solution(s). Defaults to a zero NDArray if unspecified.
        mode: Mode
           Execution mode. See :py:class:`Solver` for usage examples.
           Useful method pairs depending on the execution mode:
           * BLOCK: fit()
           * ASYNC: fit() + busy() + stop()
           * MANUAL: fit() + steps()
        tol: Real
           Tolerance for convergence, norm(residual) <= tol.  Defaults to :math:`1e-5` if unspecified.
        """
        # Create a stopping criteria from opt, using tol, and then call super().fit() method

        xp = pycu.get_array_module(self._b)
        x0 = x0 if (x0 is None) else pycrt.coerce(x0)
        try:
            assert not ((stop_crit is None) and (tol is None))
        except:
            raise ValueError(f"Both `stop_crit` and `tol` cannot be left undefined. Please define one.")
        if stop_crit is None:
            try:
                assert tol > 0
            except:
                raise ValueError(f"tol must be positive, got {tol}.")
            stop_crit = pycos.AbsError(eps=tol, var="r")

        elif (stop_crit is not None) and (tol is not None):
            warnings.warn(
                "The `tol` tolerance value given will be ignored. A custom-made Stopping Criterion has been "
                "given as argument. To avoid this warning use `tol=None` or use the default stopping "
                "criterion"
            )
        if x0 is not None:
            try:
                assert is_broadcastable(x0, self._b)

            except:
                raise ValueError(
                    f"Input initial guess has a mismatch in its shape dimension with data array `b` "
                    f"(shape {x0.shape} is not broadcastable with {self._b.shape})."
                )
            try:
                assert pycu.get_array_module(x0) == xp
            except:
                raise ValueError(
                    f"Input initial guess has a mismatch in its shape array module with data array `b` "
                    f"(array module {pycu.get_array_module(x0)} is different from {xp}."
                )
        else:
            x0 = xp.zeros((1, self._b.shape[-1]), dtype=pycrt.getPrecision().value)

        return super().fit(x0, mode=mode, stop_crit=stop_crit)

    def m_init(self, x0: pyct.NDArray, tol: float = 1e-5):
        self._mstate["x"] = x0
        self._mstate["r"] = r = self._b - self._a.apply(x0)
        self._mstate["p"] = r.copy()

    def m_step(self):
        mst = self._mstate
        x = mst["x"]
        r = mst["r"]
        p = mst["p"]

        ap = self._a.apply(p)
        rr = (r * r).sum(-1)[..., None]
        alpha = rr / (p * ap).sum(-1)[..., None]
        x = x + alpha * p
        r = r - alpha * ap
        beta = (r * r).sum(-1)[..., None] / rr
        p = r + beta * p
        mst["x"], mst["r"], mst["p"] = x, r, p

    @pycrt.enforce_precision(o=True)
    def solution(self) -> pyct.NDArray:
        """
        Returns
        -------
        x: NDArray
            (..., N) The converged solution.
        """
        _, data = self.stats()
        return data.get("x")
