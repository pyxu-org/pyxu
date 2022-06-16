# #############################################################################
# lsmr.py
# ========
# Author : Kaan Okumus [okukaan@gmail.com]
# #############################################################################

import typing as typ

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.abc.solver as pycs
import pycsou.opt.stop as pycos
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


class LSMR(pycs.Solver):
    r"""
    LSMR Method.

    The LSMR method solves the system of linear equations :math:`\mathbf{A}\mathbf{x}=\mathbf{b}` iteratively. If the
    system is inconsistent, it solves the least-squares problem :math:`\min ||\mathbf{b} - \mathbf{A}\mathbf{x}||_2`.
    :math:`\mathbf{A}` is a rectangular matrix of dimension m-by-n, where all cases are allowed: m=n, m>n or m<n.
    :math:`\mathbf{b}` is a vector of length m. The matrix :math:`\mathbf{A}` may be dense or sparse.

    ``LSMR()`` **Parameterization**

    A: :py:class:`pycsou.abc.operator.PosDefOp`
        Any positive definite linear operator is accepted.
    damp: float
        Damping coefficient. Default is 0.
    atol, btol: float
        Stopping tolerances. LSMR continues iterations until a certain backward error estimate is smaller than some
        quantity depending on atol and btol. Default is 1e-6 for both.
    conlim: float
        LSMR terminates if an estimate :math:`\text{cond}(\mathbf{A})` exceeds conlim. Default is 1e8.
    iter_lim: int, optional
        LSMR terminates if the number of iterations reaches iter_lim. Default is None.

    ``LSMR.fit()`` **Parameterization**

    b: NDArray
         (..., N) 'b' terms in the LSMR cost function. All problems are solved in parallel.
    x0: NDArray
        (..., N) initial point(s). Defaults to 0 if unspecified.
    restart_rate: int
        Number of iterations after which restart is applied. By default, restart is done after 'n' iterations, where 'n'
        corresponds to the dimension of the linear operator :math:`\mathbf{A}`.

    **Remark:** :py:class:`pycsou.opt.solver.stop.StopCriterion_LSMR` is developed for the stopping criterion of LSMR. For
    computational speed, explicit norms were not computated. Instead, their estimation was used, which is referred from
    [2]_.

    Examples
    --------
    >>> import numpy as np
     >>> from pycsou.abc import LinOp
    >>> # Create a PSD linear operator
    >>> rng = np.random.default_rng(seed=0)
    >>> mat = rng.normal(size=(10, 10))
    >>> A = LinOp.from_array(mat).gram()
    >>> # Create the ground truth 'x_star'
    >>> x_star = rng.normal(size=(2, 2, 10))
    >>> # Generate the corresponding data vector 'b'
    >>> b = A.apply(x_star)
    >>> # Solve 'Ax=b' for 'x' with the conjugate gradient method
    >>> lsmr = LSMR(A, show_progress=False)
    >>> lsmr.fit(b=b)
    >>> x_solution = lsmr.solution()
    >>> assert np.allclose(x_star, x_solution)
    True

    **References:**

    .. [2] D. C.-L. Fong and M. A. Saunders,
        "LSMR: An iterative algorithm for sparse least-squares problems",
        SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
    """

    def __init__(
        self,
        A: pyco.PosDefOp,
        *,
        damp: float = 0.0,
        atol: float = 1e-06,
        btol: float = 1e-06,
        conlim: float = 1e08,
        iter_lim: typ.Optional[int] = None,
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        show_progress: bool = True,
        log_var: pyct.VarName = ("x",),
    ):
        super().__init__(
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
            log_var=log_var,
        )

        self._A = A
        self._atol = atol
        self._btol = btol
        self._ctol = (1.0 / conlim) if (conlim > 0) else 0.0
        self._iter_lim = (min(A.shape[0], A.shape[1])) if (iter_lim is None) else iter_lim
        self._damp = damp
        self._itn = 0
        self._normb = None

    @pycrt.enforce_precision(i=["b", "x0"], allow_None=True)
    def m_init(
        self,
        b: pyct.NDArray,
        x0: typ.Optional[pyct.NDArray] = None,
    ):
        xp = pycu.get_array_module(b)

        b = xp.atleast_1d(b)
        if b.ndim > 1:
            b = b.squeeze()
        m, n = self._A.shape

        u = b
        normb = xp.linalg.norm(b)

        if x0 is None:
            x = xp.zeros(n)
            beta = normb.copy()
        else:
            x = x0.copy()
            u -= self._A.apply(x)
            beta = xp.linalg.norm(u)

        if beta > 0:
            u *= 1 / beta
            v = self._A.T.apply(u)
            alpha = xp.linalg.norm(v)
        else:
            v = xp.zeros_like(x)
            alpha = 0

        if alpha > 0:
            v *= 1 / alpha

        mst = self._mstate

        normar = alpha * beta
        if normar == 0:
            mst["trivial"] = True
            if b.ndim == 1:
                mst["x"] = xp.zeros(n)
            elif b.ndim == 2:
                mst["x"] = xp.zeros((b.shape[0], n))
            return

        mst["x"], mst["u"], mst["v"], mst["alpha"], mst["trivial"] = x, u, v, alpha, False

        # Initialize variables for 1st iteration:
        mst["zetabar"], mst["alphabar"] = alpha * beta, alpha
        mst["rho"] = mst["rhobar"] = mst["cbar"] = 1
        mst["sbar"] = 0
        mst["h"], mst["hbar"] = v.copy(), xp.zeros_like(x)

        # Initialize variables for estimation of ||r||:
        mst["betadd"] = beta
        mst["rhodold"] = 1
        mst["betad"] = mst["tautildeold"] = mst["thetatilde"] = mst["zeta"] = mst["d"] = 0

        # Initialize variables for estimation of ||A|| and cond(A):
        mst["normA2"] = alpha**2
        mst["maxrbar"], mst["minrbar"] = 0, 1e100
        mst["normA"], mst["condA"] = xp.sqrt(mst["normA2"]), 1
        mst["normx"], mst["normr"], mst["normar"] = 0, beta, normar
        self._normb = normb

        # Initialize variables for testing stopping rules:
        mst["test1"], mst["test2"], mst["test3"] = 1, alpha / beta, None

    @staticmethod
    def _sym_ortho(a, b, xp):
        """
        Stable implementation of Givens rotation.
        """
        if b == 0:
            return xp.sign(a), 0, xp.abs(a)
        elif a == 0:
            return 0, xp.sign(b), xp.abs(b)
        elif xp.abs(b) > xp.abs(a):
            tau = a / b
            s = xp.sign(b) / xp.sqrt(1 + tau**2)
            c = s * tau
            r = b / s
        else:
            tau = b / a
            c = xp.sign(a) / xp.sqrt(1 + tau**2)
            s = c * tau
            r = a / c
        return c, s, r

    def m_step(self):

        mst = self._mstate
        if mst["trivial"]:
            return
        xp = pycu.get_array_module(mst["x"])

        x, u, v, alpha = mst["x"], mst["u"], mst["v"], mst["alpha"]
        zetabar, alphabar = mst["zetabar"], mst["alphabar"]
        rho, rhobar, cbar = mst["rho"], mst["rhobar"], mst["cbar"]
        sbar, h, hbar = mst["sbar"], mst["h"], mst["hbar"]
        betadd, betad, tautildeold, thetatilde = mst["betadd"], mst["betad"], mst["tautildeold"], mst["thetatilde"]
        rhodold, zeta, d = mst["rhodold"], mst["zeta"], mst["d"]
        normA2, minrbar, maxrbar = mst["normA2"], mst["minrbar"], mst["maxrbar"]

        self._itn += 1

        # Bidiagonalizaion to obtain next beta, u, alpha, v:
        u = self._A.apply(v) - alpha * u
        beta = xp.linalg.norm(u)

        if beta > 0:
            u *= 1 / beta
            v = self._A.T.apply(u) - beta * v
            alpha = xp.linalg.norm(v)
            if alpha > 0:
                v *= 1 / alpha

        # Construct rotation:
        chat, shat, alphahat = self._sym_ortho(alphabar, self._damp, xp)

        # Use a plane rotation to turn B_i to R_i:
        rhoold = rho
        c, s, rho = self._sym_ortho(alphahat, beta, xp)
        thetanew = s * alpha
        alphabar = c * alpha

        # Use a plane rotation to turn R_i^T to R_i^bar:
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = self._sym_ortho(cbar * rho, thetanew, xp)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        # Update h, h_hat, x:
        hbar = h - hbar * (thetabar * rho / (rhoold * rhobarold))
        x = x + (zeta / (rho * rhobar)) * hbar
        h = v - h * (thetanew / rho)

        # Estimation of ||r||:

        betaacute = chat * betadd
        betacheck = -shat * betadd

        betahat = c * betaacute
        betadd = -s * betaacute

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = self._sym_ortho(rhodold, thetabar, xp)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d += betacheck**2
        normr = xp.sqrt(d + (betad - taud) ** 2 + betadd**2)

        # Estimation of ||A||:
        normA2 = normA2 + beta**2
        normA = xp.sqrt(normA2)
        normA2 = normA2 + alpha**2

        # Estimation of cond(A):
        maxrbar = max(maxrbar, rhobarold)
        if self._itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Compute norms for convergence testing:
        normar = abs(zetabar)
        normx = xp.linalg.norm(x)

        # Compute testing parameters:
        test1 = normr / self._normb
        if (normA * normr) != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = np.infty
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / self._normb)
        rtol = self._btol + self._atol * normA * normx / self._normb

        mst["x"], mst["u"], mst["v"], mst["alpha"] = x, u, v, alpha
        mst["zetabar"], mst["alphabar"] = zetabar, alphabar
        mst["rho"], mst["rhobar"], mst["cbar"] = rho, rhobar, cbar
        mst["sbar"], mst["h"], mst["hbar"] = sbar, h, hbar
        mst["betadd"], mst["betad"] = betadd, betad
        mst["tautildeold"], mst["thetatilde"] = tautildeold, thetatilde
        mst["rhodold"], mst["zeta"], mst["d"] = rhodold, zeta, d
        mst["maxrbar"], mst["minrbar"] = maxrbar, minrbar
        mst["normA2"], mst["normA"], mst["condA"] = normA2, normA, condA
        mst["normx"], mst["normr"], mst["normar"] = normx, normr, normar
        mst["test1"], mst["test2"], mst["test3"] = test1, test2, test3
        mst["t1"], mst["rtol"] = t1, rtol

    def default_stop_crit(self) -> pycs.StoppingCriterion:
        stop_crit = pycos.StopCriterion_LSQMR(
            method="lsmr",
            atol=self._atol,
            ctol=self._ctol,
            itn=0,
            iter_lim=self._iter_lim,
        )
        return stop_crit

    def solution(self) -> pyct.NDArray:
        """
        Returns
        -------
        p: NDArray
            (..., N) solution.
        """
        data, _ = self.stats()
        return data.get("x")
