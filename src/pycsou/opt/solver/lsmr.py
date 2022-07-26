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
    >>> x_star = rng.normal(size=(3, 3, 10))
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
        log_var: pyct.VarName = ("x",),
        **kwargs,
    ):
        super().__init__(
            log_var=log_var,
            **kwargs,
        )

        self._A = A
        self._atol = atol
        self._btol = btol
        self._ctol = (1.0 / conlim) if (conlim > 0) else 0.0
        self._iter_lim = (min(A.shape[0], A.shape[1])) * 2 if (iter_lim is None) else iter_lim
        self._damp = damp
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

        mst = self._mstate

        normb = xp.linalg.norm(b, axis=-1, keepdims=True)

        # Determine whether there's any trivial solution or not
        mst["trivial"] = normb == 0
        if mst["trivial"].all():
            mst["x"] = xp.zeros_like(self._A.T.apply(b))
            return
        elif mst["trivial"].any():
            # If some of them has trivial solution,
            # then remove them for the solving process
            # and add the trivial solution at the end of the process
            mst["b"] = b.copy()  # To remember the original one
            mask = xp.invert(mst["trivial"]).squeeze()
            b = b[mask]

        u = b
        normb = xp.linalg.norm(b, axis=-1, keepdims=True)

        if x0 is None:
            x = xp.zeros(n)
            beta = normb.copy()
        else:
            x = x0.copy()
            u -= self._A.apply(x)
            beta = xp.linalg.norm(u, axis=-1, keepdims=True)

        beta_cp, beta_gt0, beta_eq0 = beta.copy(), beta > 0, beta == 0
        beta_cp[beta_eq0] = 1
        u /= beta_cp
        v = beta_gt0 * self._A.T.apply(u)
        alpha = beta_gt0 * xp.linalg.norm(v, axis=-1, keepdims=True)

        alpha_cp = alpha.copy()
        alpha_cp[alpha_cp <= 0] = 1
        v /= alpha_cp

        normar = alpha * beta

        # Initialize state variables
        mst["x"], mst["u"], mst["v"], mst["alpha"] = x, u, v, alpha

        # Initialize variables for 1st iteration:
        mst["zetabar"], mst["alphabar"] = alpha * beta, alpha
        mst["rho"] = mst["rhobar"] = mst["cbar"] = xp.ones_like(beta)
        mst["sbar"] = xp.zeros_like(beta)
        mst["h"], mst["hbar"] = v.copy(), xp.zeros_like(x)

        # Initialize variables for estimation of ||r||:
        mst["betadd"] = beta
        mst["rhodold"] = xp.ones_like(beta)
        mst["betad"] = mst["tautildeold"] = mst["thetatilde"] = mst["zeta"] = mst["d"] = xp.zeros_like(beta)

        # Initialize variables for estimation of ||A|| and cond(A):
        mst["normA2"] = alpha**2
        mst["maxrbar"], mst["minrbar"] = xp.zeros_like(beta), 1e100 * xp.ones_like(beta)
        mst["normA"], mst["condA"] = xp.sqrt(mst["normA2"]), xp.zeros_like(beta)
        mst["normx"], mst["normr"], mst["normar"] = xp.zeros_like(beta), beta, normar
        self._normb = normb

        # Initialize variables for testing stopping rules:
        mst["test1"], mst["test2"], mst["test3"] = xp.ones_like(beta), alpha / beta, None

        # Vectorize the function of Givens rotation
        self._sym_ortho_func = xp.vectorize(self._sym_ortho)

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

        xp = pycu.get_array_module(mst["x"])

        x, u, v, alpha = mst["x"], mst["u"], mst["v"], mst["alpha"]
        zetabar, alphabar = mst["zetabar"], mst["alphabar"]
        rho, rhobar, cbar = mst["rho"], mst["rhobar"], mst["cbar"]
        sbar, h, hbar = mst["sbar"], mst["h"], mst["hbar"]
        betadd, betad, tautildeold, thetatilde = mst["betadd"], mst["betad"], mst["tautildeold"], mst["thetatilde"]
        rhodold, zeta, d = mst["rhodold"], mst["zeta"], mst["d"]
        normA2, minrbar, maxrbar = mst["normA2"], mst["minrbar"], mst["maxrbar"]

        # Bidiagonalizaion to obtain next beta, u, alpha, v:
        u = self._A.apply(v) - alpha * u
        beta = xp.linalg.norm(u, axis=-1, keepdims=True)

        if (beta > 0).all():
            u *= 1 / beta
            v = self._A.T.apply(u) - beta * v
            alpha = xp.linalg.norm(v, axis=-1, keepdims=True)
            if (alpha > 0).all():
                v *= 1 / alpha

        # Construct rotation:
        chat, shat, alphahat = self._sym_ortho_func(alphabar, self._damp, xp)

        # Use a plane rotation to turn B_i to R_i:
        rhoold = rho
        c, s, rho = self._sym_ortho_func(alphahat, beta, xp)
        thetanew = s * alpha
        alphabar = c * alpha

        # Use a plane rotation to turn R_i^T to R_i^bar:
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = self._sym_ortho_func(cbar * rho, thetanew, xp)
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
        ctildeold, stildeold, rhotildeold = self._sym_ortho_func(rhodold, thetabar, xp)
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
        maxrbar = xp.fmax(maxrbar, rhobarold)
        if self._astate["idx"] > 1:
            minrbar = xp.fmin(minrbar, rhobarold)
        condA = xp.fmax(maxrbar, rhotemp) / xp.fmin(minrbar, rhotemp)

        # Compute norms for convergence testing:
        normar = xp.abs(zetabar)
        normx = xp.linalg.norm(x, axis=-1, keepdims=True)

        # Compute testing parameters:
        test1 = normr / self._normb
        temp_mtx = normA * normr.copy()
        temp_mtx[temp_mtx == 0] = 1
        temp_mtx = 1 / temp_mtx
        temp_mtx[normA * normr == 0] = np.inf
        test2 = normar * temp_mtx
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
