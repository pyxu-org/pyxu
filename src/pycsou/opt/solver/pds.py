import itertools
import math
import numbers as nb
import typing as typ

import pycsou.abc as pyca
import pycsou.abc.operator as pyco
import pycsou.abc.solver as pycs
import pycsou.linop.base as pyclo
import pycsou.opt.stop as pycos
import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct


class PrimalDualSplitting(pycs.Solver):
    r"""
    Primal Dual Splitting (PDS) solver.

    This class is also accessible via the alias ``PDS()``.

    Notes
    -----
    The *Primal Dual Splitting (PDS)* method is described in [PDS]_ (this particular implementation is based on the pseudo-code Algorithm 7.1 provided in [FuncSphere]_ Chapter 7, Section1).
    It can be used to solve problems of the form:

    .. math::
       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathbf{K} \mathbf{x}).}

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with :math:`\beta`-*Lipschitz continuous* gradient,
      for some :math:`\beta\in[0,+\infty[`.
    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` and :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` are two *proper*, *lower semicontinuous* and *convex functions* with *simple proximal operators*.
    * :math:`\mathbf{K}:\mathbb{R}^N\rightarrow \mathbb{R}^M` is a *linear operator*, with **operator norm**:

      .. math::
         \Vert{\mathbf{K}}\Vert_2=\sup_{\mathbf{x}\in\mathbb{R}^N,\Vert\mathbf{x}\Vert_2=1} \Vert\mathbf{K}\mathbf{x}\Vert_2.

    * The problem is *feasible* --i.e. there exists at least one solution.

    **Remark 1:**

    The algorithm is still valid if one or more of the terms :math:`\mathcal{F}`, :math:`\mathcal{G}` or :math:`\mathcal{H}` is zero.

    **Remark 2:**

    Assume that the following holds:

    * :math:`\beta>0` and:

      - :math:`\frac{1}{\tau}-\sigma\Vert\mathbf{K}\Vert_{2}^2\geq \frac{\beta}{2}`,
      - :math:`\rho \in ]0,\delta[`, where :math:`\delta:=2-\frac{\beta}{2}\left(\frac{1}{\tau}-\sigma\Vert\mathbf{K}\Vert_{2}^2\right)^{-1}\in[1,2[.`

    * or :math:`\beta=0` and:

      - :math:`\tau\sigma\Vert\mathbf{K}\Vert_{2}^2\leq 1`
      - :math:`\rho \in [\epsilon,2-\epsilon]`, for some  :math:`\epsilon>0.`

    Then, there exists a pair :math:`(\mathbf{x}^\star,\mathbf{z}^\star)\in\mathbb{R}^N\times \mathbb{R}^M`} solution s.t. the primal and dual sequences of  estimates :math:`(\mathbf{x}_n)_{n\in\mathbb{N}}` and :math:`(\mathbf{z}_n)_{n\in\mathbb{N}}` *converge* towards :math:`\mathbf{x}^\star` and :math:`\mathbf{z}^\star` respectively, i.e.

    .. math::

       \lim_{n\rightarrow +\infty}\Vert\mathbf{x}^\star-\mathbf{x}_n\Vert_2=0, \quad \text{and} \quad  \lim_{n\rightarrow +\infty}\Vert\mathbf{z}^\star-\mathbf{z}_n\Vert_2=0.

    **Default values of the hyperparameters provided here always ensure convergence of the algorithm.**


    ``PrimalDualSplitting.fit()`` **Parameterization**

    x0: NDArray
        (..., N) initial point(s) for the primal variable.
    z0: NDArray
        (..., N) initial point(s) for the dual variable.
        If None (default), then use x0 as the initial point(s) for the dual variable as well.
    tau: Real
        Primal step size
    sigma: Real
        Dual step size
    rho: Real
        Momentum parameter
    beta: Real
        Lipschitz constant :math:`\beta` of the derivative of :math:`\mathcal{F}`.


    Examples
    --------
    Consider the following optimisation problem:

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}_+^N}\frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2\quad+\quad\lambda_1 \|\mathbf{D}\mathbf{x}\|_1\quad+\quad\lambda_2 \|\mathbf{x}\|_1,

    with :math:`\mathbf{D}\in\mathbb{R}^{N\times N}` the discrete derivative operator and :math:`\mathbf{G}\in\mathbb{R}^{L\times N}, \, \mathbf{y}\in\mathbb{R}^L, \lambda_1,\lambda_2>0.`
    This problem can be solved via PDS with :math:`\mathcal{F}(\mathbf{x})= \frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2`, :math:`\mathcal{G}(\mathbf{x})=\lambda_2\|\mathbf{x}\|_1,`
    :math:`\mathcal{H}(\mathbf{x})=\lambda \|\mathbf{x}\|_1` and :math:`\mathbf{K}=\mathbf{D}`.

    .. plot::
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from pycsou.opt.solver.pds import PDS
    >>> from pycsou._dev import FirstDerivative, DownSampling, SquaredL2Norm, L1Norm

    >>> x = np.repeat(np.asarray([0, 2, 1, 3, 0, 2, 0]), 10)
    >>> D = FirstDerivative(size=x.size, kind="forward")
    >>> D.lipschitz(tol=1e-3)
    >>> downsampling = DownSampling(size=x.size, downsampling_factor=3)
    >>> downsampling.lipschitz()
    >>> y = downsampling(x)
    >>> l22_loss = (1 / 2) * SquaredL2Norm().as_loss(data=y)
    >>> fidelity = l22_loss * downsampling
    >>> H = 0.1 * L1Norm()

    >>> G = 0.01 * L1Norm()
    >>> pds = PDS(f=fidelity, g=G, h=H, K=D)
    >>> x0, z0 = x * 0, x * 0
    >>> pds.fit(x0=x0, z0=z0)

    >>> estimate = pds.solution()
    >>> x_recons = estimate[0]
    >>>
    >>> plt.figure()
    >>> plt.stem(x, linefmt="C0-", markerfmt="C0o")
    >>> markerline, stemlines, baseline = plt.stem(
    >>>     np.where(downsampling.downsampling_mask)[0], y, linefmt="C3-", markerfmt="C3o"
    >>> )
    >>> markerline.set_markerfacecolor("none")
    >>> plt.stem(x_recons, linefmt="C1--", markerfmt="C1s")
    >>> plt.legend(["Ground truth", "Observation", "PDS Estimate"])
    >>> plt.show()

    See Also
    --------
    :py:class:`~pycsou.opt.solver.pds.PDS`, :py:class:`~pycsou.opt.solver.pds.ChambollePockSplitting`, :py:class:`~pycsou.opt.solver.pds.DouglasRachford`
    """

    def __init__(
        self,
        f: typ.Optional[pyco.DiffFunc] = None,
        g: typ.Optional[pyco.ProxFunc] = None,
        h: typ.Optional[pyco.ProxFunc] = None,
        K: typ.Optional[pyco.LinOp] = None,
        *,
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        log_var: pyct.VarName = ("x",),
    ):
        super().__init__(
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            log_var=log_var,
        )

        self._f = pyclo.NullFunc() if (f is None) else f
        self._g = pyclo.NullFunc() if (g is None) else g
        self._h = (
            pyclo.NullFunc() if (h is None) else h
        )  # TODO >> MAYBE IS EASIER TO SET TO NONE IF THIS IS NOT USE AFTERWARDS

        if h is not None:
            if K is None:
                self._K = pyclo.IdentityOp()
                self._K._lipschitz = 1
            else:
                if (K.shape[0] == h.shape[1]) or (h.shape[1] is None):
                    self._K = K
                else:
                    msg = (
                        f"Operator K with shape {K.shape} is inconsistent with functional h with dimension {h.shape}.)"
                    )
                    raise ValueError(msg)
        else:
            if K is None:
                self._K = pyclo.NullOp()

        if (f is None) and (g is None) and (h is None):  # TODO
            msg = " ".join(
                [
                    "Cannot minimize always-0 functional.",
                    "At least one of Parameter[f, g, h] must be specified.",
                ]
            )
            raise ValueError(msg)

    @pycrt.enforce_precision(i=["x0", "z0", "tau", "sigma", "rho", "beta"], allow_None=True)
    def m_init(
        self,
        x0: pyct.NDArray,
        z0: typ.Optional[pyct.NDArray],
        tau: typ.Optional[pyct.Real] = None,
        sigma: typ.Optional[pyct.Real] = None,
        rho: typ.Optional[pyct.Real] = None,
        beta: typ.Optional[pyct.Real] = None,
    ):
        mst = self._mstate  # shorthand
        mst["x"] = mst["x_prev"] = x0
        mst["z"] = mst["z_prev"] = self.set_dual_variable(z0)
        mst["beta"] = self.set_beta(beta)
        mst["tau"], mst["sigma"] = self.set_step_sizes(tau, sigma)
        mst["rho"] = self.set_momentum_term(rho)

    def m_step(self):
        mst = self._mstate  # shorthand
        x_temp = self._g.prox(
            mst["x"] - mst["tau"] * self._f.grad(mst["x"]) - mst["tau"] * self._K.adjoint(mst["z"]),
            tau=mst["tau"],
        )
        if not isinstance(self._h, pyclo.NullFunc):
            u = 2 * x_temp - mst["x"]
            z_temp = self._h.fenchel_prox(mst["z"] + mst["sigma"] * self._K(u), sigma=mst["sigma"])
            mst["z"] = mst["rho"] * z_temp + (1 - mst["rho"]) * mst["z"]
        mst["x"] = mst["rho"] * x_temp + (1 - mst["rho"]) * mst["x"]

    def default_stop_crit(self) -> pycs.StoppingCriterion:
        stop_crit_x = pycos.RelError(
            eps=1e-4,
            var="x",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        stop_crit_z = pycos.RelError(
            eps=1e-4,
            var="z",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit_x & stop_crit_z

    def solution(self) -> pyct.NDArray:
        """
        Returns
        -------
        p: NDArray
            (..., N) x solution.
        d: NDArray
            (..., N) z solution.
        """
        data, _ = self.stats()
        return data.get("x")

    def set_beta(self, beta: typ.Optional[pyct.Real]) -> float:
        r"""
        Sets the Lipschitz constant.
        Returns
        -------
        float
            Lipschitz constant.
        """
        if beta is None:
            if math.isfinite(dl := self._f._diff_lipschitz):
                return pycrt.coerce(dl)
            else:
                msg = "beta: automatic inference not supported for operators with unbounded Lipschitz gradients."
            raise ValueError(msg)
        else:
            try:
                assert beta > 0
                return pycrt.coerce(beta)
            except:
                raise ValueError(f"beta must be positive, got {beta}.")

    def set_step_sizes(self, tau: typ.Optional[pyct.Real], sigma: typ.Optional[pyct.Real]) -> typ.Tuple[float, float]:
        r"""
        Set the x/z step sizes.
        Returns
        -------
        Tuple[float, float]
            Sensible x/z step sizes.
        Notes
        -----
        In practice, the convergence speed  of the algorithm is improved by choosing :math:`\sigma` and :math:`\tau` as large as possible and relatively well-balanced --so that both the x and z variables converge at the same pace. In practice, it is hence recommended to choose perfectly balanced parameters :math:`\sigma=\tau` saturating the convergence inequalities.
        For :math:`\beta>0` this yields:
        .. math::
           \frac{1}{\tau}-\tau\Vert\mathbf{K}\Vert_{2}^2= \frac{\beta}{2} \quad\Longleftrightarrow\quad -2\tau^2\Vert\mathbf{K}\Vert_{2}^2-\beta\tau+2=0,
        which admits one positive root
        .. math::
           \tau=\sigma=\frac{1}{\Vert\mathbf{K}\Vert_{2}^2}\left(-\frac{\beta}{4}+\sqrt{\frac{\beta^2}{16}+\Vert\mathbf{K}\Vert_{2}^2}\right).
        For :math:`\beta=0`, this yields
        .. math::
           \tau=\sigma=\Vert\mathbf{K\Vert_{2}^{-1}.}
        """

        if (tau is not None) and (sigma is None):
            try:
                assert tau > 0
            except:
                raise ValueError(f"tau must be positive, got {tau}.")
            tau = sigma = tau
        elif (tau is None) and (sigma is not None):
            try:
                assert sigma > 0
            except:
                raise ValueError(f"sigma must be positive, got {sigma}.")
            tau = sigma = sigma
        elif (tau is None) and (sigma is None):
            mst = self._mstate  # shorthand
            if mst["beta"] > 0:
                if isinstance(self._h, pyclo.NullFunc):
                    tau = 2 / mst["beta"]
                    sigma = 0
                else:
                    if math.isfinite(self._K._lipschitz):
                        tau = sigma = (1 / (self._K._lipschitz) ** 2) * (
                            (-mst["beta"] / 4) + math.sqrt((mst["beta"] ** 2 / 16) + self._K._lipschitz ** 2)
                        )
                    else:
                        msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'lipschitz()'"
                        raise ValueError(msg)
            else:
                if isinstance(self._h, pyclo.NullFunc):
                    tau = 1
                    sigma = 0
                else:
                    if math.isfinite(self._K._lipschitz):
                        tau = sigma = 1 / self._K._lipschitz
                    else:
                        msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'lipschitz()'"
                        raise ValueError(msg)
        return tau, sigma

    def set_momentum_term(self, rho: typ.Optional[pyct.Real]) -> float:
        r"""
        Sets the momentum term.
        Returns
        -------
        float
            Momentum term.
        """
        if rho is None:
            mst = self._mstate
            if mst["beta"] > 0:
                rho = 0.9
            else:
                rho = 1
        return rho

    def set_dual_variable(self, z: typ.Optional[pyct.NDArray]) -> pyct.NDArray:
        r"""
        Initialize the dual variable if it is None. Creates a copy of the primal variable.
        Returns
        -------
        NDArray
            Initialized dual variable.
        """
        if isinstance(self._h, pyclo.NullFunc):
            return None
        else:
            return self._mstate["x"].copy() if z is None else z


class ChambollePockSplitting(PDS):
    def __init__(
        self,
        g: typ.Optional[pyco.ProxFunc] = None,
        h: typ.Optional[pyco.ProxFunc] = None,
        K: typ.Optional[pyco.LinOp] = None,
        *,
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        log_var: pyct.VarName = ("x",),
    ):
        super(ChambollePockSplitting).__init__(
            f=None,
            g=g,
            h=h,
            k=K,
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            log_var=log_var,
        )

    @pycrt.enforce_precision(i=["x0", "z0", "tau"], allow_None=True)
    def m_init(
        self,
        x0: pyct.NDArray,
        z0: typ.Optional[pyct.NDArray],
        tau: typ.Optional[pyct.Real] = 1.0,
    ):
        mst = self._mstate  # shorthand
        mst["x"] = mst["x_prev"] = x0
        mst["z"] = mst["z_prev"] = self.set_dual_variable(z0)
        mst["tau"] = tau
        mst["sigma"] = 1.0 / tau
        mst["rho"] = 1.0


class DouglasRachfordSplitting(PDS):
    def __init__(
        self,
        g: typ.Optional[pyco.ProxFunc] = None,
        h: typ.Optional[pyco.ProxFunc] = None,
        *,
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        log_var: pyct.VarName = ("x",),
    ):
        super(DouglasRachfordSplitting).__init__(
            f=None,
            g=g,
            h=h,
            K=None,
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            log_var=log_var,
        )

    @pycrt.enforce_precision(i=["x0", "z0", "tau"], allow_None=True)
    def m_init(
        self,
        x0: pyct.NDArray,
        z0: typ.Optional[pyct.NDArray],
        tau: typ.Optional[pyct.Real] = 1.0,
    ):
        mst = self._mstate  # shorthand
        mst["x"] = mst["x_prev"] = x0
        mst["z"] = mst["z_prev"] = self.set_dual_variable(z0)
        mst["tau"] = tau
        mst["sigma"] = 1.0 / tau
        mst["rho"] = 1.0


class ForwardBackwardSplitting(PDS):
    def __init__(
        self,
        f: typ.Optional[pyco.ProxFunc] = None,
        g: typ.Optional[pyco.ProxFunc] = None,
        *,
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        log_var: pyct.VarName = ("x",),
    ):
        super(ForwardBackwardSplitting).__init__(
            f=f,
            g=g,
            h=None,
            K=None,
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            log_var=log_var,
        )

    @pycrt.enforce_precision(i=["x0", "z0", "tau"], allow_None=True)
    def m_init(
        self,
        x0: pyct.NDArray,
        z0: typ.Optional[pyct.NDArray],
        tau: typ.Optional[pyct.Real] = None,
        rho: typ.Optional[pyct.Real] = 1.0,
    ):
        mst = self._mstate  # shorthand
        mst["x"] = mst["x_prev"] = x_init
        mst["z"] = mst["z_prev"] = self.set_dual_variable(z_init)
        mst["tau"] = tau
        mst["rho"] = 1.0


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    import pycsou.util as pycu
    from pycsou.abc.operator import DiffFunc, ProxFunc
    from pycsou.opt.solver.pds import PDS

    class SquaredL2Norm(DiffFunc):
        def __init__(self, shape: pyct.ShapeOrDim = None):
            super(SquaredL2Norm, self).__init__(shape=(1, None))
            self._diff_lipschitz = 2

        def apply(self, arr: pyct.NDArray) -> pyct.Real:
            xp = pycu.get_array_module(arr)
            return xp.linalg.norm(arr, axis=-1, keepdims=True) ** 2

        def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
            return 2 * arr

        def as_loss(self, data: typ.Optional[pyct.NDArray] = None) -> "DiffFunc":
            if data is None:
                return self
            else:
                return self.argshift(-data)

    class L1Norm(ProxFunc):
        def __init__(self, shape=None):
            super(L1Norm, self).__init__(shape=(1, None))
            self._lipschitz = 1

        def apply(self, arr: pyct.NDArray) -> pyct.Real:
            return np.linalg.norm(arr, ord=1)

        def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
            xp = pycu.get_array_module(arr)
            return xp.clip(xp.abs(arr) - tau, a_min=0, a_max=None) * xp.sign(arr)

    import numpy as np

    import pycsou.util as pycu
    import pycsou.util.ptype as pyct
    from pycsou.abc.operator import LinOp

    class FirstDerivative(LinOp):
        def __init__(self, size: int, axis: int = -1, sampling: float = 1.0, edge: bool = True, kind: str = "forward"):
            super(FirstDerivative, self).__init__((size, size))
            self.axis = axis
            self.sampling = sampling
            self.edge = edge
            self.kind = kind

            # choose apply and adjoint kind
            if kind == "forward":
                self.apply = self._apply_forward
                self.adjoint = self._adjoint_forward
            elif kind == "centered":
                self.apply = self._apply_centered
                self.adjoint = self._adjoint_centered
            elif kind == "backward":
                self.apply = self._apply_backward
                self.adjoint = self._adjoint_backward
            else:
                raise NotImplementedError("kind must be forward, centered, " "or backward")

        def _apply_forward(self, arr: pyct.NDArray) -> pyct.NDArray:
            xp = pycu.get_array_module(arr)
            arr = xp.swapaxes(arr, self.axis, -1)
            y = xp.zeros_like(arr)
            y[..., :-1] = (arr[..., 1:] - arr[..., :-1]) / self.sampling
            return y

        def _adjoint_forward(self, arr: pyct.NDArray) -> pyct.NDArray:
            xp = pycu.get_array_module(arr)
            arr = xp.swapaxes(arr, self.axis, -1)
            y = xp.zeros_like(arr)
            y[..., :-1] -= arr[..., :-1] / self.sampling
            y[..., 1:] += arr[..., :-1] / self.sampling
            return y

        def _apply_centered(self, arr: pyct.NDArray) -> pyct.NDArray:
            xp = pycu.get_array_module(arr)
            arr = xp.swapaxes(arr, self.axis, -1)
            y = xp.zeros_like(arr)
            y[..., 1:-1] = 0.5 * (arr[..., 2:] - arr[..., :-2]) / self.sampling
            if self.edge:
                y[..., 0] = (arr[..., 1] - arr[..., 0]) / self.sampling
                y[..., 0] = (arr[..., -1] - arr[..., -2]) / self.sampling
            return y

        def _adjoint_centered(self, arr: pyct.NDArray) -> pyct.NDArray:
            xp = pycu.get_array_module(arr)
            arr = xp.swapaxes(arr, self.axis, -1)
            y = xp.zeros_like(arr)
            y[..., :-2] -= 0.5 * arr[..., 1:-1] / self.sampling
            y[..., 2:] += 0.5 * arr[..., 1:-1] / self.sampling
            if self.edge:
                y[..., 0] -= arr[..., 0] / self.sampling
                y[..., 1] += arr[..., 0] / self.sampling
                y[..., -2] -= arr[..., -1] / self.sampling
                y[..., -1] += arr[..., -1] / self.sampling
            return y

        def _apply_backward(self, arr: pyct.NDArray) -> pyct.NDArray:
            xp = pycu.get_array_module(arr)
            arr = xp.swapaxes(arr, self.axis, -1)
            y = xp.zeros_like(arr)
            y[..., 1:] = (arr[..., 1:] - arr[..., :-1]) / self.sampling
            return y

        def _adjoint_backward(self, arr: pyct.NDArray) -> pyct.NDArray:
            xp = pycu.get_array_module(arr)
            arr = xp.swapaxes(arr, self.axis, -1)
            y = xp.zeros_like(arr)
            y[..., :-1] -= arr[..., 1:] / self.sampling
            y[..., 1:] += arr[..., 1:] / self.sampling
            return y

    class Masking(LinOp):
        def __init__(self, size: int, sampling_bool: typ.Union[pyct.NDArray, list]):
            if isinstance(sampling_bool, list):
                import numpy as xp
            else:
                xp = pycu.get_array_module(sampling_bool)

            self.sampling_bool = xp.asarray(sampling_bool).reshape(-1).astype(bool)
            self.input_size = size
            self.nb_of_samples = self.sampling_bool[self.sampling_bool == True].size
            if self.sampling_bool.size != size:
                raise ValueError("Invalid size of boolean sampling array.")
            super(Masking, self).__init__(shape=(self.nb_of_samples, self.input_size))

        def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
            return arr[..., self.sampling_bool]

        def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
            xp = pycu.get_array_module(arr)
            arr = arr.reshape(1, -1) if (arr.ndim == 1) else arr
            y = xp.zeros((*arr.shape[:-1], self.input_size), dtype=arr.dtype)
            y[..., self.sampling_bool] = arr
            return y

    class DownSampling(Masking):
        def __init__(
            self,
            size: int,
            downsampling_factor: typ.Union[int, tuple, list],
            shape: typ.Optional[tuple] = None,
            axis: typ.Optional[int] = None,
        ):
            if type(downsampling_factor) is int:
                if (shape is not None) and (axis is None):
                    self.downsampling_factor = len(shape) * (downsampling_factor,)
                else:
                    self.downsampling_factor = (downsampling_factor,)
            else:
                self.downsampling_factor = tuple(downsampling_factor)
            if shape is not None:
                if size != np.prod(shape):
                    raise ValueError(f"Array size {size} is incompatible with array shape {shape}.")
                if (axis is not None) and (axis > len(shape) - 1):
                    raise ValueError(f"Array size {size} is incompatible with array shape {shape}.")
            if (shape is None) and (len(self.downsampling_factor) > 1):
                raise ValueError("Please specify an array shape for multidimensional downsampling.")
            elif (shape is not None) and (axis is None) and (len(shape) != len(self.downsampling_factor)):
                raise ValueError(f"Inconsistent downsampling factors {downsampling_factor} for array of shape {shape}.")
            self.input_size = size
            self.input_shape = shape
            self.axis = axis
            self.downsampling_mask = self.compute_downsampling_mask()
            if self.input_shape is None:
                self.output_shape = None
            else:
                if len(self.downsampling_factor) > 1:
                    output_shape = []
                    for ax in range(len(self.input_shape)):
                        axis_indices = np.arange(self.input_shape[ax])
                        downsampled_axis_indices = axis_indices % self.downsampling_factor[ax]
                        output_shape.append(downsampled_axis_indices[downsampled_axis_indices == 0].size)
                    self.output_shape = tuple(output_shape)
                else:
                    output_shape = list(self.input_shape)
                    downsampled_axis_indices = np.arange(self.input_shape[self.axis])
                    downsampled_axis_indices = downsampled_axis_indices % self.downsampling_factor
                    output_shape[self.axis] = downsampled_axis_indices[downsampled_axis_indices == 0].size
                    self.output_shape = tuple(output_shape)

            super(DownSampling, self).__init__(size=self.input_size, sampling_bool=self.downsampling_mask)

        def compute_downsampling_mask(self) -> np.ndarray:
            if self.input_shape is None:
                indices = np.arange(self.input_size)
                downsampled_mask = (indices % self.downsampling_factor) == 0
            else:
                if len(self.downsampling_factor) > 1:
                    downsampled_mask = True
                    for ax in range(len(self.input_shape)):
                        axis_indices = np.arange(self.input_shape[ax])
                        downsampled_axis_indices = axis_indices % self.downsampling_factor[ax]
                        downsampled_axis_indices = downsampled_axis_indices.reshape(
                            downsampled_axis_indices.shape + (len(self.input_shape) - 1) * (1,)
                        )
                        downsampled_axis_indices = np.swapaxes(downsampled_axis_indices, 0, ax)
                        downsampled_mask = downsampled_mask * (downsampled_axis_indices == 0)
                else:
                    downsampled_mask = np.zeros(shape=self.input_shape, dtype=bool)
                    downsampled_mask = np.swapaxes(downsampled_mask, 0, self.axis)
                    downsampled_axis_indices = np.arange(self.input_shape[self.axis])
                    downsampled_axis_indices = downsampled_axis_indices % self.downsampling_factor
                    downsampled_mask[downsampled_axis_indices == 0, ...] = True
                    downsampled_mask = np.swapaxes(downsampled_mask, 0, self.axis)
            return downsampled_mask.reshape(-1)

    x = np.repeat([0, 2, 1, 3, 0, 2, 0], 10)
    D = FirstDerivative(size=x.size, kind="forward")
    D.lipschitz(tol=1e-3)
    rng = np.random.default_rng(0)
    downsampling = DownSampling(size=x.size, downsampling_factor=3)
    downsampling.lipschitz()
    y = downsampling(x)
    l22_loss = (1 / 2) * SquaredL2Norm().as_loss(data=y)
    fidelity = l22_loss * downsampling
    H = 0.1 * L1Norm()

    G = 0.01 * L1Norm()
    pds = PDS(f=fidelity, g=G, h=H, K=D)
    x0, z0 = x * 0, x * 0
    pds.fit(x_init=x0, z_init=z0)

    estimate = pds.solution()
    plt.figure()
    plt.stem(x, linefmt="C0-", markerfmt="C0o")
    markerline, stemlines, baseline = plt.stem(
        np.where(downsampling.downsampling_mask)[0], y, linefmt="C3-", markerfmt="C3o"
    )
    markerline.set_markerfacecolor("none")
    plt.stem(estimate[0], linefmt="C1--", markerfmt="C1s")
    plt.legend(["Ground truth", "Observation", "PDS Estimate"])
    plt.show()
