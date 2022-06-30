import collections.abc as cabc
import typing as typ

import dask.array as da
import finufft
import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "NUFFT",
]


def _wrap_if_dask(func: cabc.Callable) -> cabc.Callable:
    def wrapper(obj, arr):
        xp = pycu.get_array_module(arr)
        out = func(obj, pycu.compute(arr))

        if xp == da:
            return xp.array(out)
        else:
            return xp.array(out, copy=False)

    return wrapper


class NUFFT(pyco.LinOp):
    r"""
    Non-Uniform Fast Fourier Transform (NUFFT) of Type 1/2/3 (for :math:`d=1,2,3`).

    The *Non-Uniform Fast Fourier Transform (NUFFT)* generalizes the FFT to off-grid data. There are three main types of
    NUFFTs proposed in the literature: Type 1 (*non-uniform to uniform*), Type 2 (*uniform to non-uniform*), or Type 3 (*non-uniform to non-uniform*).
    These transforms are performed to a user-prescribed tolerance, at close-to-FFT speeds. Under the hood, this involves detailed kernel design, custom spreading/interpolation stages, and FFT calls.
    See the notes below as well as [FINUFFT]_ for definitions of the various transform types and algorithmic details.

    The transforms can be instantiated via the custom class constructors :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type1`, :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type2`, and :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type3`
    respectively. The dimension of the NUFFT transforms is inferred from the dimensions of the input arguments, with support for dimensions 1, 2 and 3.

    Notes
    -----
    We adopt here the same notational conventions as in [FINUFFT]_.

    **Mathematical Definition.**
    Let :math:`d\in\{1,2,3\}` and consider the Cartesian-product mesh

    .. math::

        \mathcal{I}_{N_1,\ldots, N_d}=\mathcal{I}_{N_1}\times \cdots\times  \mathcal{I}_{N_d}\subset\mathbb{Z}^d

    of the hyperinterval :math:`\Pi_{i=1}^d [-N_i/2,N_i/2]\subset \mathbb{R}^d`, where the mesh indices :math:`\mathcal{I}_{N_i}\subset\mathbb{Z}` are given for each
    dimension :math:`i=1,\dots, d` by:

    .. math::

        \mathcal{I}_{N_i}=\begin{cases}\{-N_i/2, \ldots, N_i/2-1\}, & N_i\in 2\mathbb{Z} \text{ (even)}, \\  \{-(N_i-1)/2, \ldots, (N_i-1)/2\}, & N_i\in 2\mathbb{Z}+1 \text{ (odd)}.\end{cases}


    Then, the various NUFFT operators approximate, up to a requested relative accuracy :math:`\varepsilon>0`, the following exponential sums respectively:

    .. math::

        &u_{\mathbf{n}} = \sum_{j=1}^{M} w_{j} e^{\sigma i\langle \mathbf{n}, \mathbf{x}_{j} \rangle}, \quad &\mathbf{n}\in \mathcal{I}_{N_1,\ldots, N_d},\qquad &\text{Type 1 (non-uniform to uniform)}\\
        &w_{j} = \sum_{\mathbf{n}\in\mathcal{I}_{N_1,\ldots, N_d}} u_{\mathbf{n}} e^{\sigma i\langle \mathbf{n}, \mathbf{x}_{j} \rangle }, \quad &j=1,\ldots, M,\qquad  &\text{Type 2 (uniform to non-uniform)}\\
        &v_{k} = \sum_{j=1}^{M} w_{j} e^{\sigma i\langle \mathbf{z}_k, \mathbf{x}_{j} \rangle }, \quad &k=1,\ldots, N, \qquad &\text{Type 3 (non-uniform to non-uniform)}


    where :math:`\sigma \in \{+1, -1\}` defines the sign of the transforms, and where :math:`u_{\mathbf{n}}, v_{k}, w_{j}\in \mathbb{C}` are complex coefficients.
    For the NUFFTs of Types 1 and 2, the non-uniform samples :math:`\mathbf{x}_{j}` are assumed to lie in :math:`[-\pi,\pi)^d`. For the NUFFT of Type 3, the non-uniform samples
    :math:`\mathbf{x}_{j}` and :math:`\mathbf{z}_{k}` are arbitrary points in :math:`\mathbb{R}^d`.

    **Adjoint NUFFTs.**
    The NUFFTs of Types 1 and 2 with opposite signs form an *adjoint pair*. The adjoint of the NUFFT of Type 3 is obtained by
    flipping the transform's sign and switching the roles of   :math:`\mathbf{z}_k` and :math:`\mathbf{x}_{j}` in (3).

    **Lipschitz Constants.** The NUFFT of Type 1 can be interpreted as a truncated Fourier Series  of a :math:`2\pi`-periodic
    Dirac stream with innovations :math:`w_j, \mathbf{x}_j`. From Parseval's equality, we have hence

    .. math::

        \|u_{\mathbf{n}}\|^2= \frac{1}{2\pi} \left\|\sum_{j=1}^M w_j d_{N_1,\ldots,N_d}(\cdot-\mathbf{x}_j)\right\|_2^2 = \mathbf{w}^HG\mathbf{w}\leq \|G\|_2\|\mathbf{w}\|_2^2,

    where :math:`d_{N_1,\ldots,N_d}:[-\pi, \pi)^d \to \mathbb{R}` is the :math:`d`-dimensional Dirichlet kernel with bandwidths :math:`N_1,\ldots,N_d` and
    :math:`G\in\mathbb{R}^{M \times M}` is the Gram matrix with entries :math:`G_{ij}=d_{N_1,\ldots,N_d}(\mathbf{x}_i-\mathbf{x}_j)`.
    The Lipschitz constant of the NUFFT of Type 1 is then proportional to the square root of the largest singular value of :math:`G`. Since the Gram is positive semi-definite,
    its largest eigenvalue can be bounded by its trace, which yields :math:`L = \sqrt{\|G\|_2/2\pi}\leq \sqrt{M\Pi_{i=1}^d N_i/2\pi}`. We therefore set the Lipschitz constant of
    the NUFFT of Type 1 (and Type 2 since the latter is the adjoint of Type 1) to this value. For the NUFFT of Type 3, we bound the Lipschitz constant by the Frobenius norm of the operator,
    which yields :math:`L \leq \sqrt{NM}`. Not that these Lipschitz constants are cheap to compute but can be quite pessimistic. Tighter Lipschitz constants can be computed
    by calling the method :py:meth:`~pycsou.abc.operator.LinOp.lipschitz`.

    **Error Analysis.**
    Let :math:`\tilde{\mathbf{u}}\in\mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}` and :math:`\tilde{\mathbf{w}}\in\mathbb{C}^M`
    be the outputs of the NUFFT algorithms of Types 1 and 2,  which approximate the sequences :math:`{\mathbf{u}}\in\mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}` and :math:`{\mathbf{w}}\in\mathbb{C}^M`
    defined in (1) and (2) respectively. Then, it is shown in [FINUFFT]_ that the relative errors :math:`\|\tilde{\mathbf{u}}-{\mathbf{u}}\|_2/\|{\mathbf{u}}\|_2`
    and :math:`\|\tilde{\mathbf{w}}-{\mathbf{w}}\|_2/\|{\mathbf{w}}\|_2` are **almost always similar to the user-requested tolerance** :math:`\varepsilon`, except
    when round-off error dominates (i.e. very small user-requested tolerances). The same holds approximately for the NUFFT of Type 3, the latter being a Type 2 NUFFT nested into a Type 1 NUFFT.
    Note however that this is a *typical error analysis*: some degenerate (but rare) worst-case scenarios can result in much higher errors.


    **Complexity.**
    Evaluating naively the exponential sums in any of the three types of transform (1), (2) and (3) has *bilinear complexity* :math:`O(NM)`, where :math:`N=N_1\ldots N_d` for the NUFFTs of Types 1 and 2.
    NUFFT algorithms compute these sums, to a user-specified relative tolerance :math:`\varepsilon`, in log-linear complexity in both :math:`N` and :math:`M`.
    More specifically, the complexity of the various NUFFTs are given by (see [FINUFFT]_):

    .. math::

        &\mathcal{O}\left(N\log(N) + M|\log(\varepsilon)|^d\right)\qquad &\text{(Types 1 and 2)}\\
        &\mathcal{O}\left(\Pi_{i=1}^dX_iZ_i\sum_{i=1}^d\log(X_iZ_i) + (M + N)|\log(\varepsilon)|^d\right)\qquad &\text{(Type 3)}

    where :math:`X_i = \max_{j=1,\ldots,M}|(\mathbf{x}_j)_i|` and :math:`Z_i = \max_{k=1,\ldots,N}|(\mathbf{z}_k)_i|` for :math:`i=1,\ldots,d`.
    The two terms intervening in the complexities above correspond to the complexity of the FFT and spreading/interpolation steps respectively.
    The complexity of the NUFFT of Type 3 can be arbitrarily large for poorly-centered data. In certain cases however, an easy fix consists in
    translating the data before and after the NUFFT via pre/post-phasing operations with linear complexity, as described in equation (3.24) of [FINUFFT]_.
    This fix can be activated on request via the optional argument ``center`` of the class constructor :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type3`.

    **Backend.** The NUFFT tansforms are computed via Python wrappers to the CPU-based and multithreaded C++ library `FINUFFT <https://github.com/flatironinstitute/finufft>`_
    or its CUDA equivalent `cuFINUFFT <https://github.com/flatironinstitute/cufinufft>`_ for GPU computing (see also [FINUFFT]_ and [cuFINUFFT]_).
    It uses minimal RAM, and performs the expensive spreading/interpolation between nonuniform points and the fine grid via the “exponential of semicircle” kernel in a cache-aware and load-balanced multithreaded fashion.
    This kernel is simpler and faster to evaluate than other kernels used in NUFFT algorithms, such as the Kaiser–Bessel, yet has essentially identical error (see [FINUFFT]_).

    **Optional Parameters.**
    Aside from the mandatory inputs the FINUFFT library on which this class builds
    accepts multiple optional parameters. These adjust the performances of the algorithm, change the output format, or provide debug/timing text to stdout.
    While the default options are sensible in most setups, advanced users may want finer control and change options from their defaults.
    This can be done by passing a dictionary of keyword arguments ``kwargs`` to the constructors of the various transforms, which will then be passed to the
    Python wrapper of the C++ FINUFFT library. See the `guru interface <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_ from FINUFFT and its `companion page <https://finufft.readthedocs.io/en/latest/opts.html#options-parameters>`_ detailing additional optional parameters for
    a comprehensive list of optional parameters.

    Warnings
    --------
    The FINUFFT library exposes a ``dtype`` keyword to control the precision (single or double) at which the transforms are performed.
    Do not rely on this optional parameter to set the precision as the latter is ignored by the :py:class:`~pycsou.operator.linop.nufft.NUFFT` class. Instead, use the context manager :py:class:`~pycsou.runtime.Precision`
    to control the floating point precision.

    See Also
    --------
    FFT, DCT, Radon
    """

    # The goal of this wrapper class is to sanitize __init__() inputs.

    def __init__(self, shape: pyct.NonAgnosticShape):
        r"""
        For internal purposes only. For instantiating a NUFFT transform, consider the custom class constructors
        :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type1`, :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type2`, and :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type3`
        respectively.
        """
        super().__init__(shape)

    @staticmethod
    @pycrt.enforce_precision(i="x", o=False, allow_None=False)
    def type1(
        x: pyct.NDArray,
        N: typ.Union[int, tuple[int, ...]],
        isign: typ.Literal[1, -1] = 1,
        eps: float = 1e-6,
        real: bool = False,
        **kwargs,
    ) -> pyco.LinOp:
        r"""
        Type 1 NUFFT (non-uniform to uniform).

        Parameters
        ----------
        x: NDArray
            (M, [d]) d-dimensional sample points :math:`\mathbf{x}_{j} \in \mathbb{R}^{d}`.
        N: int | tuple[int]
            ([d],) mesh size in each dimension :math:`(N_1, \ldots, N_d)`. If an integer is passed the mesh is assumed to
            have the same size in each dimension.
        isign: 1 | -1
            Sign :math:`\sigma` of the transform.
        eps: float
            Requested accuracy.
        real: bool
            If ``True``, assumes real inputs to the NUFFT.
        **kwargs
            Extra keyword parameters to `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_. (Illegal keywords are dropped silently.)
            Most useful is `n_trans`.

        Returns
        -------
        op: LinOp
            An NUFFT operator of type 1 with pre-computed plan.
        """
        init_kwargs = _NUFFT1._sanitize_init_kwargs(
            x=x, N=N, isign=isign, eps=eps, real_input=real, real_output=False, **kwargs
        )
        return _NUFFT1(**init_kwargs)

    @staticmethod
    @pycrt.enforce_precision(i="x", o=False, allow_None=False)
    def type2(
        x: pyct.NDArray,
        N: typ.Union[int, tuple[int, ...]],
        isign: typ.Literal[1, -1] = 1,
        eps: float = 1e-6,
        real: bool = False,
        **kwargs,
    ) -> pyco.LinOp:
        r"""
        Type 2 NUFFT (non-uniform to uniform).

        Parameters
        ----------
        x: NDArray
            (M, [d]) d-dimensional query points :math:`\mathbf{x}_{j} \in \mathbb{R}^{d}`.
        N: int | tuple[int]
            ([d],) mesh size in each dimension :math:`(N_1, \ldots, N_d)`. If an integer is passed the mesh is assumed to
            have the same size in each dimension.
        isign: 1 | -1
            Sign :math:`\sigma` of the transform.
        eps: float
            Requested accuracy.
        real: bool
            If ``True``, assumes real inputs to the NUFFT.
        **kwargs
            Extra keyword parameters to `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_. (Illegal keywords are silently dropped.)
            Most useful is `n_trans`.

        Returns
        -------
        op: LinOp
            An NUFFT operator of type 2 with pre-computed plan.
        """
        init_kwargs = _NUFFT1._sanitize_init_kwargs(
            x=x, N=N, isign=-isign, eps=eps, real_input=False, real_output=real, **kwargs
        )
        return _NUFFT1(**init_kwargs).T

    @staticmethod
    @pycrt.enforce_precision(i=("x", "z"), o=False, allow_None=False)
    def type3(
        x: pyct.NDArray,
        z: pyct.NDArray,
        isign: typ.Literal[1, -1] = 1,
        eps: float = 1e-6,
        real: bool = False,
        center: typ.Tuple[bool] = (False, False),
        **kwargs,
    ) -> pyco.LinOp:
        r"""
        Type 3 NUFFT (non-uniform to non-uniform).

        Parameters
        ----------
        x: NDArray
            (M, [d]) d-dimensional sample points :math:`\mathbf{x}_{j} \in \mathbb{R}^{d}`.
        z: NDArray
            (N, [d]) d-dimensional query points :math:`\mathbf{z}_{k} \in \mathbb{R}^{d}`.
        isign: 1 | -1
            Sign :math:`\sigma` of the transform.
        eps: float
            Requested accuracy.
        real: bool
            If ``True``, assumes real inputs to the NUFFT.
        center: tuple(bool)
            (2,) boolean tuple. If the first (respectively second) entry is ``True``, an alternative but equivalent NUFFT algorithm performing on
            translated  sample points ``x`` (respectively query points ``z``) is used (see eq. (3.24) of [FINUFFT]_ for a description).
            This can make the inner FFTs less computationally/memory intensive, to the price of a small
            computational overhead with linear complexity in N and/or M. This is especially effective for poorly centered data.
        **kwargs
            Extra keyword parameters to `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_. (Illegal keywords are silently dropped.)
            Most useful is `n_trans`.

        Returns
        -------
        op: LinOp
            An NUFFT operator of type 3 with pre-computed plan.
        """
        init_kwargs = _NUFFT3._sanitize_init_kwargs(x=x, z=z, isign=isign, eps=eps, real=real, center=center, **kwargs)
        return _NUFFT3(**init_kwargs)

    @staticmethod
    def _as_canonical_coordinate(x: pyct.NDArray) -> pyct.NDArray:
        if (N_dim := x.ndim) == 1:
            x = x.reshape((-1, 1))
        elif N_dim == 2:
            assert 1 <= x.shape[-1] <= 3, "Only (1,2,3)-D transforms supported."
        else:
            raise ValueError(f"Expected 1D/2D array, got {N_dim}-D array.")
        return x

    @staticmethod
    def _as_canonical_mode(N) -> tuple[int]:
        if not isinstance(N, cabc.Sequence):
            N = (N,)
        N = tuple(map(int, N))
        assert all(_ > 0 for _ in N), f"The mesh size must be positive for every dimension, got: {N}."
        assert 1 <= len(N) <= 3, "Only (1,2,3)-D transforms supported."
        return N

    @classmethod
    def _sanitize_init_kwargs(cls, **kwargs) -> dict:
        # check init() params + put in standardized form
        raise NotImplementedError

    @staticmethod
    def _plan_fw(**kwargs) -> finufft.Plan:
        # create plan and set points
        raise NotImplementedError

    def _fw(self, arr: pyct.NDArray) -> pyct.NDArray:
        # apply forward operator.
        # input: (n_trans, Q1) complex-valued
        # output: (n_trans, Q2) complex-valued
        raise NotImplementedError

    @staticmethod
    def _plan_bw(**kwargs) -> finufft.Plan:
        # create plan and set points
        raise NotImplementedError

    def _bw(self, arr: pyct.NDArray) -> pyct.NDArray:
        # apply backward operator.
        # input: (n_trans, Q2) complex-valued
        # output: (n_trans, Q1) complex-valued
        raise NotImplementedError

    @staticmethod
    def _preprocess(arr: pyct.NDArray, n_trans: int, dim_out: int):
        # Internal method for apply/adjoint.
        #
        # Parameters
        # ----------
        # arr: NDArray
        #     ([n_trans,], N1) complex-valued input of [apply|adjoint]().
        # n_trans: int
        #     n_trans parameter given to finufft.Plan()
        # dim_out: int
        #     Trailing dimension [apply|adjoint](arr) should have.
        #
        # Returns
        # -------
        # x: NDArray
        #     (N_blk, n_trans, N1) complex-valued blocks to input to [_fw|_bw](), suitably augmented
        #     as needed.
        # N_stack: int
        #     Amount of "valid" data to extract from [_fw|_bw](). {For _postprocess()}
        # sh_out: tuple[int]
        #     Shape [apply|adjoint](arr) should have. {For _postprocess()}
        sh_out = arr.shape[:-1] + (dim_out,)
        if arr.ndim == 1:
            arr = arr.reshape((1, -1))
        N_stack, dim_in = arr.shape

        N_blk, r = divmod(N_stack, n_trans)
        N_blk += 1 if (r > 0) else 0
        if r == 0:
            x = arr
        else:
            xp = pycu.get_array_module(arr)
            x = xp.concatenate([arr, xp.zeros((n_trans - r, dim_in), dtype=arr.dtype)], axis=0)
        x = x.reshape((N_blk, n_trans, dim_in))
        return x, N_stack, sh_out

    @staticmethod
    def _postprocess(blks: list[pyct.NDArray], N_stack: int, sh_out: tuple[int]) -> pyct.NDArray:
        # Internal method for apply/adjoint.
        #
        # Parameters
        # ----------
        # blks: list[NDArray]
        #     (N_blk,) complex-valued outputs of [_fw|_bw]().
        # N_stack: int
        #     Amount of "valid" data to extract from [_fw|_bw]()
        # sh_out: tuple[int]
        #     Shape [apply|adjoint](arr) should have.
        xp = pycu.get_array_module(blks[0])
        return xp.concatenate(blks, axis=0)[:N_stack].reshape(sh_out)


class _NUFFT1(NUFFT):
    def __init__(self, **kwargs):
        self._plan = dict(
            fw=self._plan_fw(**kwargs),
            bw=self._plan_bw(**kwargs),
        )
        self._M, self._D = kwargs["x"].shape  # Useful constants
        self._N = kwargs["N"]
        self._n = self._plan["fw"].n_trans
        self._real_input = kwargs["real_input"]
        self._real_output = kwargs["real_output"]
        super().__init__(
            shape=(
                np.prod(self._N) if self._real_output else 2 * np.prod(self._N),
                self._M if self._real_input else 2 * self._M,
            )
        )  # Complex valued inputs/outputs so dimension is doubled.
        self._lipschitz = np.sqrt(
            self._M * np.prod(self._N) / 2 * np.pi
        )  # Should be called after super().__init__. This is an overestimation.

    @classmethod
    def _sanitize_init_kwargs(cls, **kwargs) -> dict:
        kwargs = kwargs.copy()
        for k in ("nufft_type", "n_modes_or_dim", "dtype"):
            kwargs.pop(k, None)
        x = kwargs["x"] = cls._as_canonical_coordinate(kwargs["x"])
        N = kwargs["N"] = cls._as_canonical_mode(kwargs["N"])
        kwargs["real_input"] = bool(kwargs["real_input"])
        kwargs["real_output"] = bool(kwargs["real_output"])
        if (D := x.shape[-1]) == len(N):
            pass
        elif len(N) == 1:
            kwargs["N"] = N * D
        else:
            raise ValueError("N must have the same lenght as the dimension of the non-uniform samples x.")
        return kwargs

    @staticmethod
    def _plan_fw(**kwargs) -> finufft.Plan:
        kwargs = kwargs.copy()
        x, N = [kwargs.pop(_) for _ in ("x", "N")]
        _, N_dim = x.shape

        plan = finufft.Plan(
            nufft_type=1,
            n_modes_or_dim=N,
            dtype=pycrt.getPrecision().value,
            eps=kwargs.pop("eps", pycrt.getPrecision().eps() * 10),  # provide some slack
            n_trans=kwargs.pop("n_trans", 1),
            isign=kwargs.pop("isign", 1),
            modeord=kwargs.pop("modeord", 0),
            **kwargs,
        )
        plan.setpts(**dict(zip("xyz"[:N_dim], pycu.compute(x.T[:N_dim]))))
        return plan

    @_wrap_if_dask
    def _fw(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._n == 1:  # finufft limitation: insists on having no
            arr = arr[0]  # leading-dim if n_trans==1.
        out = self._plan["fw"].execute(arr)  # ([n_trans], M) -> ([n_trans], N1,..., Nd)
        return out.reshape((self._n, np.prod(self._N)))

    @staticmethod
    def _plan_bw(**kwargs) -> finufft.Plan:
        kwargs = kwargs.copy()
        x, N = [kwargs.pop(_) for _ in ("x", "N")]
        _, N_dim = x.shape

        plan = finufft.Plan(
            nufft_type=2,
            n_modes_or_dim=N,
            dtype=pycrt.getPrecision().value,
            eps=kwargs.pop("eps", pycrt.getPrecision().eps() * 10),  # provide some slack
            n_trans=kwargs.pop("n_trans", 1),
            isign=-kwargs.pop("isign", 1),
            modeord=kwargs.pop("modeord", 0),
            **kwargs,
        )
        plan.setpts(**dict(zip("xyz"[:N_dim], pycu.compute(x.T[:N_dim]))))
        return plan

    @_wrap_if_dask
    def _bw(self, arr: pyct.NDArray) -> pyct.NDArray:
        arr = arr.reshape((self._n, *self._N))
        if self._n == 1:  # finufft limitation: insists on having no
            arr = arr[0]  # leading-dim if n_trans==1.
        out = self._plan["bw"].execute(arr)  # ([n_trans], N1, ..., Nd) -> ([n_trans], M)
        return out.reshape((self._n, self._M))  # req. if squeeze-like behaviour above kicked in.

    @pycrt.enforce_precision("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            ([N_stack], 2M) input weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array (see :py:func:`~pycsou.util.complex.view_as_real`).

        Returns
        -------
        out: NDArray
            ([N_stack], 2N1*...*Nd) output of NUFFT :math:`\mathbf{u} \in \mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}`
            viewed as a real array (see :py:func:`~pycsou.util.complex.view_as_real`).
        """
        if self._real_input:
            r_width = pycrt.Width(arr.dtype)
            arr = arr.astype(r_width.complex.value)
        else:
            arr = pycu.view_as_complex(arr)
        data, N_stack, sh = self._preprocess(arr, self._n, np.prod(self._N))
        blks = [self._fw(blk) for blk in data]
        out = self._postprocess(blks, N_stack, sh)
        return out.real if self._real_output else pycu.view_as_real(out)

    @pycrt.enforce_precision("arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            ([N_stack], 2N1*...Nd) input :math:`\mathbf{u} \in \mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}`
            viewed as a real array (see :py:func:`~pycsou.util.complex.view_as_real`).

        Returns
        -------
        out: NDArray
            ([N_stack], 2M) output of adjoint NUFFT :math:`\mathbf{w} \in \mathbb{C}^{M}`
            viewed as a real array (see :py:func:`~pycsou.util.complex.view_as_real`).
        """
        if self._real_output:
            r_width = pycrt.Width(arr.dtype)
            arr = arr.astype(r_width.complex.value)
        else:
            arr = pycu.view_as_complex(arr)
        data, N_stack, sh = self._preprocess(arr, self._n, self._M)
        blks = [self._bw(blk) for blk in data]
        out = self._postprocess(blks, N_stack, sh)
        return out.real if self._real_input else pycu.view_as_real(out)


class _NUFFT3(NUFFT):
    def __init__(self, **kwargs):
        kwargs = kwargs.copy()
        self._plan = dict(
            fw=self._plan_fw(**kwargs),
            bw=self._plan_bw(**kwargs),
        )
        self._M, self._D = kwargs["x"].shape  # Useful constants
        self._N, _ = kwargs["z"].shape
        self._n = self._plan["fw"].n_trans
        self._real = kwargs["real"]
        self._cx, self._cz = kwargs["center"]
        isign = kwargs.pop("isign", 1)
        if self._cx:  # Do not interchange this two if statements or incorrect
            x_center = (kwargs["x"].min(axis=0) + kwargs["x"].max(axis=0)) / 2
            kwargs["x"] -= x_center
            xp = pycu.get_array_module(kwargs["x"])
            self._postphasing = xp.exp(isign * 1j * (kwargs["z"] * x_center).sum(axis=-1))  # Shape (N,)
        if self._cz:
            z_center = (kwargs["z"].min(axis=0) + kwargs["z"].max(axis=0)) / 2
            kwargs["z"] -= z_center
            xp = pycu.get_array_module(kwargs["z"])
            self._prephasing = xp.exp(isign * 1j * (kwargs["x"] * z_center).sum(axis=-1))  # Shape (M,)
        super().__init__(shape=(2 * self._N, self._M if self._real else 2 * self._M))
        self._lipschitz = np.sqrt(self._N * self._M)  # Overestimation via Frobenius norm

    @classmethod
    def _sanitize_init_kwargs(cls, **kwargs) -> dict:
        kwargs = kwargs.copy()
        for k in ("nufft_type", "n_modes_or_dim", "dtype"):
            kwargs.pop(k, None)
        x = kwargs["x"] = cls._as_canonical_coordinate(kwargs["x"])
        z = kwargs["z"] = cls._as_canonical_coordinate(kwargs["z"])
        kwargs["real"] = bool(kwargs["real"])
        kwargs["center"] = [bool(_) for _ in kwargs["center"]]
        assert x.shape[-1] == z.shape[-1], "Dimensionality mis-match between sample and query points."
        return kwargs

    @staticmethod
    def _plan_fw(**kwargs) -> finufft.Plan:
        kwargs = kwargs.copy()
        x, z = [kwargs.pop(_) for _ in ("x", "z")]
        _, N_dim = x.shape

        plan = finufft.Plan(
            nufft_type=3,
            n_modes_or_dim=N_dim,
            dtype=pycrt.getPrecision().value,
            eps=kwargs.pop("eps", pycrt.getPrecision().eps() * 10),  # provide some slack
            n_trans=kwargs.pop("n_trans", 1),
            isign=kwargs.pop("isign", 1),
            **kwargs,
        )
        plan.setpts(
            **dict(
                zip(
                    "xyz"[:N_dim] + "stu"[:N_dim],
                    pycu.compute(*x.T[:N_dim], *z.T[:N_dim]),
                )
            ),
        )
        return plan

    @_wrap_if_dask
    def _fw(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._n == 1:  # finufft limitation: insists on having no
            arr = arr[0]  # leading-dim if n_trans==1.
        out = self._plan["fw"].execute(arr)  # ([n_trans], M) -> ([n_trans], N)
        return out.reshape((self._n, self._N))

    @staticmethod
    def _plan_bw(**kwargs) -> finufft.Plan:
        kwargs = kwargs.copy()
        x, z = [kwargs.pop(_) for _ in ("x", "z")]
        _, N_dim = x.shape

        plan = finufft.Plan(
            nufft_type=3,
            n_modes_or_dim=N_dim,
            dtype=pycrt.getPrecision().value,
            eps=kwargs.pop("eps", pycrt.getPrecision().eps() * 10),  # provide some slack
            n_trans=kwargs.pop("n_trans", 1),
            isign=-kwargs.pop("isign", 1),
            **kwargs,
        )
        plan.setpts(
            **dict(
                zip(
                    "xyz"[:N_dim] + "stu"[:N_dim],
                    pycu.compute(*z.T[:N_dim], *x.T[:N_dim]),
                )
            ),
        )
        return plan

    @_wrap_if_dask
    def _bw(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._n == 1:  # finufft limitation: insists on having no
            arr = arr[0]  # leading-dim if n_trans==1.
        out = self._plan["bw"].execute(arr)  # ([n_trans,] N) -> ([n_trans,] M)
        return out.reshape((self._n, self._M))

    @pycrt.enforce_precision("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            ([N_stack], 2M) input weights :math:`\mathbf{w} \in \mathbb{C}^{M}`
            viewed as a real array (see :py:func:`~pycsou.util.complex.view_as_real`).

        Returns
        -------
        out: NDArray
            ([N_stack], 2N) output of NUFFT :math:`\mathbf{v} \in \mathbb{C}^{N}`
            viewed as a real array (see :py:func:`~pycsou.util.complex.view_as_real`).
        """
        if self._real:
            r_width = pycrt.Width(arr.dtype)
            arr = arr.astype(r_width.complex.value)
        else:
            arr = pycu.view_as_complex(arr)
        if self._cz:
            arr *= self._prephasing  # Automatic casting to type of arr
        data, N_stack, sh = self._preprocess(arr, self._n, self._N)
        blks = [self._fw(blk) for blk in data]
        out = self._postprocess(blks, N_stack, sh)
        if self._cx:
            out *= self._postphasing  # Automatic casting to type of arr
        return pycu.view_as_real(out)

    @pycrt.enforce_precision("arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            ([N_stack], 2N) input :math:`\mathbf{v} \in \mathbb{C}^{N}`
            viewed as a real array (see :py:func:`~pycsou.util.complex.view_as_real`).

        Returns
        -------
        out: NDArray
            ([N_stack], 2M) output of adjoint NUFFT :math:`\mathbf{w} \in \mathbb{C}^{M}`
            viewed as a real array (see :py:func:`~pycsou.util.complex.view_as_real`).
        """
        arr = pycu.view_as_complex(arr)
        if self._cx:
            arr *= self._postphasing.conj()
        data, N_stack, sh = self._preprocess(arr, self._n, self._M)
        blks = [self._bw(blk) for blk in data]
        out = self._postprocess(blks, N_stack, sh)
        if self._cz:
            out *= self._prephasing.conj()
        return out.real if self._real else pycu.view_as_real(out)
