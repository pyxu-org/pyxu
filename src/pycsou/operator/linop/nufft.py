import collections.abc as cabc
import math
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
    The various transforms can be instantiated via the custom class constructors :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type1`, :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type2`, and :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type3`
    respectively (see also :py:meth:`~pycsou.operator.linop.nufft.NUFFT.rtype1`, :py:meth:`~pycsou.operator.linop.nufft.NUFFT.rtype2` and :py:meth:`~pycsou.operator.linop.nufft.NUFFT.rtype3` for real inputs).
    The dimension of the NUFFT transforms is inferred from the dimensions of the input arguments, with support for dimensions 1, 2 and 3.

    Notes
    -----
    We adopt here the same notational conventions as in [FINUFFT]_.

    *Mathematical Definition*

    Let :math:`d\in\{1,2,3\}` and consider the Cartesian-product mesh

    .. math::

        \mathcal{I}_{N_1,\ldots, N_d}=\mathcal{I}_{N_1}\times \cdots\times  \mathcal{I}_{N_d}\subset\mathbb{Z}^d

    of the hyperinterval :math:`\Pi_{i=1}^d [-N_i/2,N_i/2]\subset \mathbb{R}^d`, where the mesh indices :math:`\mathcal{I}_{N_i}\subset\mathbb{Z}` are given for each
    dimension :math:`i=1,\dots, d` by:

    .. math::

        \mathcal{I}_{N_i}=\begin{cases}\{-N_i/2, \ldots, N_i/2-1\}, & N_i\in 2\mathbb{Z} \text{ (even)}, \\  \{-(N_i-1)/2, \ldots, (N_i-1)/2\}, & N_i\in 2\mathbb{Z}+1 \text{ (odd)}.\end{cases}


    Then, the various NUFFT operators approximate, up to a requested relative accuracy :math:`\varepsilon>0`, the following exponential sums respectively:

        - *NUFFT of Type 1 (non-uniform to uniform):*

            .. math::

                u_{\mathbf{n}} = \sum_{j=1}^{M} w_{j} e^{\sigma i\langle \mathbf{n}, \mathbf{x}_{j} \rangle}, \quad \mathbf{n}\in \mathcal{I}_{N_1,\ldots, N_d}, \tag{1}


        - *NUFFT of Type 2 (uniform to non-uniform):*

            .. math::

                w_{j} = \sum_{\mathbf{n}\in\mathcal{I}_{N_1,\ldots, N_d}} u_{\mathbf{n}} e^{\sigma i\langle \mathbf{n}, \mathbf{x}_{j} \rangle }, \quad j=1,\ldots, M, \tag{2}


        - *NUFFT of Type 3 (non-uniform to non-uniform):*

            .. math::

                v_{k} = \sum_{j=1}^{M} w_{j} e^{\sigma i\langle \mathbf{z}_k, \mathbf{x}_{j} \rangle }, \quad k=1,\ldots, N, \tag{3}

    where :math:`\sigma \in \{+1, -1\}` defines the sign of the transforms, and where :math:`u_{\mathbf{n}}, v_{k}, w_{j}\in \mathbb{C}` are complex coefficients.
    For the NUFFTs of Types 1 and 2, the non-uniform samples :math:`\mathbf{x}_{j}` are assumed to lie in :math:`[-\pi,\pi)^d`. For the NUFFT of Type 3, the non-uniform samples
    :math:`\mathbf{x}_{j}` and :math:`\mathbf{z}_{k}` are arbitrary points in :math:`\mathbb{R}^d`.

    *Adjoint NUFFTs*

    The NUFFTs of Types 1 and 2 with opposite signs form an adjoint pair. The adjoint of the NUFFT of Type 3 is obtained by
    flipping the transform's sign and switching the roles of   :math:`\mathbf{z}_k` and :math:`\mathbf{x}_{j}` in (3).

    *Accuracy*

    *Complexity*

    *Backend*

    The NUFFT tansforms are computed via the CPU-based and multithreaded C++ library `FINUFFT <https://github.com/flatironinstitute/finufft>`_ and its CUDA equivalent `cuFINUFFT <https://github.com/flatironinstitute/cufinufft>`_ (see also [FINUFFT]_ and [cuFINUFFT]_).
    It uses minimal RAM, and performs the expensive spreading/interpolation between nonuniform points and the fine grid via the “exponential of semicircle” kernel in a cache-aware load-balanced multithreaded implementation.
    This kernel is simpler and faster to evaluate than the Kaiser–Bessel, yet has essentially identical error (see [FINUFFT]_).

    """

    # The goal of this wrapper class is to sanitize __init__() inputs.

    def __init__(self, shape: pyct.NonAgnosticShape):
        super().__init__(shape)

    @staticmethod
    @pycrt.enforce_precision(i="t", o=False, allow_None=False)
    def type1(
        x: pyct.NDArray,
        N: typ.Union[int, tuple[int, ...]],
        **kwargs,
    ) -> pyco.LinOp:
        r"""
        Type 1 NUFFT (nonuniform to uniform).

        Approximates the following computation:

        .. math::

           \beta_{\mathbf{n}} = \sum_{j=1}^{M} \alpha_{j} e^{i s \langle \mathbf{n}, \mathbf{x}_{j} \rangle },

        where :math:`s \in \{+1, -1\}`, :math:`(\alpha_{j})_j \in \mathbb{C}^M`, :math:`\{\mathbf{x}_{j}\}_j \subset
        \mathbb{R}^{D}`, and :math:`(\beta_{\mathbf{n}})_{\mathbf{n}\in\mathcal{I}} \in \mathbb{C}^{\mathcal{I}}` with
        :math:`f_{k} \in [-M_{1}/2, \ldots, (M_{1}-1)/2] \times \cdots \times [-M_{D}/2, \ldots, (M_{D}-1)/2]`.

        Parameters
        ----------
        t: NDArray
            (J, [D]) D-dimensional spatial coordinates :math:`t_{j} \in \mathbb{R}^{D}`.
        M: int | tuple[int]
            ([D],) number of Fourier modes per dimension. An integer parameter applies to each
            dimension.
        **kwargs
            Extra keyword parameters to :py:func:`finufft.Plan`. (Illegal keywords are dropped.)
            Most useful are `isign`, `n_trans` and `eps`.

        Returns
        -------
        op: LinOp
        """
        init_kwargs = _NUFFT1._sanitize_init_kwargs(t=t, M=M, **kwargs)
        return _NUFFT1(**init_kwargs)

    @staticmethod
    @pycrt.enforce_precision(i="t", o=False, allow_None=False)
    def rtype1(
        t: pyct.NDArray,
        M: typ.Union[int, tuple[int, ...]],
        **kwargs,
    ) -> pyco.LinOp:
        r"""
        Type-1 NUFFT for real-valued inputs. [Syntactic sugar for .type1(): no performance
        advantage.]

        Performs the following computation:

        .. math::

           \alpha_{k}^{F} = \sum_{j=1}^{J} \alpha_{j} \exp^{i s \langle f_{k}, t_{j} \rangle },

        where :math:`s \in \{+1, -1\}`, :math:`\alpha_{j} \in \mathbb{R}`, :math:`t_{j} \in
        \mathbb{R}^{D}`, and :math:`f_{k} \in [-M_{1}/2, \ldots, (M_{1}-1)/2] \times \cdots \times
        [-M_{D}/2, \ldots, (M_{D}-1)/2]`.

        Parameters
        ----------
        t: NDArray
            (J, [D]) D-dimensional spatial coordinates :math:`t_{j} \in \mathbb{R}^{D}`.
        M: int | tuple[int]
            (D,) number of Fourier modes per dimension. An integer parameter applies to each
            dimension.
        **kwargs
            Extra keyword parameters to :py:func:`finufft.Plan`. (Illegal keywords are dropped.)
            Most useful are `isign`, `n_trans` and `eps`.

        Returns
        -------
        op: LinOp
        """
        init_kwargs = _rNUFFT1._sanitize_init_kwargs(t=t, M=M, **kwargs)
        return _rNUFFT1(**init_kwargs)

    @staticmethod
    @pycrt.enforce_precision(i="t", o=False, allow_None=False)
    def type2(
        t: pyct.NDArray,
        M: typ.Union[int, tuple[int, ...]],
        **kwargs,
    ) -> pyco.LinOp:
        r"""
        Type-2 NUFFT, adjoint of the Type-1 NUFFT.

        Performs the following computation:

        .. math::

           \alpha_{j} = \sum_{k \in \mathbb{Z}^{D}} \alpha_{k}^{F} \exp^{i s \langle f_{k}, t_{j} \rangle },

        where :math:`s \in \{+1, -1\}`, :math:`\alpha_{k}^{F} \in \mathbb{C}`, :math:`t_{j} \in
        \mathbb{R}^{D}`, and :math:`f_{k} \in [-M_{1}/2, \ldots, (M_{1}-1)/2] \times \cdots \times
        [-M_{D}/2, \ldots, (M_{D}-1)/2]`.

        Parameters
        ----------
        t: NDArray
            (J, [D]) D-dimensional spatial coordinates :math:`t_{j} \in \mathbb{R}^{D}`.
        M: int | tuple[int]
            (D,) number of Fourier modes per dimension. An integer parameter applies to each
            dimension.
        **kwargs
            Extra keyword parameters to :py:func:`finufft.Plan`. (Illegal keywords are dropped.)
            Most useful are `isign`, `n_trans` and `eps`.

        Returns
        -------
        op: LinOp
        """
        init_kwargs = _NUFFT1._sanitize_init_kwargs(t=t, M=M, **kwargs)
        init_kwargs["isign"] = -init_kwargs.get("isign", 1)
        return _NUFFT1(**init_kwargs).T

    @staticmethod
    @pycrt.enforce_precision(i=("t", "f"), o=False, allow_None=False)
    def type3(
        t: pyct.NDArray,
        f: pyct.NDArray,
        **kwargs,
    ) -> pyco.LinOp:
        r"""
        Type-3 NUFFT.

        Performs the following computation:

        .. math::

           \alpha_{k}^{F} = \sum_{j=1}^{J} \alpha_{j} \exp^{i s \langle f_{k}, t_{j} \rangle },

        where :math:`s \in \{+1, -1\}`, :math:`\alpha_{j} \in \mathbb{C}`, :math:`t_{j} \in
        \mathbb{R}^{D}`, and :math:`f_{k} \in \mathbb{R}^{D}`.

        Parameters
        ----------
        t: NDArray
            (J, [D]) D-dimensional spatial coordinates :math:`t_{j} \in \mathbb{R}^{D}`.
        f: NDArray
            (K, [D]) D-dimensional spectral coordinates :math:`f_{k} \in \mathbb{R}^{D}`.
        kwargs
            Extra keyword-arguments to :py:func:`finufft.Plan`. (Illegal keywords are dropped.)
            Most useful are `isign`, `n_trans` and `eps`.

        Returns
        -------
        op: LinOp
        """
        init_kwargs = _NUFFT3._sanitize_init_kwargs(t=t, f=f, **kwargs)
        return _NUFFT3(**init_kwargs)

    @staticmethod
    @pycrt.enforce_precision(i=("t", "f"), o=False, allow_None=False)
    def rtype3(
        t: pyct.NDArray,
        f: pyct.NDArray,
        **kwargs,
    ) -> pyco.LinOp:
        r"""
        Type-3 NUFFT for real-valued inputs. [Syntactic sugar for .type3(): no performance
        advantage.]

        Performs the following computation:

        .. math::

           \alpha_{k}^{F} = \sum_{j=1}^{J} \alpha_{j} \exp^{i s \langle f_{k}, t_{j} \rangle },

        where :math:`s \in \{+1, -1\}`, :math:`\alpha_{j} \in \mathbb{R}`, :math:`t_{j} \in
        \mathbb{R}^{D}`, and :math:`f_{k} \in \mathbb{R}^{D}`.

        Parameters
        ----------
        t: NDArray
            (J, [D]) D-dimensional spatial coordinates :math:`t_{j} \in \mathbb{R}^{D}`.
        f: NDArray
            (K, [D]) D-dimensional spectral coordinates :math:`f_{k} \in \mathbb{R}^{D}`.
        kwargs
            Extra keyword-arguments to :py:func:`finufft.Plan`. (Illegal keywords are dropped.)
            Most useful are `isign`, `n_trans` and `eps`.

        Returns
        -------
        op: LinOp
        """
        init_kwargs = _rNUFFT3._sanitize_init_kwargs(t=t, f=f, **kwargs)
        return _rNUFFT3(**init_kwargs)

    @staticmethod
    def _as_canonical_coordinate(x: pyct.NDArray) -> pyct.NDArray:
        if (N_dim := x.ndim) == 1:
            x = x.reshape((1, -1))
        elif N_dim == 2:
            assert 1 <= x.shape[-1] <= 3, "Only (1,2,3)-D transforms supported."
        else:
            raise ValueError(f"Expected 1D/2D array, got {N_dim}-D array.")
        return x

    @staticmethod
    def _as_canonical_mode(M) -> tuple[int]:
        if not isinstance(M, cabc.Sequence):
            M = (M,)
        M = tuple(map(int, M))
        assert all(_ > 0 for _ in M)
        assert 1 <= len(M) <= 3, "Only (1,2,3)-D transforms supported."
        return M

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
        #     ([N,], N1) complex-valued input of [apply|adjoint]().
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
        # N: int
        #     Amount of "valid" data to extract from [_fw|_bw](). {For _postprocess()}
        # sh_out: tuple[int]
        #     Shape [apply|adjoint](arr) should have. {For _postprocess()}
        sh_out = arr.shape[:-1] + (dim_out,)
        if arr.ndim == 1:
            arr = arr.reshape((1, -1))
        N, dim_in = arr.shape

        N_blk, r = divmod(N, n_trans)
        N_blk += 1 if (r > 0) else 0
        if r == 0:
            x = arr
        else:
            xp = pycu.get_array_module(arr)
            x = xp.concatenate([arr, xp.zeros((n_trans - r, dim_in), dtype=arr.dtype)], axis=0)
        x = x.reshape((N_blk, n_trans, dim_in))
        return x, N, sh_out

    @staticmethod
    def _postprocess(blks: list[pyct.NDArray], N: int, sh_out: tuple[int]) -> pyct.NDArray:
        # Internal method for apply/adjoint.
        #
        # Parameters
        # ----------
        # blks: list[NDArray]
        #     (N_blk,) complex-valued outputs of [_fw|_bw]().
        # N: int
        #     Amount of "valid" data to extract from [_fw|_bw]()
        # sh_out: tuple[int]
        #     Shape [apply|adjoint](arr) should have.
        xp = pycu.get_array_module(blks[0])
        return xp.concatenate(blks, axis=0)[:N].reshape(sh_out)


class _NUFFT1(NUFFT):
    def __init__(self, **kwargs):
        self._lipschitz = 0  # TODO: [Sepand -> Matthieu] Closed-form possible?
        self._plan = dict(
            fw=self._plan_fw(**kwargs),
            bw=self._plan_bw(**kwargs),
        )
        self._J, self._D = kwargs["t"].shape  # Useful constants
        self._M = kwargs["M"]
        self._N = self._plan["fw"].n_trans
        super().__init__(shape=(2 * np.prod(self._M), 2 * self._J))

    @classmethod
    def _sanitize_init_kwargs(cls, **kwargs) -> dict:
        kwargs = kwargs.copy()
        for k in ("nufft_type", "n_modes_or_dim", "dtype", "modeord"):
            kwargs.pop(k, None)
        t = kwargs["t"] = cls._as_canonical_coordinate(kwargs["t"])
        xp = pycu.get_array_module(t)
        t = kwargs["t"] = xp.fmod(t, 2 * np.pi)
        M = kwargs["M"] = cls._as_canonical_mode(kwargs["M"])
        if (D := t.shape[-1]) == len(M):
            pass
        elif len(M) == 1:
            kwargs["M"] = M * D
        else:
            raise ValueError("Spatial/Fourier-mode dimensionality mis-match")
        return kwargs

    @staticmethod
    def _plan_fw(**kwargs) -> finufft.Plan:
        kwargs = kwargs.copy()
        t, M = [kwargs.pop(_) for _ in ("t", "M")]
        _, N_dim = t.shape

        plan = finufft.Plan(
            nufft_type=1,
            n_modes_or_dim=M,
            dtype=pycrt.getPrecision().value,
            eps=kwargs.pop("eps", pycrt.getPrecision().eps() * 10),  # provide some slack
            n_trans=kwargs.pop("n_trans", 1),
            isign=kwargs.pop("isign", 1),
            modeord=0,
            **kwargs,
        )
        plan.setpts(**dict(zip("xyz"[:N_dim], pycu.compute(t.T[:N_dim]))))
        return plan

    @_wrap_if_dask
    def _fw(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._N == 1:  # finufft limitation: insists on having no
            arr = arr[0]  # leading-dim if n_trans==1.
        out = self._plan["fw"].execute(arr)  # ([N], J) -> ([N], M1, ..., MD)
        return out.reshape((self._N, np.prod(self._M)))

    @staticmethod
    def _plan_bw(**kwargs) -> finufft.Plan:
        kwargs = kwargs.copy()
        t, M = [kwargs.pop(_) for _ in ("t", "M")]
        _, N_dim = t.shape

        plan = finufft.Plan(
            nufft_type=2,
            n_modes_or_dim=M,
            dtype=pycrt.getPrecision().value,
            eps=kwargs.pop("eps", pycrt.getPrecision().eps() * 10),  # provide some slack
            n_trans=kwargs.pop("n_trans", 1),
            isign=-kwargs.pop("isign", 1),
            modeord=0,
            **kwargs,
        )
        plan.setpts(**dict(zip("xyz"[:N_dim], pycu.compute(t.T[:N_dim]))))
        return plan

    @_wrap_if_dask
    def _bw(self, arr: pyct.NDArray) -> pyct.NDArray:
        arr = arr.reshape((self._N, *self._M))
        if self._N == 1:  # finufft limitation: insists on having no
            arr = arr[0]  # leading-dim if n_trans==1.
        out = self._plan["bw"].execute(arr)  # ([N], M1, ..., MD) -> ([N], J)
        return out.reshape((self._N, self._J))  # req. if squeeze-like behaviour above kicked in.

    @pycrt.enforce_precision("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            ([N], 2J) spatial amplitudes :math:`\alpha_{j} \in \mathbb{C}^{J}`.

        Returns
        -------
        out: NDArray
            ([N], 2\prod(M)) spectral mode amplitudes :math:`\alpha_{k}^{F} \in \mathbb{C}^{M_{1}
            \times \cdots \times M_{D}}`.
            D-dimensional modes are C-ordered.
        """
        arr = pycu.view_as_complex(arr)
        data, N, sh = self._preprocess(arr, self._N, np.prod(self._M))
        blks = [self._fw(blk) for blk in data]
        out = self._postprocess(blks, N, sh)
        return pycu.view_as_real(out)

    @pycrt.enforce_precision("arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            ([N], 2\prod(M)) spectral mode amplitudes :math:`\alpha_{k}^{F} \in \mathbb{C}^{M_{1}
            \times \cdots \times M_{D}}`.
            D-dimensional modes are C-ordered.

        Returns
        -------
        out: NDArray
            ([N], 2J) spatial amplitudes :math:`\alpha_{j} \in \mathbb{C}^{J}`.
        """
        arr = pycu.view_as_complex(arr)
        data, N, sh = self._preprocess(arr, self._N, self._J)
        blks = [self._bw(blk) for blk in data]
        out = self._postprocess(blks, N, sh)
        return pycu.view_as_real(out)


class _rNUFFT1(_NUFFT1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._shape = (2 * np.prod(self._M), self._J)

    @pycrt.enforce_precision("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            ([N], J) spatial amplitudes :math:`\alpha_{j} \in \mathbb{R}^{J}`.

        Returns
        -------
        out: NDArray
            ([N], 2\prod(M)) spectral mode amplitudes :math:`\alpha_{k}^{F} \in \mathbb{C}^{M_{1}
            \times \cdots \times M_{D}}`.
            D-dimensional modes are C-ordered.
        """
        r_width = pycrt.Width(arr.dtype)
        arr = arr.astype(r_width.complex.value)
        return super().apply(pycu.view_as_real(arr))

    @pycrt.enforce_precision("arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            ([N], 2\prod(M)) spectral mode amplitudes :math:`\alpha_{k}^{F} \in \mathbb{C}^{M_{1}
            \times \cdots \times M_{D}}`.
            D-dimensional modes are C-ordered.

        Returns
        -------
        out: NDArray
            ([N], J) spatial amplitudes :math:`\alpha_{j} \in \mathbb{R}^{J}`.
        """
        return super().adjoint(arr)[..., ::2]


class _NUFFT3(NUFFT):
    def __init__(self, **kwargs):
        self._lipschitz = 0  # TODO: [Sepand -> Matthieu] Closed-form possible?
        self._plan = dict(
            fw=self._plan_fw(**kwargs),
            bw=self._plan_bw(**kwargs),
        )
        self._J, self._D = kwargs["t"].shape  # Useful constants
        self._K, _ = kwargs["f"].shape
        self._N = self._plan["fw"].n_trans
        super().__init__(shape=(2 * self._K, 2 * self._J))

    @classmethod
    def _sanitize_init_kwargs(cls, **kwargs) -> dict:
        kwargs = kwargs.copy()
        for k in ("nufft_type", "n_modes_or_dim", "dtype"):
            kwargs.pop(k, None)
        t = kwargs["t"] = cls._as_canonical_coordinate(kwargs["t"])
        f = kwargs["f"] = cls._as_canonical_coordinate(kwargs["f"])
        assert t.shape[-1] == f.shape[-1], "Spatial/Spectral dimensionality mis-match."
        assert pycu.get_array_module(t) == pycu.get_array_module(f)
        return kwargs

    @staticmethod
    def _plan_fw(**kwargs) -> finufft.Plan:
        kwargs = kwargs.copy()
        t, f = [kwargs.pop(_) for _ in ("t", "f")]
        _, N_dim = t.shape

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
                    pycu.compute(*t.T[:N_dim], *f.T[:N_dim]),
                )
            ),
        )
        return plan

    @_wrap_if_dask
    def _fw(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._N == 1:  # finufft limitation: insists on having no
            arr = arr[0]  # leading-dim if n_trans==1.
        out = self._plan["fw"].execute(arr)  # ([N], J) -> ([N], K)
        return out.reshape((self._N, self._K))

    @staticmethod
    def _plan_bw(**kwargs) -> finufft.Plan:
        kwargs = kwargs.copy()
        t, f = [kwargs.pop(_) for _ in ("t", "f")]
        _, N_dim = t.shape

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
                    pycu.compute(*f.T[:N_dim], *t.T[:N_dim]),
                )
            ),
        )
        return plan

    @_wrap_if_dask
    def _bw(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._N == 1:  # finufft limitation: insists on having no
            arr = arr[0]  # leading-dim if n_trans==1.
        out = self._plan["bw"].execute(arr)  # ([N,] K) -> ([N,] J)
        return out.reshape((self._N, self._J))

    @pycrt.enforce_precision("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            ([N], 2J) spatial amplitudes :math:`\alpha_{j} \in \mathbb{C}^{J}`.

        Returns
        -------
        out: NDArray
            ([N], 2K) spectral amplitudes :math:`\alpha_{k}^{F} \in \mathbb{C}^{K}`.
        """
        arr = pycu.view_as_complex(arr)
        data, N, sh = self._preprocess(arr, self._N, self._K)
        blks = [self._fw(blk) for blk in data]
        out = self._postprocess(blks, N, sh)
        return pycu.view_as_real(out)

    @pycrt.enforce_precision("arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            ([N], 2K) spectral amplitudes :math:`\alpha_{k}^{F} \in \mathbb{C}^{K}`.

        Returns
        -------
        out: NDArray
            ([N], 2J) spatial amplitudes :math:`\alpha_{j} \in \mathbb{C}^{J}`.
        """
        arr = pycu.view_as_complex(arr)
        data, N, sh = self._preprocess(arr, self._N, self._J)
        blks = [self._bw(blk) for blk in data]
        out = self._postprocess(blks, N, sh)
        return pycu.view_as_real(out)


class _rNUFFT3(_NUFFT3):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._shape = (2 * self._K, self._J)

    @pycrt.enforce_precision("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            ([N], J) spatial amplitudes :math:`\alpha_{j} \in \mathbb{R}^{J}`.

        Returns
        -------
        out: NDArray
            ([N], 2K) spectral amplitudes :math:`\alpha_{k}^{F} \in \mathbb{C}^{K}`.
        """
        r_width = pycrt.Width(arr.dtype)
        arr = arr.astype(r_width.complex.value)
        return super().apply(pycu.view_as_real(arr))

    @pycrt.enforce_precision("arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            ([N], 2K) spectral amplitudes :math:`\alpha_{k}^{F} \in \mathbb{C}^{K}`.

        Returns
        -------
        out: NDArray
            ([N], J) spatial amplitudes :math:`\alpha_{j} \in \mathbb{R}^{J}`.
        """
        return super().adjoint(arr)[..., ::2]
