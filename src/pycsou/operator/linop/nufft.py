"""
Working notes on NUFFT interface for pycsou.
"""

import collections.abc as cabc
import math
import typing as typ

import finufft
import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "NUFFT",
]


class NUFFT(pyco.LinOp):
    """
    Non-Uniform Fast Fourier Transform (Type 1/2/3).

    Use custom class constructors to create the desired transform.
    """

    # The goal of this wrapper class is to sanitize __init__() inputs.

    def __init__(self, shape: pyct.NonAgnosticShape):
        super().__init__(shape)

    @staticmethod
    @pycrt.enforce_precision(i="t", o=False, allow_None=False)
    def type1(
        t: pyct.NDArray,
        M: typ.Union[int, tuple[int, ...]],
        **kwargs,
    ) -> pyco.LinOp:
        r"""
        Type-1 NUFFT.

        Performs the following computation:

        .. math::

           \alpha_{k}^{F} = \sum_{j=1}^{J} \alpha_{j} \exp^{i s \langle f_{k}, t_{j} \rangle },

        where :math:`s \in \{+1, -1\}`, :math:`\alpha_{j} \in \mathbb{C}`, :math:`t_{j} \in
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
        return _NUFFT1(**init_kwargs)

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

        where :math:`s \in \{+1, -1\}`, :math:`\alpha_{j} \in \mathbb{C}`, :math:`t_{j} \in
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
        #     if needed.
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
        plan.setpts(**dict(zip("xyz"[:N_dim], t.T[:N_dim])))
        return plan

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
        plan.setpts(**dict(zip("xyz"[:N_dim], t.T[:N_dim])))
        return plan

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

        Note
        ----
        :py:class:`~pycsou.operator.linop.nufft._NUFFT1` pre-allocates resources to do `n_trans`
        transforms simultaneously. Runtime is thus determined at the planning stage. I.e., if you
        supply less that `n_trans` inputs, then computation time will still be identical.
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

        Note
        ----
        :py:class:`~pycsou.operator.linop.nufft._NUFFT1` pre-allocates resources to do `n_trans`
        transforms simultaneously. Runtime is thus determined at the planning stage. I.e., if you
        supply less that `n_trans` inputs, then computation time will still be identical.
        """
        arr = pycu.view_as_complex(arr)
        data, N, sh = self._preprocess(arr, self._N, self._J)
        blks = [self._bw(blk) for blk in data]
        out = self._postprocess(blks, N, sh)
        return pycu.view_as_real(out)


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
                    (*t.T[:N_dim], *f.T[:N_dim]),
                )
            ),
        )
        return plan

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
                    (*f.T[:N_dim], *t.T[:N_dim]),
                )
            ),
        )
        return plan

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

        Note
        ----
        :py:class:`~pycsou.operator.linop.nufft._NUFFT3` pre-allocates resources to do `n_trans`
        transforms simultaneously. Runtime is thus determined at the planning stage. I.e., if you
        supply less that `n_trans` inputs, then computation time will still be identical.
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

        Note
        ----
        :py:class:`~pycsou.operator.linop.nufft._NUFFT3` pre-allocates resources to do `n_trans`
        transforms simultaneously. Runtime is thus determined at the planning stage. I.e., if you
        supply less that `n_trans` inputs, then computation time will still be identical.
        """
        arr = pycu.view_as_complex(arr)
        data, N, sh = self._preprocess(arr, self._N, self._J)
        blks = [self._bw(blk) for blk in data]
        out = self._postprocess(blks, N, sh)
        return pycu.view_as_real(out)
