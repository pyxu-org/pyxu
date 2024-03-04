import collections
import collections.abc as cabc
import concurrent.futures as cf
import string
import warnings

import numpy as np

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.math.cluster as pxm_cl
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "UniformSpread",
]


class UniformSpread(pxa.LinOp):
    r"""
    :math:`D`-dimensional spreading operator :math:`A: \mathbb{R}^{M} \to \mathbb{R}^{N_{1} \times
    \cdots \times N_{D}}`:

    .. math::

       (A \, \mathbf{w})[n_{1}, \ldots, n_{D}]
       =
       \sum_{m = 1}^{M} w_{m} \phi(z_{n_{1}, \ldots, n_{D}} - x_{m}),

    .. math::

       (A^{*} \mathbf{v})_{m}
       =
       \sum_{n_{1}, \ldots, n_{D} = 1}^{N_{1}, \ldots, N_{D}}
       v[n_{1}, \ldots, n_{D}] \phi(z_{n_{1}, \ldots, n_{D}} - x_{m}),

    .. math::

       \mathbf{w} \in \mathbb{R}^{M},
       \quad
       \mathbf{v} \in \mathbb{R}^{N_{1} \times\cdots\times N_{D}},
       \quad
       z_{n_{1},\ldots,n_{D}} \in \mathcal{D},
       \quad
       \phi: \mathcal{K} \to \mathbb{R},

    where
    :math:`\mathcal{D} = [\alpha_{1}, \beta_{1}] \times\cdots\times [\alpha_{D}, \beta_{D}]` and
    :math:`\mathcal{K} = [-s_{1}, s_{1}] \times\cdots\times [-s_{D}, s_{D}]`,
    :math:`s_{d} > 0`.


    .. rubric:: Implementation Notes

    * :py:class:`~pyxu.operator.UniformSpread` is not **precision-agnostic**: it will only work on NDArrays with the
      same dtype as `x`.  A warning is emitted if inputs must be cast to the support dtype.
    * :py:class:`~pyxu.operator.UniformSpread` instances are **not arraymodule-agnostic**: they will only work with
      NDArrays belonging to the same array module as `x`. Only NUMPY/DASK backends are currently supported.
    * Spread/interpolation are performed efficiently via the algorithm described in [FINUFFT]_, i.e. partition
      :math:`\{\mathbf{x}_{m}\}` into sub-grids, spread onto each sub-grid, then add the results to the global grid.
      This approach works best when the kernel is *localized*. For kernels with huge support (w.r.t the full grid), spreading via a tensor contraction is preferable.
    * The domain is partitioned using a kd-tree from SciPy.
    * The kd-tree is built at init-time only when `x` is a NUMPY array.
      For DASK arrays, the tree is built online on subsets of the data to be spreaded.

    """

    def __init__(
        self,
        x: pxt.NDArray,
        z: dict,
        kernel: cabc.Sequence[pxt.OpT],
        enable_warnings: bool = True,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        x: NDArray
            (M, D) support points :math:`\{x_{1},\ldots,x_{M}\}`.
        z: dict
            Lattice specifier, with keys:

            * `start`: (D,) values :math:`\{\alpha_{1}, \ldots, \alpha_{D}\} \in \mathbb{R}`.
            * `stop` : (D,) values :math:`\{\beta_{1}, \ldots, \beta_{D}\} \in \mathbb{R}`.
            * `num`  : (D,) values :math:`\{N_{1}, \ldots, N_{D}\} \in \mathbb{N}^{*}`.

            Scalars are broadcasted to all dimensions.

            The lattice is defined as:

            .. math::

               \left[z_{n_{1}, \ldots, n_{D}}\right]_{d}
               =
               \alpha_{d} + \frac{\beta_{d} - \alpha_{d}}{N_{d} - 1} n_{d},
               \quad
               n_{d} \in \{0, \ldots, N_{d}-1\}

        kernel: list[OpT]
            (D,) seperable kernel specifiers :math:`\phi_{d}: \mathcal{K}_{d} \to \mathbb{R}` such that

            .. math::

               \phi(\mathbf{x}) = \prod_{d=1}^{D} \phi_{d}(x_{d}).

            Functions should be ufuncs with same semantics as :py:class:`~pyxu.abc.Map`, i.e. have a
            :py:meth:`~pyxu.abc.Map.__call__` method.  In addition each kernel should have a ``support()`` method with
            the following signature:

            .. code-block:: python3

               def support(self) -> float
                   pass

            ``support()`` informs :py:class:`~pyxu.operator.UniformSpread` what the kernel's :math:`[-s, s]` support is.
            Note that kernels must have symmetric support, but the kernel itself need not be symmetric.

        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.

        kwargs: dict
            Extra kwargs to configure :py:class:`~pyxu.operator.UniformSpread`.
            Supported parameters are:

                * max_cluster_size: int = 10_000
                    Maximum number of support points per sub-grid/cluster.

                * max_window_ratio: float = 10
                    Maximum size of the sub-grids, expressed as multiples of the kernel's support.

                * workers: int = 2 (# virtual cores)
                    Number of threads used to spread sub-grids.
                    Specifying `None` uses all cores.

            Default values are chosen if unspecified.

            Some guidelines to set these parameters:

                * The pair (`max_window_ratio`, `max_cluster_size`) determines the maximum memory requirements per
                  sub-grid.
                * `workers` sub-grids are processed in parallel.
                  Due to the Python GIL, the speedup is not linear with the number of workers.
                  Choosing a small value (ex 2-4) seems to offer the best parallel efficiency.
                * `max_cluster_size` should be chosen large enough for there to be meaningful work done by each thread.
                  If chosen too small, then many sub-grids need to be written to the global grid, which may introduce
                  overheads.
                * `max_window_ratio` should be chosen based on the point distribution. Set it to `inf` if only cluster
                  size matters.
        """
        # Put all internal variables in canonical form ------------------------
        #   x: (M, D) array (NUMPY/DASK)
        #   z: start: (D,)-float,
        #      stop : (D,)-float,
        #      num  : (D,)-int,
        #   kernel: tuple[OpT]
        assert x.ndim in {1, 2}
        if x.ndim == 1:
            x = x[:, np.newaxis]
        M, D = x.shape

        kernel = self._as_seq(kernel, D)
        for k in kernel:
            assert hasattr(k, "support"), "[Kernel] Missing support() info."
            s = k.support()
            assert s > 0, "[Kernel] Support must be non-zero."

        z["start"] = self._as_seq(z["start"], D, float)
        z["stop"] = self._as_seq(z["stop"], D, float)
        z["num"] = self._as_seq(z["num"], D, int)
        msg_lattice = "[z] Degenerate lattice detected."
        for d in range(D):
            alpha, beta, N = z["start"][d], z["stop"][d], z["num"][d]
            assert alpha <= beta, msg_lattice
            if alpha < beta:
                assert N >= 1, msg_lattice
            else:
                # Lattices with overlapping nodes are not allowed.
                assert N == 1, msg_lattice

        kwargs = {
            "max_cluster_size": kwargs.get("max_cluster_size", 10_000),
            "max_window_ratio": kwargs.get("max_window_ratio", 10),
            "workers": kwargs.get("workers", 2),
        }
        assert kwargs["max_cluster_size"] > 0
        assert kwargs["max_window_ratio"] >= 3

        # Object Initialization -----------------------------------------------
        super().__init__(
            dim_shape=M,
            codim_shape=z["num"],
        )
        self._x = pxrt.coerce(x)
        self._z = z
        self._kernel = kernel
        self._enable_warnings = bool(enable_warnings)
        self._kwargs = kwargs

        # Acceleration metadata -----------------------------------------------
        ndi = pxd.NDArrayInfo.from_obj(self._x)
        if ndi == pxd.NDArrayInfo.DASK:
            # Built at runtime, so just validate chunk structure of `x`.
            assert self._x.chunks[1] == (D,), "[x] Chunking along last dimension unsupported."
        elif ndi == pxd.NDArrayInfo.CUPY:
            raise NotImplementedError
        else:  # NUMPY
            # Compile low-level spread/interp kernels.
            code = self._gen_code(dim_rank=D, dtype=self._x.dtype)
            exec(code, locals())
            self._nb_spread = eval("f_spread")
            self._nb_interpolate = eval("f_interpolate")

            self._cl_info = self._build_cl_info(
                x=self._x,
                z=self._z,
                kernel=self._kernel,
                **self._kwargs,
            )

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M) input weights :math:`\mathbf{w} \in \mathbb{R}^{M}`.

        Returns
        -------
        out: NDArray
            (..., N1,...,ND) lattice values :math:`\mathbf{v} \in \mathbb{R}^{N_{1} \times\cdots\times N_{D}}`.
        """
        arr = self._cast_warn(arr)
        ndi = pxd.NDArrayInfo.from_obj(arr)
        xp = ndi.module()

        sh = arr.shape[: -self.dim_rank]
        if ndi == pxd.NDArrayInfo.DASK:
            # High-level idea:
            # 1. split the lattice into non-overlapping sub-regions.
            # 2. foreach (x/w, sub-lattice) pair: spread (x/w,) onto the sub-lattice.
            # 3. collapse all support points contributing to the same sub-lattice.
            #
            # Concretely, we rely on DASK.blockwise() to achieve this.
            #
            # For each sub-problem to compute the right outputs, it must know onto which sub-lattice to spread.
            # As such we need to encode the sub-lattice limits as an array and give them to blockwise(). This is encoded
            # in `z_spec` below.
            #
            # Reminder of array shape/block structures that blockwise() will use:
            # [legend] array: shape, blocks/dim, dimension index {see blockwise().}]
            # * x: (M, D), (Bx, 1), (0, 1)
            # * w: (..., M), (Bw1,...,BwT, Bx), (-T,...,-1, 0)
            # * z_spec: (Bz1,...,BzD, D, 2), (Bz1,...,BzD, 1, 1), (2,...,D+3)
            # * parts: [ this is the output of blockwise() ]
            #       (        ...,  N1,..., ND, Bx),
            #       (Bw1,...,BwT, Bz1,...,BzD, Bx), -> we 'sumed' over the single-block axes (1, D+2, D+3)
            #       ( -T,..., -1,   2,...,D+1,  0)
            # * out [ = parts.sum(axis=-1) ]
            #       (        ...,  N1,..., ND),
            #       (Bw1,...,BwT, Bz1,...,BzD)

            assert (
                arr.chunks[-1] == self._x.chunks[0]
            ), "Support weights `w` must have same chunk-structure as support points `x`."

            # Compute `z` info, letting dask decide how large lattice chunks should be.
            N_stack = len(sh)
            z_chunks = xp.core.normalize_chunks(
                chunks=arr.chunks[:N_stack] + ("auto",) * self.codim_rank,
                shape=(*sh, *self.codim_shape),
                dtype=arr.dtype,
            )[-self.codim_rank :]
            z_bcount = [len(chks) for chks in z_chunks]
            z_bounds = [np.r_[0, chks].cumsum() for chks in z_chunks]
            z_spec = np.zeros((*z_bcount, self.codim_rank, 2), dtype=int)
            for *c_idx, d, i in np.ndindex(*z_spec.shape):
                z_spec[*c_idx, d, i] = z_bounds[d][c_idx[d] + i]
            z_spec = xp.asarray(z_spec, chunks=(1,) * self.codim_rank + (self.codim_rank, 2))

            # Compute (x,w,z,o)_ind & output chunks
            x_ind = (0, 1)
            w_ind = tuple(range(-N_stack, 1))
            z_ind = tuple(range(2, self.codim_rank + 4))
            o_ind = (*range(-N_stack, 0), *range(2, self.codim_rank + 2), 0)
            o_chunks = {0: 1}
            for d, ax in enumerate(range(2, self.codim_rank + 2)):
                o_chunks[ax] = z_chunks[d]

            parts = xp.blockwise(
                # shape:  (...,        |  N1,..., ND | Bx)
                # bcount: (Bw1,...,BwT | Bz1,...,BzD | Bx)
                *(self._blockwise_spread, o_ind),
                *(self._x, x_ind),
                *(arr, w_ind),
                *(z_spec, z_ind),
                dtype=arr.dtype,
                adjust_chunks=o_chunks,
                align_arrays=False,
                concatenate=True,
                meta=self._x._meta,
            )
            out = parts.sum(axis=-1)  # (..., N1,...,ND)
        else:  # NUMPY
            # Re-order (x, w)
            x_idx = self._cl_info["x_idx"]
            x = self._x[x_idx]
            w = arr[..., x_idx]

            # Spread each cluster onto its own sub-grid
            Q = len(self._cl_info["cl_bound"]) - 1
            with cf.ThreadPoolExecutor(max_workers=self._kwargs["workers"]) as executor:
                fs = [executor.submit(self._spread, x=x, w=w, q=q) for q in range(Q)]

            # Update global grid
            out = xp.zeros((*sh, *self.codim_shape), dtype=arr.dtype)
            for f in cf.as_completed(fs):
                q, v = f.result()

                z_anchor = self._cl_info["z_anchor"][q]
                z_num = self._cl_info["z_num"][q]
                roi = [slice(n0, n0 + num) for (n0, num) in zip(z_anchor, z_num)]

                out[..., *roi] += v
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., N1,...,ND) lattice values :math:`\mathbf{v} \in \mathbb{R}^{N_{1} \times\cdots\times N_{D}}`.

        Returns
        -------
        out: NDArray
            (..., M) non-uniform weights :math:`\mathbf{w} \in \mathbb{R}^{M}`.
        """
        arr = self._cast_warn(arr)
        ndi = pxd.NDArrayInfo.from_obj(arr)
        xp = ndi.module()

        sh = arr.shape[: -self.codim_rank]
        if ndi == pxd.NDArrayInfo.DASK:
            # High-level idea:
            # 1. foreach (x, v) pair: interpolate (v,) onto (x,).
            # 2. collapse all sub-lattices contributing to the same support points.
            #
            # Concretely, we rely on DASK.blockwise() to achieve this.
            #
            # For each sub-problem to compute the right outputs, it must know from which sub-lattice to interpolate.
            # As such we need to encode the sub-lattice limits as an array and give them to blockwise(). This is encoded
            # in `z_spec` below.
            #
            # Reminder of array shape/block structures that blockwise() will use:
            # [legend] array: shape, blocks/dim, dimension index {see blockwise().}]
            # * x: (M, D), (Bx, 1), (0, 1)
            # * v: (..., N1,...,ND), (Bv1,...,BvT, Bz1,...BzD), (-T,...,-1, 2,...,D+1)
            # * z_spec: (Bz1,...,BzD, D, 2), (Bz1,...,BzD, 1, 1), (2,...,D+3)
            # * parts: [ this is the output of blockwise() ]
            #       (        ...,   M, Bz1,...,BzD),
            #       (Bv1,...,BvT,  Bx, Bz1,...,BzD), -> we 'sumed' over the single-block axes (1, D+2, D+3)
            #       ( -T,..., -1,   0,   2,...,D+1)
            # * out [ = parts.sum(axis=(-D,...,-1)) ]
            #       (        ...,   M),
            #       (Bv1,...,BvT,  Bx)

            # Compute `z` info from `v`
            N_stack = len(sh)
            z_chunks = arr.chunks[-self.codim_rank :]
            z_bcount = [len(chks) for chks in z_chunks]
            z_bounds = [np.r_[0, chks].cumsum() for chks in z_chunks]
            z_spec = np.zeros((*z_bcount, self.codim_rank, 2), dtype=int)
            for *c_idx, d, i in np.ndindex(*z_spec.shape):
                z_spec[*c_idx, d, i] = z_bounds[d][c_idx[d] + i]
            z_spec = xp.asarray(z_spec, chunks=(1,) * self.codim_rank + (self.codim_rank, 2))

            # Compute (x,v,z,o)_ind & output chunks
            x_ind = (0, 1)
            v_ind = (*range(-N_stack, 0), *range(2, self.codim_rank + 2))
            z_ind = tuple(range(2, self.codim_rank + 4))
            o_ind = (*range(-N_stack, 1), *range(2, self.codim_rank + 2))
            o_chunks = {ax: 1 for ax in range(2, self.codim_rank + 2)}

            parts = xp.blockwise(
                # shape:  (...,        | M  | Bz1,...,BzD)
                # bcount: (Bv1,...,BvT | Bx | Bz1,...,BzD)
                *(self._blockwise_interpolate, o_ind),
                *(self._x, x_ind),
                *(arr, v_ind),
                *(z_spec, z_ind),
                dtype=arr.dtype,
                adjust_chunks=o_chunks,
                align_arrays=False,
                concatenate=True,
                meta=self._x._meta,
            )
            out = parts.sum(axis=tuple(range(-self.codim_rank, 0)))  # (..., M)
        else:  # NUMPY
            # Re-order (x,)
            x_idx = self._cl_info["x_idx"]
            x = self._x[x_idx]

            # Interpolate each sub-grid onto support points within.
            Q = len(self._cl_info["cl_bound"]) - 1
            with cf.ThreadPoolExecutor(max_workers=self._kwargs["workers"]) as executor:
                fs = [executor.submit(self._interpolate, x=x, v=arr, q=q) for q in range(Q)]

            # Update global support
            out = xp.zeros((*sh, self.dim_size), dtype=arr.dtype)
            for f in cf.as_completed(fs):
                q, w = f.result()

                a = self._cl_info["cl_bound"][q]
                b = self._cl_info["cl_bound"][q + 1]
                idx = x_idx[a:b]

                out[..., idx] = w
        return out

    def asarray(self, **kwargs) -> pxt.NDArray:
        # Perform computation in `x`-backend/precision ... --------------------
        xp = pxu.get_array_module(self._x)
        dtype = self._x.dtype

        lattice = self._lattice(xp, dtype, flatten=False)

        A = xp.ones((*self.codim_shape, *self.dim_shape), dtype=dtype)  # (N1,...,ND, M)
        for d in range(self.codim_rank):
            _l = lattice[d]  # (1,...,1,Nd,1,...,1)
            _x = self._x[:, d]  # (M,)
            _phi = self._kernel[d]
            _A = _phi(_l[..., np.newaxis] - _x)  # (1,...,1,Nd,1,...,1, M)
            A *= _A

        # ... then abide by user's backend/precision choice. ------------------
        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)
        B = xp.array(pxu.to_NUMPY(A), dtype=dtype)
        return B

    # Helper routines (internal) ----------------------------------------------
    def _cast_warn(self, arr: pxt.NDArray) -> pxt.NDArray:
        if arr.dtype == self._x.dtype:
            out = arr
        else:
            if self._enable_warnings:
                msg = "Computation may not be performed at the requested precision."
                warnings.warn(msg, pxw.PrecisionWarning)
            out = arr.astype(dtype=self._x.dtype)
        return out

    @staticmethod
    def _as_seq(x, N, _type=None) -> tuple:
        if isinstance(x, cabc.Iterable):
            _x = tuple(x)
        else:
            _x = (x,)
        if len(_x) == 1:
            _x *= N  # broadcast
        assert len(_x) == N

        if _type is None:
            return _x
        else:
            return tuple(map(_type, _x))

    @staticmethod
    def _build_cl_info(
        x: np.ndarray,
        z: dict[str, tuple],
        kernel: tuple[pxt.OpT],
        **kwargs,
    ) -> dict[int, dict]:
        # Build acceleration metadata.
        #
        # * Partitions the support points into Q clusters.
        # * Identifies the sub-grids onto which each cluster is spread.
        #
        # Parameters
        # ----------
        # x: np.ndarray[float]
        #     (M, D) support points.
        # z: dict[str, tuple]
        #     Lattice (start, stop, num) specifier.
        # kernel: tuple[OpT]
        #     (D,) axial kernels.
        # kwargs: dict
        #     Spreadder config info.
        #
        # Returns
        # -------
        # info: dict[str, np.ndarray]
        #     (Q,) cluster metadata, with fields:
        #
        #     * x_idx: np.ndarray[int64]
        #         (M,) indices to re-order `x` s.t. points in each cluster are sequential.
        #         Its length may be smaller than M if points do not contribute to the lattice in any way.
        #     * cl_bound: np.ndarray[int64]
        #         (Q+1,) indices into `x_idx` indicating where the q-th cluster's support points start/end.
        #         Cluster `q` contains support points `x[x_idx][cl_bound[q] : cl_bound[q+1]]`.
        #     * z_anchor: np.ndarray[int64]
        #         (Q, D) lower-left coordinate of each sub-grid w.r.t. the global grid.
        #     * z_num: np.ndarray[int64]
        #         (Q, D) sub-grid sizes.

        # Get kernel/lattice parameters.
        s = np.array([k.support() for k in kernel])
        alpha = np.array(z["start"])
        beta = np.array(z["stop"])
        N = np.array(z["num"])

        # Restrict clustering to support points which contribute to the lattice.
        active = np.all(alpha - s <= x, axis=1) & np.all(x <= beta + s, axis=1)  # (M,)
        active2global = np.flatnonzero(active)
        x = x[active]

        # Quick exit if no support points.
        if len(x) == 0:
            factory = lambda: np.array([], dtype=np.int64)
            return collections.defaultdict(factory)

        # Group support points into clusters to match max window size.
        max_window_ratio = kwargs.get("max_window_ratio")
        bbox_dim = (2 * s) * max_window_ratio
        clusters = pxm_cl.grid_cluster(x, bbox_dim)

        # Recursively split clusters to match max cluster size.
        N_max = kwargs.get("max_cluster_size")
        clusters = pxm_cl.bisect_cluster(x, clusters, N_max)

        # Gather metadata per cluster
        M, D = x.shape
        Q = len(clusters)
        info = dict(
            x_idx=np.zeros(M, dtype=np.int64),
            cl_bound=np.full(Q + 1, fill_value=M, dtype=np.int64),
            z_anchor=np.zeros((Q, D), dtype=np.int64),
            z_num=np.zeros((Q, D), dtype=np.int64),
        )
        i = 0
        for q, x_idx in enumerate(clusters.values()):
            # 1) Compute off-grid lattice boundaries after spreading.
            _x = x[x_idx]  # (Mq,)
            LL = _x.min(axis=0) - s  # lower-left lattice coordinate
            UR = _x.max(axis=0) + s  # upper-right lattice coordinate

            # 2) Get gridded equivalents.
            #
            # Note: using `ratio` safely handles the problematic (alpha==beta) case.
            ratio = N - 1.0
            ratio[N > 1] /= (beta - alpha)[N > 1]
            LL_idx = np.floor((LL - alpha) * ratio)
            UR_idx = np.ceil((UR - alpha) * ratio)

            # 3) Clip LL/UR to lattice boundaries.
            LL_idx = np.fmax(0, LL_idx).astype(int)
            UR_idx = np.fmin(UR_idx, N - 1).astype(int)

            # 4) Store metadata
            Mq = len(_x)
            info["x_idx"][i : i + Mq] = active2global[x_idx]  # indices w.r.t input `x`
            info["cl_bound"][q] = i
            info["z_anchor"][q] = LL_idx
            info["z_num"][q] = UR_idx - LL_idx + 1
            i += Mq

        return info

    def _lattice(
        self,
        xp: pxt.ArrayModule,
        dtype: pxt.DType,
        roi: tuple[slice] = None,
        flatten: bool = True,
    ) -> tuple[pxt.NDArray]:
        # Create sparse lattice mesh.
        #
        # Parameters
        # ----------
        # xp: ArrayModule
        #     Which array module to use to represent the mesh.
        # dtype: DType
        #     Precision of the arrays.
        # roi: tuple[slice]
        #     If provided, the lattice is restricted to a specific region-of-interest.
        #     The full lattice is returned by default.
        # flatten: bool
        #
        # Returns
        # -------
        # lattice: tuple[NDArray]
        #     * flatten=True : (D,) 1D lattice nodes.
        #     * flatten=False: (D,) sparse ND-meshgrid of lattice nodes.
        if roi is None:
            roi = (slice(None),) * self.codim_rank

        lattice = [None] * self.codim_rank
        for d in range(self.codim_rank):
            alpha = self._z["start"][d]
            beta = self._z["stop"][d]
            N = self._z["num"][d]
            step = 0 if (N == 1) else (beta - alpha) / (N - 1)
            _roi = roi[d]
            lattice[d] = (alpha + xp.arange(N)[_roi] * step).astype(dtype)
        if not flatten:
            lattice = xp.meshgrid(
                *lattice,
                indexing="ij",
                sparse=True,
            )
        return lattice

    def _spread(self, x: np.ndarray, w: np.ndarray, q: int) -> tuple[int, np.ndarray]:
        # Spread (support, weight) pairs onto sub-lattice of specific cluster.
        #
        # Parameters
        # ----------
        # x: NDArray[float]
        #     (M, D) support points. [NUMPY, canonical order]
        # w: NDArray[float]
        #     (..., M) support weights. [NUMPY, canonical order]
        # q: int
        #     Cluster index.
        #
        # Returns
        # -------
        # q: int
        #     Cluster index.
        # v: NDArray[float]
        #     (..., S1,...,SD) q-th spreaded sub-lattice.
        dtype = w.dtype

        # Extract relevant fields from `_cl_info`
        a = self._cl_info["cl_bound"][q]
        b = self._cl_info["cl_bound"][q + 1]
        Mq, D = b - a, x.shape[1]
        z_anchor = self._cl_info["z_anchor"][q]  # (D,)
        z_num = S = self._cl_info["z_num"][q]  # (S1,...,SD)
        roi = [slice(n0, n0 + num) for (n0, num) in zip(z_anchor, z_num)]

        # Sub-sample (x, w)
        x = x[a:b]  # (Mq, D)
        w = w[..., a:b]  # (..., Mq)

        # Evaluate 1D kernel weights per support point
        lattice = self._lattice(np, dtype, roi)  # (S1,),...,(SD,)
        kernel = np.zeros((Mq, D, max(S)), dtype=dtype)  # (Mq, D, S_max)
        for d, kern in enumerate(self._kernel):
            kernel[:, d, : S[d]] = kern(lattice[d] - x[:, [d]])

        # Spread onto sub-lattice
        v = np.zeros((*w.shape[:-1], *S), dtype=dtype)  # (..., S1,...,SD)
        self._nb_spread(
            w=w.reshape(-1, Mq),
            kernel=kernel,
            out=v.reshape(-1, *S),
        )

        return (q, v)

    def _blockwise_spread(self, x: pxt.NDArray, w: pxt.NDArray, z_spec: pxt.NDArray) -> pxt.NDArray:
        # Spread (support, weight) pairs onto sub-lattice.
        #
        # Parameters
        # ----------
        # x: NDArray[float]
        #     (Mq, D) support points. [NUMPY]
        # w: NDArray[float]
        #     (..., Mq) support weights. [NUMPY]
        # z_spec: NDArray[float]
        #     (<D 1s>, D, 2) start/stop lattice bounds per dimension.
        #     This parameter is identical to _lattice()'s `roi` parameter, but in array form.
        #
        # Returns
        # -------
        # v: NDArray[float]
        #     (..., S1,...,SD, 1) sub-lattice weights.
        #
        #     [Note the trailing size-1 dim; this is required since blockwise() expects to
        #      stack these outputs given how it was called.]

        # Get lattice descriptor in suitable form for UniformSpread().
        z_spec = z_spec[(0,) * self.codim_rank]  # (D, 2)
        lattice = self._lattice(
            xp=pxu.get_array_module(x),
            dtype=x.dtype,
            roi=[slice(start, stop) for (start, stop) in z_spec],
        )
        z_spec = dict(
            start=[_l[0] for _l in lattice],
            stop=[_l[-1] for _l in lattice],
            num=[_l.size for _l in lattice],
        )

        op = UniformSpread(
            x=x,
            z=z_spec,
            kernel=self._kernel,
            enable_warnings=self._enable_warnings,
            **self._kwargs,
        )
        v = op.apply(w)  # (..., S1,...,SD)
        return v[..., np.newaxis]

    def _interpolate(self, x: np.ndarray, v: np.ndarray, q: int) -> tuple[int, np.ndarray]:
        # Interpolate (lattice, weight) pairs onto support points within cluster.
        #
        # Parameters
        # ----------
        # x: NDArray[float]
        #     (M, D) support points. [NUMPY, canonical order]
        # v: NDArray[float]
        #     (..., N1,...,ND) lattice weights. [NUMPY]
        # q: int
        #     Cluster index.
        #
        # Returns
        # -------
        # q: int
        #     Cluster index.
        # w: NDArray[float]
        #     (..., Mq) interpolated support weights of q-th cluster.
        dtype = v.dtype

        # Extract relevant fields from `_cl_info`
        a = self._cl_info["cl_bound"][q]
        b = self._cl_info["cl_bound"][q + 1]
        Mq, D = b - a, x.shape[1]
        z_anchor = self._cl_info["z_anchor"][q]  # (D,)
        z_num = S = self._cl_info["z_num"][q]  # (S1,...,SD)
        roi = [slice(n0, n0 + num) for (n0, num) in zip(z_anchor, z_num)]

        # Sub-sample (x, v)
        x = x[a:b]  # (Mq, D)
        v = v[..., *roi]  # (..., S1,...,SD)

        # Evaluate 1D kernel weights per support point
        lattice = self._lattice(np, dtype, roi)  # (S1,),...,(SD,)
        kernel = np.zeros((Mq, D, max(S)), dtype=dtype)  # (Mq, D, S_max)
        for d, kern in enumerate(self._kernel):
            kernel[:, d, : S[d]] = kern(lattice[d] - x[:, [d]])

        # Interpolate onto support points
        w = np.zeros((*v.shape[:-D], Mq), dtype=dtype)  # (..., Mq)
        self._nb_interpolate(
            v=v.reshape(-1, *S),
            kernel=kernel,
            out=w.reshape(-1, Mq),
        )

        return (q, w)

    def _blockwise_interpolate(self, x: pxt.NDArray, v: pxt.NDArray, z_spec: pxt.NDArray) -> pxt.NDArray:
        # Spread (lattice, weight) pairs onto support points.
        #
        # Parameters
        # ----------
        # x: NDArray[float]
        #     (Mq, D) support points. [NUMPY]
        # v: NDArray[float]
        #     (..., S1,...,SD) lattice weights. [NUMPY]
        # z_spec: NDArray[float]
        #     (<D 1s>, D, 2) start/stop lattice bounds per dimension.
        #     This parameter is identical to _lattice()'s `roi` parameter, but in array form.
        #
        # Returns
        # -------
        # w: NDArray[float]
        #     (..., Mq, <D 1s>) support weights.
        #
        #     [Note the trailing size-1 dims; these are required since blockwise() expects to
        #      stack these outputs given how it was called.]

        # Get lattice descriptor in suitable form for UniformSpread().
        z_spec = z_spec[(0,) * self.codim_rank]  # (D, 2)
        lattice = self._lattice(
            xp=pxu.get_array_module(x),
            dtype=x.dtype,
            roi=[slice(start, stop) for (start, stop) in z_spec],
        )
        z_spec = dict(
            start=[_l[0] for _l in lattice],
            stop=[_l[-1] for _l in lattice],
            num=[_l.size for _l in lattice],
        )

        op = UniformSpread(
            x=x,
            z=z_spec,
            kernel=self._kernel,
            enable_warnings=self._enable_warnings,
            **self._kwargs,
        )
        w = op.adjoint(v)  # (..., Mq)

        expand = (np.newaxis,) * self.codim_rank
        return w[..., *expand]

    @staticmethod
    def _gen_code(dim_rank: int, dtype: pxt.DType) -> str:
        # Given the dimension rank D, generate Numba kernel codes used in _[spread,interpolate]():
        #
        # * void f_spread(w, kernel, out)
        # * void f_interpolate(v, kernel, out)
        template = string.Template(
            r"""
import numba as nb
import numpy as np

f_flags = dict(
    nopython=True,
    nogil=True,
    cache=False,  # not applicable to dynamically-defined functions (https://github.com/numba/numba/issues/3501)
    forceobj=False,
    parallel=False,
    error_model="numpy",
    fastmath=True,
    locals={},
    boundscheck=False,
)

@nb.jit(**f_flags)
def find_bounds(x: np.ndarray[float]) -> tuple[int, int]:
    # Parameters:
    #     x: (N,)
    # Returns
    #     a, b: indices s.t. x[a:b] contains the non-zero segment of `x`.
    N = len(x)
    a, a_found = N, False
    b, b_found = N, False
    for i in range(N):
        lhs, rhs = x[i], x[N - 1 - i]
        if (not a_found) and (abs(lhs) > 0):
            a, a_found = i, True
        if (not b_found) and (abs(rhs) > 0):
            b, b_found = N - i, True
        if a_found and b_found:  # early exit
            return a, b
    return a, b

@nb.jit(
    "${signature_spread}",
    **f_flags,
)
def f_spread(
    w: np.ndarray[float],  # (Ns, M)
    kernel: np.ndarray[float],  # (M, D, S_max)
    out: np.ndarray[float],  # (Ns, S1,...,SD)
):
    Ns, M = w.shape
    S = out.shape[1:]
    D = len(S)

    lb = np.zeros(D, dtype=np.int64)
    ub = np.zeros(D, dtype=np.int64)
    for m in range(M):
        for d in range(D):
            lb[d], ub[d] = find_bounds(kernel[m, d, : S[d]])

        support = ${support}
        for offset in np.ndindex(support):
            idx = ${idx}

            # Compute kernel weight
            k = 1
            for d in range(D):
                k *= kernel[m, d, idx[d]]

            # Spread onto lattice
            for ns in range(Ns):
                out[ns, *idx] += k * w[ns, m]

@nb.jit(
    "${signature_interpolate}",
    **f_flags,
)
def f_interpolate(
    v: np.ndarray[float],  # (Ns, S1,...,SD)
    kernel: np.ndarray[float],  # (M, D, S_max)
    out: np.ndarray[float],  # (Ns, M)
):
    Ns, M = out.shape
    S = v.shape[1:]
    D = len(S)

    lb = np.zeros(D, dtype=np.int64)
    ub = np.zeros(D, dtype=np.int64)
    for m in range(M):
        for d in range(D):
            lb[d], ub[d] = find_bounds(kernel[m, d, : S[d]])

        support = ${support}
        for offset in np.ndindex(support):
            idx = ${idx}

            # Compute kernel weight
            k = 1
            for d in range(D):
                k *= kernel[m, d, idx[d]]

            # Spread onto support point
            for ns in range(Ns):
                out[ns, m] += k * v[ns, *idx]

"""
        )

        width = pxrt.Width(dtype)
        _type = {
            pxrt.Width.SINGLE: "float32",
            pxrt.Width.DOUBLE: "float64",
        }[width]
        sig_w_sp = _type + "[:,:]"
        sig_v_sp = _type + "[" + (":," * dim_rank) + "::1]"
        sig_kernel = _type + "[:,:,::1]"
        sig_v_int = _type + "[" + (":," * dim_rank) + ":]"
        sig_w_int = _type + "[:,::1]"
        support = ",".join([f"ub[{d}] - lb[{d}]" for d in range(dim_rank)])
        idx = ",".join([f"lb[{d}] + offset[{d}]" for d in range(dim_rank)])

        code = template.substitute(
            signature_spread=f"void({sig_w_sp},{sig_kernel},{sig_v_sp})",
            signature_interpolate=f"void({sig_v_int},{sig_kernel},{sig_w_int})",
            support="(" + support + ",)",
            idx="(" + idx + ",)",
        )
        return code
