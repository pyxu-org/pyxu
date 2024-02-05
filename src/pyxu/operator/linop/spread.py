import collections
import collections.abc as cabc
import concurrent.futures as cf
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

    * Spread/interpolation are performed efficiently via the algorithm described in [FINUFFT]_, i.e. partition
      :math:`\{\mathbf{x}_{m}\}` into sub-grids, spread onto each sub-grid, then add the results to the global grid.
    * The domain is partitioned using a kd-tree from SciPy. The SciPy implementation is CPU-only however, so building
      the data structure will induce a device -> host -> device transfer.
    * The kd-tree is built at init-time only when `x` is a NUMPY/CUPY array.
      For DASK arrays, the tree is built online on subsets of the data to be spreaded.
    * :py:class:`~pyxu.operator.UniformSpread` instances are **not arraymodule-agnostic**: they will only work with
      NDArrays belonging to the same array module as `x`.
    * :py:class:`~pyxu.operator.UniformSpread` is not **precision-agnostic**: it will only work on NDArrays with the
      same dtype as `x`.  A warning is emitted if inputs must be cast to the support dtype.

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

            Functions should be ufuncs with same semantics as :py:class:`~pyxu.abc.Map`, i.e. have an
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
            Extra kwargs passed to ``UniformSpread._build_info()``.
            Supported parameters are:

                * max_cluster_size: int
                    Maximum number of support points per sub-grid/cluster.

                * max_window_ratio: float
                    Maximum size of the sub-grids, expressed as multiples of the kernel's support.

            Default values are chosen if unspecified.

            Some guidelines to set these parameters:

                * The pair (`max_window_ratio`, `max_cluster_size`) determines the maximum memory requirements per sub-grid.
                * Sub-grids are processed in parallel.
                * `max_cluster_size` should be chosen large enough for there to be meaningful work done by each thread.
                  If chosen too small, then many sub-grids need to be written to the global grid, which may introduce
                  overheads.
        """
        # Put all internal variables in canonical form ------------------------
        #   x: (M, D) array (NUMPY/CUPY/DASK)
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
        else:
            self._cluster_info = self._build_info(
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
            (...,  M) input weights :math:`\mathbf{w} \in \mathbb{R}^{M}`.

        Returns
        -------
        out: NDArray
            (...,  N1,...,ND) lattice values :math:`\mathbf{v} \in \mathbb{R}^{N_{1} \times\cdots\times N_{D}}`.
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
        else:  # NUMPY/CUPY
            # 1) Spread each cluster onto its own sub-grid.
            with cf.ThreadPoolExecutor() as executor:
                func = lambda cl: self._spread(w=arr, cluster_info=cl)
                parts = executor.map(func, self._cluster_info)

            # 2) Add sub-grids to the global grid.
            out = xp.zeros((*sh, *self.codim_shape), dtype=arr.dtype)
            for v, cl in zip(parts, self._cluster_info):
                select = tuple(
                    slice(a, a + n)
                    for (a, n) in zip(
                        cl["z_anchor"],
                        cl["z_num"],
                    )
                )
                out[..., *select] += v
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
            (...,  M) non-uniform weights :math:`\mathbf{w} \in \mathbb{R}^{M}`.
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
        else:  # NUMPY/CUPY
            # 1) Interpolate each sub-grid onto support points within.
            with cf.ThreadPoolExecutor() as executor:
                func = lambda cl: self._interpolate(v=arr, cluster_info=cl)
                parts = executor.map(func, self._cluster_info)

            # 2) Re-order/merge support points.
            out = xp.zeros((*sh, self.dim_size), dtype=arr.dtype)
            for w, cl in zip(parts, self._cluster_info):
                select = cl["x_idx"]
                out[..., select] = w
        return out

    def asarray(self, **kwargs) -> pxt.NDArray:
        # Perform computation in `x`-backend/precision ... --------------------
        xp = pxu.get_array_module(self._x)
        dtype = self._x.dtype

        lattice = self._lattice(xp, dtype)

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
    def _build_info(
        x: pxt.NDArray,
        z: dict,
        kernel: tuple[pxt.OpT],
        **kwargs,
    ) -> tuple[dict]:
        # Build acceleration metadata.
        #
        # * Partitions the support points into Q clusters.
        # * Identifies the sub-grids onto which each cluster is spread.
        #
        # Metadata is built on the CPU. If `x` was a CUPY array, a device->host->device
        # transfer will take place.
        #
        # Parameters
        # ----------
        # x: NDArray
        #     (M, D) support points. [NUMPY/CUPY-only]
        # z: dict
        #     Lattice (start, stop, num) specifier.
        # kernel: tuple[OpT]
        #     (D,) axial kernels.
        # kwargs: dict
        #     Spreadder config info.
        #
        # Returns
        # -------
        # info: tuple[dict]
        #     (Q,) cluster metadata, with fields:
        #
        #     * x_idx: NDArray[int] (NUMPY/CUPY)
        #         (Mq,) indices into `x` which identify support points participating in q-th sub-grid.
        #     * z_anchor: tuple[int]
        #         (D,) lower-left coordinate of the sub-grid w.r.t. global grid.
        #     * z_num: tuple[int]
        #         (D,) sub-grid size in each direction.

        # Process via NUMPY. (scipy.spatial.KDTree limitation.)
        ndi = pxd.NDArrayInfo.from_obj(x)
        x_orig, x = x, pxu.to_NUMPY(x)

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
            return tuple()

        # Group support points into clusters to match max window size.
        max_window_ratio = kwargs.get("max_window_ratio")
        bbox_dim = (2 * s) * max_window_ratio
        clusters = pxm_cl.grid_cluster(x, bbox_dim)

        # Recursively split clusters to match max cluster size limits.
        N_max = kwargs.get("max_cluster_size")
        clusters = pxm_cl.bisect_cluster(x, clusters, N_max)

        # Gather metadata per cluster.
        info = collections.defaultdict(dict)
        for c_idx, x_idx in clusters.items():
            # 1) Compute off-grid lattice boundaries after spreading.
            _x = x[x_idx]
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

            info[c_idx]["x_idx"] = active2global[x_idx]  # indices w.r.t input `x`
            info[c_idx]["z_anchor"] = tuple(LL_idx)
            info[c_idx]["z_num"] = tuple(UR_idx - LL_idx + 1)

        # Transfer to GPU, if required.
        if ndi == pxd.NDArrayInfo.CUPY:
            xp = ndi.module()
            with xp.cuda.Device(x_orig.device):
                for cl in info.values():
                    cl["x_idx"] = xp.asarray(cl["x_idx"])

        return tuple(info.values())

    def _lattice(
        self,
        xp: pxt.ArrayModule,
        dtype: pxt.DType,
        roi: tuple[slice] = None,
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
        #
        # Returns
        # -------
        # lattice: tuple[NDArray]
        #     (D,) sparse meshgrid of lattice nodes.
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
        lattice = xp.meshgrid(
            *lattice,
            indexing="ij",
            sparse=True,
        )
        return lattice

    def _spread(self, w: pxt.NDArray, cluster_info: dict) -> pxt.NDArray:
        # Spread (support, weight) pairs onto sub-lattice of specific cluster.
        #
        # Parameters
        # ----------
        # w: NDArray[float]
        #     (..., M) support weights. [NUMPY/CUPY]
        # cluster_info: dict
        #     Cluster's metadata from _build_info().
        #
        # Returns
        # -------
        # v: NDArray[float]
        #     (..., S1,...,SD) sub-lattice weights.
        xp = pxu.get_array_module(w)
        dtype = w.dtype

        # Build sparse lattice mesh on RoI
        roi = [
            slice(n0, n0 + num)
            for (n0, num) in zip(
                cluster_info["z_anchor"],
                cluster_info["z_num"],
            )
        ]
        lattice = self._lattice(xp, dtype, roi)

        # Sub-sample (x, w)
        x_idx = cluster_info["x_idx"]  # (Mq,)
        x = self._x[x_idx]  # (Mq, D)
        w = w[..., x_idx]  # (..., Mq)

        # Spread onto lattice.
        Mq, S = len(x), cluster_info["z_num"]
        expand = (np.newaxis,) * self.codim_rank
        A = xp.ones((Mq, *S), dtype=dtype)  # (Mq, S1,...,SD)
        for d in range(self.codim_rank):
            _l = lattice[d]  # (1,...,1,Sd,1,...,1)
            _phi = self._kernel[d]
            _A = _phi(_l[np.newaxis] - x[:, d, *expand])  # (Mq, 1,...,1,Sd,1,...,1)
            A *= _A
        v = xp.tensordot(w, A, axes=1)

        return v

    def _blockwise_spread(self, x: pxt.NDArray, w: pxt.NDArray, z_spec: pxt.NDArray) -> pxt.NDArray:
        # Spread (support, weight) pairs onto sub-lattice.
        #
        # Parameters
        # ----------
        # x: NDArray[float]
        #     (Mq, D) support points. [NUMPY/CUPY]
        # w: NDArray[float]
        #     (..., Mq) support weights. [NUMPY/CUPY]
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
            start=[_l.ravel()[0] for _l in lattice],
            stop=[_l.ravel()[-1] for _l in lattice],
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

    def _interpolate(self, v: pxt.NDArray, cluster_info: dict) -> pxt.NDArray:
        # Interpolate (lattice, weight) pairs onto support points within cluster.
        #
        # Parameters
        # ----------
        # v: NDArray[float]
        #     (..., N1,...,ND) lattice weights. [NUMPY/CUPY]
        # cluster_info: dict
        #     Cluster's metadata from _build_info().
        #
        # Returns
        # -------
        # w: NDArray[float]
        #     (..., Mq) cluster support weights.
        xp = pxu.get_array_module(v)
        dtype = v.dtype

        # Build sparse lattice mesh on RoI
        roi = [
            slice(n0, n0 + num)
            for (n0, num) in zip(
                cluster_info["z_anchor"],
                cluster_info["z_num"],
            )
        ]
        lattice = self._lattice(xp, dtype, roi)

        # Sub-sample (x, v)
        x_idx = cluster_info["x_idx"]  # (Mq,)
        x = self._x[x_idx]  # (Mq, D)
        v = v[..., *roi]  # (..., S1,...,SD)

        # Interpolate onto support points.
        Mq, S = len(x), v.shape[-self.codim_rank :]
        A = xp.ones((*S, Mq), dtype=dtype)  # (S1,...,SD, Mq)
        for d in range(self.codim_rank):
            _l = lattice[d]  # (1,...,1,Sd,1,...,1)
            _phi = self._kernel[d]
            _A = _phi(_l[..., np.newaxis] - x[:, d])  # (1,...,1,Sd,1,...,1, Mq)
            A *= _A
        w = xp.tensordot(v, A, axes=self.codim_rank)

        return w

    def _blockwise_interpolate(self, x: pxt.NDArray, v: pxt.NDArray, z_spec: pxt.NDArray) -> pxt.NDArray:
        # Spread (lattice, weight) pairs onto support points.
        #
        # Parameters
        # ----------
        # x: NDArray[float]
        #     (Mq, D) support points. [NUMPY/CUPY]
        # v: NDArray[float]
        #     (..., S1,...,SD) lattice weights. [NUMPY/CUPY]
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
            start=[_l.ravel()[0] for _l in lattice],
            stop=[_l.ravel()[-1] for _l in lattice],
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
