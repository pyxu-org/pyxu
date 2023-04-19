import collections.abc as cabc
import functools
import operator
import typing as typ
import warnings

import numpy as np

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycuw
from pycsou.operator.linop.pad import Pad
from pycsou.operator.linop.select import Trim
from pycsou.operator.linop.stencil._stencil import _Stencil

__all__ = [
    "Stencil",
]


class Stencil(pyca.SquareOp):
    r"""
    Multi-dimensional JIT-compiled stencil.

    Stencils are a common computational pattern in which array elements are updated according to some fixed
    pattern called the stencil kernel. Notable examples include multi-dimensional convolutions, correlations and finite
    differences (see Notes for a definition).

    Stencils can be evaluated efficiently on CPU/GPU architectures.

    Several boundary conditions are supported.
    Moreover boundary conditions may differ per axis.

    **Implementation Notes**

    * Numba (an its ``@stencil``'s decorator) is used behind the scenes to compile efficient machine code from a stencil kernel
      specification.
      This has 2 consequences:

      * :py:class:`~pycsou.operator.linop.stencil.stencil.Stencil` instances are **not
        arraymodule-agnostic**: they will only work with NDArrays belonging to the same array module
        as ``kernel``.
      * Compiled stencils are not **precision-agnostic**: they will only work on NDArrays with the
        same dtype as ``kernel``.
        A warning is emitted if inputs must be cast to the kernel dtype.

    * Stencil kernels can be specified in two forms:
      (See :py:meth:`~pycsou.operator.linop.stencil.stencil.Stencil.__init__` for details.)

      * A single non-separable :math:`D`-dimensional kernel :math:`k[i_{1},\ldots,i_{D}]` of shape
        :math:`(k_{1},\ldots,k_{D})`.
      * A sequence of separable :math:`1`-dimensional kernel(s) :math:`k_{d}[i]` of shapes
        :math:`(k_{1},),\ldots,(k_{D},)` such that :math:`k[i_{1},\ldots,i_{D}] = \Pi_{d=1}^{D}
        k_{d}[i_{d}]`.

    Notes
    -----
    Given a :math:`D`-dimensional array :math:`x\in\mathbb{R}^{N_1\times\cdots\times N_D}` and kernel
    :math:`k\in\mathbb{R}^{k_1\times\cdots\times k_D}` with center :math:`(c_1, \ldots, c_D)`, the ouput of the stencil operator
    is an array :math:`y\in\mathbb{R}^{N_1\times\cdots\times N_D}` given by:

    .. math::

               y[i_{1},\ldots,i_{D}]
               =
               \sum_{q_{1},\ldots,q_{D}=0}^{k_{1},\ldots,k_{D}}
               x[i_{1} - c_{1} + q_{1},\ldots,i_{D} - c_{D} + q_{D}]
               \,\cdot\,
               k[q_{1},\ldots,q_{D}]

    This corresponds to a *correlation* with a shifted version of the kernel :math:`k`.

    Summation terms involving out-of-bounds indices of :math:`x` are handled by Numba via zero-padding. We offer support for
    additional padding types as follows: pad the input array with one of the supported mode, perform the zero-padded stencil operation
    on the padded array, and discard the padded region with trimming.

    Any stencil operator :math:`S` instantiated with this class can hence be written as the composition :math:`S = TS_0P`, where :math:`T, S_0, P` are the trimming, stencil with zero-padding conditions and
    padding operators respectively. This construct allows us to handle complex boundary conditions under which :math:`S` *may not be a proper
    stencil* (i.e., varying kernel) but can still be implemented efficiently via a proper stencil upon appropriate trimming/padding.

    For example consider the decomposition of the following (improper) stencil operator:

    >>> S = Stencil(arg_shape=(5,), kernel=np.r_[1,2,-3], center=(2,), mode="reflect")
    [[-3.  2.  1.  0.  0.]
     [ 2. -2.  0.  0.  0.]
     [ 1.  2. -3.  0.  0.]
     [ 0.  1.  2. -3.  0.]
     [ 0.  0.  1.  2. -3.]] # Improper stencil (kernel varies across rows)
     =
     [[0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0.]] #Trimming
     @
     [[-3.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 2. -3.  0.  0.  0.  0.  0.  0.  0.]
     [ 1.  2. -3.  0.  0.  0.  0.  0.  0.]
     [ 0.  1.  2. -3.  0.  0.  0.  0.  0.]
     [ 0.  0.  1.  2. -3.  0.  0.  0.  0.]
     [ 0.  0.  0.  1.  2. -3.  0.  0.  0.]
     [ 0.  0.  0.  0.  1.  2. -3.  0.  0.]
     [ 0.  0.  0.  0.  0.  1.  2. -3.  0.]
     [ 0.  0.  0.  0.  0.  0.  1.  2. -3.]] #Proper stencil (Toeplitz structure)
     @
     [[0. 0. 1. 0. 0.]
     [0. 1. 0. 0. 0.]
     [1. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0.]
     [0. 0. 1. 0. 0.]
     [0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1.]
     [0. 0. 0. 1. 0.]
     [0. 0. 1. 0. 0.]] #Padding with reflect mode.

    Note that the adjoint of a stencil operator may not necessarily be a stencil operator, or the associated center and boundary
    conditions may be hard to predict.  For example, the adjoint of the stencil operator defined above is given by:

    >>> S.T.asarray()
    [[-3.,  2.,  1.,  0.,  0.],
    [ 2., -2.,  2.,  1.,  0.],
    [ 1.,  0., -3.,  2.,  1.],
    [ 0.,  0.,  0., -3.,  2.],
    [ 0.,  0.,  0.,  0., -3.]]

    which resembles as stencil with time-reversed kernel, but with weird (if not improper) boundary conditions. This can also
    be seen from the fact that :math:`S^\ast = P^\ast S_0^\ast T^\ast = P^\ast S_0^\ast P_0,` and :math:`P^\ast` is in general
    not a trimming operator (see :py:class:`~pycsou.operator.linop.pad.Pad`).

    Same holds with the gram/cogram operators. Consider indeed the following order 1 backward finite-difference operator with zero-padding:

    >>> S = Stencil(arg_shape=(5,), kernel=np.r_[-1, 1], center=(0,), mode='constant')
    >>> S.gram().asarray()
    [[ 1. -1.  0.  0.  0.]
     [-1.  2. -1.  0.  0.]
     [ 0. -1.  2. -1.  0.]
     [ 0.  0. -1.  2. -1.]
     [ 0.  0.  0. -1.  2.]]

    We observe that the Gram differs from the order 2 centered finite-difference operator (reduced-order derivative on one side).


    Example
    -------

    * **Moving average of a 1D signal**

      Let :math:`x[n]` denote a 1D signal.
      The weighted moving average

      .. math::

         y[n] = x[n-2] + 2 x[n-1] + 3 x[n]

      can be viewed as the output of the 3-point stencil of kernel :math:`k = [1, 2, 3]`.

      .. code-block:: python3

         import numpy as np
         from pycsou.operator.linop import Stencil

         x = np.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

         op = Stencil(
             arg_shape=x.shape,
             kernel=np.array([1, 2, 3]),
             center=(2,),  # k[2] applies on x[n]
         )

         y = op.apply(x)  # [0, 3, 8, 14, 20, 26, 32, 38, 44, 50]


    * **Non-seperable image filtering**

      Let :math:`x[n, m]` denote a 2D image.
      The blurred image

      .. math::

         y[n, m] = 2 x[n-1,m-1] + 3 x[n-1,m+1] + 4 x[n+1,m-1] + 5 x[n+1,m+1]

      can be viewed as the output of the 9-point stencil

      .. math::

         k =
         \left[
         \begin{array}{ccc}
            2 & 0 & 3 \\
            0 & 0 & 0 \\
            4 & 0 & 5
         \end{array}
         \right].

      .. code-block:: python3

         import numpy as np
         from pycsou.operator.linop import Stencil

         x = np.arange(64).reshape(8, 8)  # square image
         # [[ 0,  1,  2,  3,  4,  5,  6,  7]
         #  [ 8,  9, 10, 11, 12, 13, 14, 15]
         #  [16, 17, 18, 19, 20, 21, 22, 23]
         #  [24, 25, 26, 27, 28, 29, 30, 31]
         #  [32, 33, 34, 35, 36, 37, 38, 39]
         #  [40, 41, 42, 43, 44, 45, 46, 47]
         #  [48, 49, 50, 51, 52, 53, 54, 55]
         #  [56, 57, 58, 59, 60, 61, 62, 63]]

         op = Stencil(
             arg_shape=x.shape,
             kernel=np.array(
                 [[2, 0, 3],
                  [0, 0, 0],
                  [4, 0, 5]]),
             center=(1, 1),  # k[1, 1] applies on x[n, m]
         )

         y = op.apply(x.reshape(-1)).reshape(8, 8)
         # [[ 45   82   91  100  109  118  127   56 ]
         #  [ 88  160  174  188  202  216  230  100 ]
         #  [152  272  286  300  314  328  342  148 ]
         #  [216  384  398  412  426  440  454  196 ]
         #  [280  496  510  524  538  552  566  244 ]
         #  [344  608  622  636  650  664  678  292 ]
         #  [408  720  734  748  762  776  790  340 ]
         #  [147  246  251  256  261  266  271  108 ]]

    * **Seperable image filtering**

      Let :math:`x[n, m]` denote a 2D image.
      The warped image

      .. math::

         \begin{align*}
             y[n, m] = & +  4 x[n-1,m-1] +  5 x[n-1,m] +  6 x[n-1,m+1] \\
                       & +  8 x[n  ,m-1] + 10 x[n  ,m] + 12 x[n  ,m+1] \\
                       & + 12 x[n+1,m-1] + 15 x[n+1,m] + 18 x[n+1,m+1]
         \end{align*}

      can be viewed as the output of the 9-point stencil

      .. math::

         k_{2D} =
         \left[
         \begin{array}{ccc}
             4 &  5 &  6 \\
             8 & 10 & 12 \\
            12 & 15 & 18 \\
         \end{array}
         \right].

      Notice however that :math:`y[n, m]` can be implemented more efficiently
      by factoring the 9-point stencil as a cascade of two 3-point stencils:

      .. math::

         k_{2D}
         = k_{1} k_{2}^{T}
         = \left[
         \begin{array}{c}
            1 \\ 2 \\ 3
         \end{array}
         \right]
         \left[
         \begin{array}{c}
            4 & 5 & 6
         \end{array}
         \right].

      Seperable stencils are supported and should be preferred when applicable.

      .. code-block:: python3

         import numpy as np
         from pycsou.operator.linop import Stencil

         x = np.arange(64).reshape(8, 8)  # square image
         # [[ 0,  1,  2,  3,  4,  5,  6,  7]
         #  [ 8,  9, 10, 11, 12, 13, 14, 15]
         #  [16, 17, 18, 19, 20, 21, 22, 23]
         #  [24, 25, 26, 27, 28, 29, 30, 31]
         #  [32, 33, 34, 35, 36, 37, 38, 39]
         #  [40, 41, 42, 43, 44, 45, 46, 47]
         #  [48, 49, 50, 51, 52, 53, 54, 55]
         #  [56, 57, 58, 59, 60, 61, 62, 63]]

         op_2D = Stencil(  # using non-seperable kernel
             arg_shape=x.shape,
             kernel=np.array(
                 [[ 4,  5,  6],
                  [ 8, 10, 12],
                  [12, 15, 18]]),
             center=(1, 1),  # k[1, 1] applies on x[n, m]
         )
         op_sep = Stencil(  # using seperable kernels
             arg_shape=x.shape,
             kernel=[
                 np.array([1, 2, 3]),  # k1: stencil along 1st axis
                 np.array([4, 5, 6]),  # k2: stencil along 2nd axis
             ],
             center=(1, 1),  # k1[1] * k2[1] applies on x[n, m]
         )

         y_2D = op_2D.apply(x.reshape(-1)).reshape(8, 8)
         y_sep = op_sep.apply(x.reshape(-1)).reshape(8, 8)  # np.allclose(y_2D, y_sep) -> True
         # [[ 294   445   520   595   670   745   820   511 ]
         #  [ 740  1062  1152  1242  1332  1422  1512   930 ]
         #  [1268  1782  1872  1962  2052  2142  2232  1362 ]
         #  [1796  2502  2592  2682  2772  2862  2952  1794 ]
         #  [2324  3222  3312  3402  3492  3582  3672  2226 ]
         #  [2852  3942  4032  4122  4212  4302  4392  2658 ]
         #  [3380  4662  4752  4842  4932  5022  5112  3090 ]
         #  [1778  2451  2496  2541  2586  2631  2676  1617 ]]
    """

    KernelSpec = typ.Union[
        pyct.NDArray,  # (k1, ..., kD) non-seperable kernel
        cabc.Sequence[pyct.NDArray],  # [(k1,), ..., (kD,)] seperable kernels
    ]

    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        kernel: KernelSpec,
        center: _Stencil.IndexSpec,
        mode: Pad.ModeSpec = "constant",
        enable_warnings: bool = True,
    ):
        r"""
        Parameters
        ----------
        arg_shape: pyct.NDArrayShape
            Shape of the rank-:math:`D` input array.
        kernel: KernelSpec
            Stencil coefficients.
            Two forms are accepted:

            * NDArray of rank-:math:`D`: denotes a non-seperable stencil.
            * tuple[NDArray_1, ..., NDArray_D]: a sequence of 1D stencils such that dimension[k]
              is filtered by stencil `kernel[k]`, that is:

              .. math::
                 k = k_1\otimes \cdots\otimes k_D,

              or in Python: ``k = reduce(numpy.outer, kernel)``.

        center: IndexSpec
            (i_1, ..., i_D) index of the stencil's center.

            `center` defines how a kernel is overlaid on inputs to produce outputs.

        mode: str | list(str)
            Boundary conditions.
            Multiple forms are accepted:

            * str: unique mode shared amongst dimensions.
              Must be one of:

              * 'constant' (zero-padding)
              * 'wrap'
              * 'reflect'
              * 'symmetric'
              * 'edge'
            * tuple[str, ...]: dimension[k] uses `mode[k]` as boundary condition.

            (See :py:func:`numpy.pad` for details.)
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.
        """
        arg_shape, _kernel, _center, _mode = self._canonical_repr(arg_shape, kernel, center, mode)
        codim = dim = np.prod(arg_shape)
        super().__init__(shape=(codim, dim))

        # Pad/Trim operators
        pad_width = self._compute_pad_width(_kernel, _center, _mode)
        self._pad = Pad(
            arg_shape=arg_shape,
            pad_width=pad_width,
            mode=_mode,
        )
        self._trim = Trim(
            arg_shape=self._pad._pad_shape,
            trim_width=pad_width,
        )

        # Stencil operators used in .apply()
        self._st_fw = [None] * len(_kernel)
        for i, (k_fw, c_fw) in enumerate(zip(_kernel, _center)):
            self._st_fw[i] = _Stencil.init(kernel=k_fw, center=c_fw)

        # Stencil operators used in .adjoint()
        self._st_bw = [None] * len(_kernel)
        _kernel, _center = self._bw_equivalent(_kernel, _center)
        for i, (k_bw, c_bw) in enumerate(zip(_kernel, _center)):
            self._st_bw[i] = _Stencil.init(kernel=k_bw, center=c_bw)

        self._dispatch_params = dict()  # Extra kwargs passed to _Stencil.apply()
        self._dtype = _kernel[0].dtype  # useful constant
        self._enable_warnings = bool(enable_warnings)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        x = self._pad.apply(arr)
        x = x.reshape(-1, *self._pad._pad_shape)

        y = self._stencil_chain(self._cast_warn(x), self._st_fw)
        y = y.reshape(*arr.shape[:-1], -1)

        out = self._trim.apply(y)
        return out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        x = self._trim.adjoint(arr)
        x = x.reshape(-1, *self._pad._pad_shape)

        y = self._stencil_chain(self._cast_warn(x), self._st_bw)
        y = y.reshape(*arr.shape[:-1], -1)

        out = self._pad.adjoint(y)
        return out

    def configure_dispatcher(self, **kwargs):
        """
        (Only applies if `kernel` is a CuPy array.)

        Configure stencil Dispatcher.

        See :py:meth:`~pycsou.operator.linop.stencil._stencil._Stencil.apply` for accepted options.

        Example
        -------
        .. code-block:: python3

           import cupy as cp
           from pycsou.operator.linop import Stencil

           x = cp.arange(10)

           op = Stencil(
               arg_shape=x.shape,
               kernel=np.array([1, 2, 3]),
               center=(1,),
           )

           y = op.apply(x)  # uses default threadsperblock/blockspergrid values

           op.configure_dispatcher(
               threadsperblock=50,
               blockspergrid=3,
           )
           y = op.apply(x)  # supplied values used instead
        """
        for k, v in kwargs.items():
            self._dispatch_params.update(k=v)

    @staticmethod
    def _canonical_repr(arg_shape, kernel, center, mode):
        # Create canonical representations
        #   * `arg_shape`: tuple[int]
        #   * `_kernel`: list[ndarray[float], ...]
        #   * `_center`: list[ndarray[int], ...]
        #   * `_mode`: list[str, ...]
        if not isinstance(arg_shape, cabc.Sequence):
            arg_shape = (arg_shape,)
        arg_shape = tuple(arg_shape)

        N = len(arg_shape)
        assert len(center) == N

        kernel = pycu.compute(kernel, traverse=True)
        try:
            # array input -> non-seperable filter
            pycu.get_array_module(kernel)
            assert kernel.ndim == N

            _kernel = [pycrt.coerce(kernel)]
            _center = [np.array(center, dtype=int)]
        except:
            # sequence input -> seperable filter(s)
            assert len(kernel) == N  # one filter per dimension

            _kernel = [None] * N
            for i in range(N):
                sh = [1] * N
                sh[i] = -1
                _kernel[i] = pycrt.coerce(kernel[i]).reshape(sh)

            _center = np.zeros((N, N), dtype=int)
            _center[np.diag_indices(N)] = center

        _mode = Pad(  # get `mode` in canonical form
            (2,) * _kernel[0].ndim,
            pad_width=1,
            mode=mode,
        )._mode

        return arg_shape, _kernel, _center, _mode

    @staticmethod
    def _compute_pad_width(_kernel, _center, _mode) -> Pad.WidthSpec:
        N = _kernel[0].ndim
        pad_width = [None] * N
        for i in range(N):
            if len(_kernel) == 1:  # non-seperable filter
                c = _center[0][i]
                n = _kernel[0].shape[i]
            else:  # seperable filter(s)
                c = _center[i][i]
                n = _kernel[i].size

            # 1. Pad/Trim operators are shared amongst [apply,adjoint]():
            #    lhs/rhs are thus padded equally.
            # 2. When mode != "constant", pad width must match kernel dimensions to retain border
            #    effects.
            if _mode[i] == "constant":
                p = max(c, n - c - 1)
            else:  # anything else supported by Pad()
                p = n - 1
            pad_width[i] = (p, p)
        return tuple(pad_width)

    @staticmethod
    def _bw_equivalent(_kernel, _center):
        # Transform FW kernel/center specification to BW variant.
        k_bw = [np.flip(k_fw) for k_fw in _kernel]

        if len(_kernel) == 1:  # non-seperable filter
            c_bw = [(_kernel[0].shape - _center[0]) - 1]
        else:  # seperable filter(s)
            N = _kernel[0].ndim
            c_bw = np.zeros((N, N), dtype=int)
            for i in range(N):
                c_bw[i, i] = _kernel[i].shape[i] - _center[i][i] - 1

        return k_bw, c_bw

    def _stencil_chain_DASK(self, x: pyct.NDArray, stencils: list) -> pyct.NDArray:
        # Apply sequence of stencils to `x`.
        #
        # x: (N_stack, N_1, ..., N_D)
        # y: (N_stack, N_1, ..., N_D)
        #
        # [2022.12.28, Sepand]
        #   For some unknown reason, .map_overlap() gives incorrect results when inputs to
        #   .map_overlap() contain stacking dimensions.
        #   Workaround: call .map_overlap() on each stack-dim seperately, then re-assemble.

        # _compute_pad_width(): LHS/RHS padded equally, so choose either one
        depth = [lhs for (lhs, rhs) in self._pad._pad_width]
        y = [
            _.map_overlap(
                self._stencil_chain,
                depth=depth,
                boundary=0,
                trim=True,
                dtype=x.dtype,
                meta=x._meta,
                stencils=stencils,
            )
            for _ in x
        ]

        xp = pycu.get_array_module(x)
        y = xp.stack(y, axis=0)
        return y

    @pycu.redirect("x", DASK=_stencil_chain_DASK)
    def _stencil_chain(self, x: pyct.NDArray, stencils: list) -> pyct.NDArray:
        # Apply sequence of stencils to `x`.
        # (For Dask inputs, see _stencil_chain_DASK().)
        #
        # x: (N_stack, N_1, ..., N_D)
        # y: (N_stack, N_1, ..., N_D)
        if len(stencils) == 1:
            x, y = x, x.copy()
        else:
            # [2023.04.17, Sepand]
            # In-place updates of `x` breaks thread-safety of Stencil().
            # This is problematic if Stencil() is used with DASK inputs.
            x, y = x.copy(), x.copy()

        for st in stencils:
            st.apply(x, y, **self._dispatch_params)
            x, y = y, x
        y = x
        return y

    def _cast_warn(self, arr: pyct.NDArray) -> pyct.NDArray:
        if arr.dtype == self._dtype:
            out = arr
        else:
            if self._enable_warnings:
                msg = "Computation may not be performed at the requested precision."
                warnings.warn(msg, pycuw.PrecisionWarning)
            out = arr.astype(dtype=self._dtype)
        return out

    @pycrt.enforce_precision()
    def lipschitz(self, **kwargs) -> pyct.Real:
        if kwargs.get("tight", False):
            self._lipschitz = super().lipschitz(**kwargs)
        else:
            # Analytic upper bound from Young's convolution inequality:
            #     \norm{x \ast h}{2} \le \norm{x}{2}\norm{h}{1}
            #
            # -> L \le \norm{h}{1}
            kernels = [st._kernel for st in self._st_fw]
            kernel = functools.reduce(operator.mul, kernels, 1)
            L_st = np.linalg.norm(pycu.to_NUMPY(kernel).reshape(-1), ord=1)

            L_pad = self._pad.lipschitz()
            L_trim = self._trim.lipschitz()

            L_ub = L_trim * L_st * L_pad
            self._lipschitz = min(L_ub, self._lipschitz)
        return self._lipschitz

    def to_sciop(self, **kwargs):
        # Stencil.apply/adjoint() prefer precision provided at init-time.
        kwargs.update(dtype=self._dtype)
        op = pyca.SquareOp.to_sciop(self, **kwargs)
        return op

    def asarray(self, **kwargs) -> pyct.NDArray:
        # Stencil.apply() prefers precision provided at init-time.
        xp = pycu.get_array_module(self._st_fw[0]._kernel)
        _A = super().asarray(xp=xp, dtype=self._dtype)

        xp = kwargs.get("xp", pycd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pycrt.getPrecision().value)
        A = xp.array(pycu.to_NUMPY(_A), dtype=dtype)
        return A

    @pycrt.enforce_precision()
    def trace(self, **kwargs) -> pyct.Real:
        if all(m == "constant" for m in self._pad._mode):
            # tr = (kernel center coefficient) * N
            tr = functools.reduce(
                operator.mul,
                [st._kernel[tuple(st._center)] for st in self._st_fw],
                1,
            )
            tr *= self.dim
        else:
            # Standard algorithm, with computations restricted to precision supported by
            # Stencil.apply().
            kwargs.update(dtype=self._dtype)
            tr = super().trace(**kwargs)
        return float(tr)

    @property
    def kernel(self) -> KernelSpec:
        r"""
        Stencil kernel coefficients.

        Returns
        -------
        kern: :py:attr:`~pycsou.operator.linop.stencil.stencil.Stencil.KernelSpec`
            Stencil coefficients.

            If the kernel is non-seperable, a single array is returned.
            Otherwise :math:`D` arrays are returned, one per axis.
        """
        if len(self._st_fw) == 1:
            kern = self._st_fw[0]._kernel
        else:
            kern = [st._kernel for st in self._st_fw]
        return kern

    @property
    def center(self) -> _Stencil.IndexSpec:
        """
        Stencil central position.

        Returns
        -------
        ctr: :py:attr:`~pycsou.operator.linop.stencil._stencil._Stencil.IndexSpec`
            Stencil central position.
        """
        if len(self._st_fw) == 1:
            ctr = self._st_fw[0]._center
        else:
            ctr = [st._center[d] for (d, st) in enumerate(self._st_fw)]
        return tuple(ctr)

    @property
    def relative_indices(self) -> typ.Sequence[pyct.NDArray]:
        r"""
        Relative indices of the stencil.

        Returns
        -------
        Sequence[NDArray]
            Relative indices of the stencil's kernel in each dimension.

        Examples
        --------

        >>> S = Stencil(arg_shape=(5,6,9), kernel=[np.r_[1, -1], np.r_[3, 2, 1], np.r_[2,-1, 3 ,1]], center=(1, 0, 3))
        >>> S.relative_indices
        [array([-1,  0]), array([0, 1, 2]), array([-3, -2, -1,  0])]

        """
        if len(self._st_fw) == 1:
            return [np.arange(s) - c for c, s in zip(self.center, self.kernel.shape)]
        else:
            return [np.arange(k.size) - c for c, k in zip(self.center, self.kernel)]

    def print_kernel(self):
        r"""
        Print the :math:`D`-dimensional stencil's kernel.

        The stencil's center is identified by surrounding parentheses.

        Examples
        --------

        >>> S = Stencil(arg_shape=(5,6,), kernel=[np.r_[3, 2, 1], np.r_[2,-1, 3 ,1]], center=(1, 2))
        >>> S.print_kernel()
        [[6.0 -3.0 9.0 3.0]
         [4.0 -2.0 (6.0) 2.0]
         [2.0 -1.0 3.0 1.0]]

        """
        if len(self._st_fw) == 1:
            kernel_np = pycu.to_NUMPY(self.kernel)
        else:
            for i, k in enumerate(self.kernel):
                if i == 0:
                    kernel = k
                else:
                    kernel = kernel * k
            kernel_np = pycu.to_NUMPY(kernel)
        kernel_np = kernel_np.astype(str)
        kernel_np[self.center] = "(" + kernel_np[self.center] + ")"
        print(np.array2string(kernel_np).replace("'", ""))
