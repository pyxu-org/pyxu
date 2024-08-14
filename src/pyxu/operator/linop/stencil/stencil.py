import collections.abc as cabc
import functools
import operator
import typing as typ
import warnings

import numpy as np

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.runtime as pxrt
import pyxu.util as pxu
from pyxu.operator.linop.pad import Pad
from pyxu.operator.linop.select import Trim
from pyxu.operator.linop.stencil._stencil import _Stencil

__all__ = [
    "Stencil",
    "Correlate",
    "Convolve",
]


class Stencil(pxa.SquareOp):
    r"""
    Multi-dimensional JIT-compiled stencil.

    Stencils are a common computational pattern in which array elements are updated according to some fixed pattern
    called the *stencil kernel*.  Notable examples include multi-dimensional convolutions, correlations and finite
    differences.  (See Notes for a definition.)

    Stencils can be evaluated efficiently on CPU/GPU architectures.

    Several boundary conditions are supported.  Moreover boundary conditions may differ per axis.

    .. rubric:: Implementation Notes

    * Numba (and its ``@stencil`` decorator) is used to compile efficient machine code from a stencil kernel
      specification.  This has 2 consequences:

      * :py:class:`~pyxu.operator.Stencil` instances are **not arraymodule-agnostic**: they will only work with NDArrays
         belonging to the same array module as `kernel`.
      * Compiled stencils are not **precision-agnostic**: they will only work on NDArrays with the same dtype as
        `kernel`.  A warning is emitted if inputs must be cast to the kernel dtype.

    * Stencil kernels can be specified in two forms: (See :py:meth:`~pyxu.operator.Stencil.__init__` for details.)

      * A single non-separable :math:`D`-dimensional kernel :math:`k[i_{1},\ldots,i_{D}]` of shape
        :math:`(K_{1},\ldots,K_{D})`.
      * A sequence of separable :math:`1`-dimensional kernel(s) :math:`k_{d}[i]` of shapes
        :math:`(K_{1},),\ldots,(K_{D},)` such that :math:`k[i_{1},\ldots,i_{D}] = \Pi_{d=1}^{D} k_{d}[i_{d}]`.

    .. rubric:: Mathematical Notes

    Given a :math:`D`-dimensional array :math:`x\in\mathbb{R}^{N_1\times\cdots\times N_D}` and kernel
    :math:`k\in\mathbb{R}^{K_1\times\cdots\times K_D}` with center :math:`(c_1, \ldots, c_D)`, the output of the stencil
    operator is an array :math:`y\in\mathbb{R}^{N_1\times\cdots\times N_D}` given by:

    .. math::

       y[i_{1},\ldots,i_{D}]
       =
       \sum_{q_{1},\ldots,q_{D}=0}^{K_{1},\ldots,K_{D}}
       x[i_{1} - c_{1} + q_{1},\ldots,i_{D} - c_{D} + q_{D}]
       \,\cdot\,
       k[q_{1},\ldots,q_{D}].

    This corresponds to a *correlation* with a shifted version of the kernel :math:`k`.

    Numba stencils assume summation terms involving out-of-bound indices of :math:`x` are set to zero.
    :py:class:`~pyxu.operator.Stencil` lifts this constraint by extending the stencil to boundary values via pre-padding
    and post-trimming.  Concretely, any stencil operator :math:`S` instantiated with :py:class:`~pyxu.operator.Stencil`
    can be written as the composition :math:`S = TS_0P`, where :math:`(T, S_0, P)` are trimming, stencil with
    zero-padding conditions, and padding operators respectively.  This construct allows
    :py:class:`~pyxu.operator.Stencil` to handle complex boundary conditions under which :math:`S` *may not be a proper
    stencil* (i.e., varying kernel) but can still be implemented efficiently via a proper stencil upon appropriate
    trimming/padding.

    For example consider the decomposition of the following (improper) stencil operator:

    .. code-block:: python3

       >>> S = Stencil(
       ...      dim_shape=(5,),
       ...      kernel=np.r_[1, 2, -3],
       ...      center=(2,),
       ...      mode="reflect",
       ... )

       >>> S.asarray()
       [[-3   2   1   0   0]
        [ 2  -2   0   0   0]
        [ 1   2  -3   0   0]
        [ 0   1   2  -3   0]
        [ 0   0   1   2  -3]] # Improper stencil (kernel varies across rows)
       =
       [[0  0  1  0  0  0  0  0  0]
        [0  0  0  1  0  0  0  0  0]
        [0  0  0  0  1  0  0  0  0]
        [0  0  0  0  0  1  0  0  0]
        [0  0  0  0  0  0  1  0  0]]  # Trimming
       @
       [[-3   0   0   0   0   0   0   0   0]
        [ 2  -3   0   0   0   0   0   0   0]
        [ 1   2  -3   0   0   0   0   0   0]
        [ 0   1   2  -3   0   0   0   0   0]
        [ 0   0   1   2  -3   0   0   0   0]
        [ 0   0   0   1   2  -3   0   0   0]
        [ 0   0   0   0   1   2  -3   0   0]
        [ 0   0   0   0   0   1   2  -3   0]
        [ 0   0   0   0   0   0   1   2  -3]]  # Proper stencil (Toeplitz structure)
       @
       [[0  0  1  0  0]
        [0  1  0  0  0]
        [1  0  0  0  0]
        [0  1  0  0  0]
        [0  0  1  0  0]
        [0  0  0  1  0]
        [0  0  0  0  1]
        [0  0  0  1  0]
        [0  0  1  0  0]]  # Padding with reflect mode.

    Note that the adjoint of a stencil operator may not necessarily be a stencil operator, or the associated center and
    boundary conditions may be hard to predict.  For example, the adjoint of the stencil operator defined above is given
    by:

    .. code-block:: python3

       >>> S.T.asarray()
       [[-3   2   1   0   0],
        [ 2  -2   2   1   0],
        [ 1   0  -3   2   1],
        [ 0   0   0  -3   2],
        [ 0   0   0   0  -3]]

    which resembles a stencil with time-reversed kernel, but with weird (if not improper) boundary conditions.  This can
    also be seen from the fact that :math:`S^\ast = P^\ast S_0^\ast T^\ast = P^\ast S_0^\ast P_0,` and :math:`P^\ast` is
    in general not a trimming operator.  (See :py:class:`~pyxu.operator.Pad`.)

    The same holds for gram/cogram operators.  Consider indeed the following order-1 backward finite-difference operator
    with zero-padding:

    .. code-block:: python3

       >>> S = Stencil(
       ...      dim_shape=(5,),
       ...      kernel=np.r_[-1, 1],
       ...      center=(0,),
       ...      mode='constant',
       ... )

       >>> S.gram().asarray()
       [[ 1  -1   0   0   0]
        [-1   2  -1   0   0]
        [ 0  -1   2  -1   0]
        [ 0   0  -1   2  -1]
        [ 0   0   0  -1   2]]

    We observe that the Gram differs from the order 2 centered finite-difference operator.  (Reduced-order derivative on
    one side.)

    Example
    -------

    * **Moving average of a 1D signal**

      Let :math:`x[n]` denote a 1D signal.  The weighted moving average

      .. math::

         y[n] = x[n-2] + 2 x[n-1] + 3 x[n]

      can be viewed as the output of the 3-point stencil of kernel :math:`k = [1, 2, 3]`.

      .. code-block:: python3

         import numpy as np
         from pyxu.operator import Stencil

         x = np.arange(10)  # [0  1  2  3  4  5  6  7  8  9]

         op = Stencil(
             dim_shape=x.shape,
             kernel=np.array([1, 2, 3]),
             center=(2,),  # k[2] applies on x[n]
         )

         y = op.apply(x)  # [0  3  8  14  20  26  32  38  44  50]


    * **Non-seperable image filtering**

      Let :math:`x[n, m]` denote a 2D image.  The blurred image

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
         from pyxu.operator import Stencil

         x = np.arange(64).reshape(8, 8)  # square image
         # [[ 0   1   2   3   4   5   6   7]
         #  [ 8   9  10  11  12  13  14  15]
         #  [16  17  18  19  20  21  22  23]
         #  [24  25  26  27  28  29  30  31]
         #  [32  33  34  35  36  37  38  39]
         #  [40  41  42  43  44  45  46  47]
         #  [48  49  50  51  52  53  54  55]
         #  [56  57  58  59  60  61  62  63]]

         op = Stencil(
             dim_shape=x.shape,
             kernel=np.array(
                 [[2, 0, 3],
                  [0, 0, 0],
                  [4, 0, 5]]),
             center=(1, 1),  # k[1, 1] applies on x[n, m]
         )

         y = op.apply(x)
         # [[ 45   82   91  100  109  118  127   56 ]
         #  [ 88  160  174  188  202  216  230  100 ]
         #  [152  272  286  300  314  328  342  148 ]
         #  [216  384  398  412  426  440  454  196 ]
         #  [280  496  510  524  538  552  566  244 ]
         #  [344  608  622  636  650  664  678  292 ]
         #  [408  720  734  748  762  776  790  340 ]
         #  [147  246  251  256  261  266  271  108 ]]

    * **Seperable image filtering**

      Let :math:`x[n, m]` denote a 2D image.  The warped image

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

      Notice however that :math:`y[n, m]` can be implemented more efficiently by factoring the 9-point stencil as a
      cascade of two 3-point stencils:

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
         from pyxu.operator import Stencil

         x = np.arange(64).reshape(8, 8)  # square image
         # [[ 0   1   2   3   4   5   6   7]
         #  [ 8   9  10  11  12  13  14  15]
         #  [16  17  18  19  20  21  22  23]
         #  [24  25  26  27  28  29  30  31]
         #  [32  33  34  35  36  37  38  39]
         #  [40  41  42  43  44  45  46  47]
         #  [48  49  50  51  52  53  54  55]
         #  [56  57  58  59  60  61  62  63]]

         op_2D = Stencil(  # using non-seperable kernel
             dim_shape=x.shape,
             kernel=np.array(
                 [[ 4,  5,  6],
                  [ 8, 10, 12],
                  [12, 15, 18]]),
             center=(1, 1),  # k[1, 1] applies on x[n, m]
         )
         op_sep = Stencil(  # using seperable kernels
             dim_shape=x.shape,
             kernel=[
                 np.array([1, 2, 3]),  # k1: stencil along 1st axis
                 np.array([4, 5, 6]),  # k2: stencil along 2nd axis
             ],
             center=(1, 1),  # k1[1] * k2[1] applies on x[n, m]
         )

         y_2D = op_2D.apply(x)
         y_sep = op_sep.apply(x)  # np.allclose(y_2D, y_sep) -> True
         # [[ 294   445   520   595   670   745   820   511 ]
         #  [ 740  1062  1152  1242  1332  1422  1512   930 ]
         #  [1268  1782  1872  1962  2052  2142  2232  1362 ]
         #  [1796  2502  2592  2682  2772  2862  2952  1794 ]
         #  [2324  3222  3312  3402  3492  3582  3672  2226 ]
         #  [2852  3942  4032  4122  4212  4302  4392  2658 ]
         #  [3380  4662  4752  4842  4932  5022  5112  3090 ]
         #  [1778  2451  2496  2541  2586  2631  2676  1617 ]]

    .. Warning::

        For large, non-separable kernels, stencil compilation can be time-consuming. Depending on your computer's
        architecture, using the :py:class:~pyxu.operator.FFTCorrelate operator might offer a more efficient solution.
        However, the performance improvement varies, so we recommend evaluating this alternative in your specific
        environment.

    See Also
    --------
    :py:class:`~pyxu.operator.Convolve`,
    :py:class:`~pyxu.operator._Stencil`,
    :py:class:`~pyxu.operator.FFTCorrelate`,
    :py:class:`~pyxu.operator.FFTConvolve`
    """

    KernelSpec = typ.Union[
        pxt.NDArray,  # (k1,...,kD) non-seperable kernel
        cabc.Sequence[pxt.NDArray],  # [(k1,), ..., (kD,)] seperable kernels
    ]

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        kernel: KernelSpec,
        center: _Stencil.IndexSpec,
        mode: Pad.ModeSpec = "constant",
        enable_warnings: bool = True,
    ):
        r"""
        Parameters
        ----------
        dim_shape: NDArrayShape
            (M1,...,MD) input dimensions.
        kernel: ~pyxu.operator.Stencil.KernelSpec
            Stencil coefficients.  Two forms are accepted:

            * NDArray of rank-:math:`D`: denotes a non-seperable stencil.
            * tuple[NDArray_1, ..., NDArray_D]: a sequence of 1D stencils such that dimension[k] is filtered by stencil
              `kernel[k]`, that is:

              .. math::

                 k = k_1 \otimes\cdots\otimes k_D,

              or in Python: ``k = functools.reduce(numpy.multiply.outer, kernel)``.

        center: ~pyxu.operator._Stencil.IndexSpec
            (i1,...,iD) index of the stencil's center.

            `center` defines how a kernel is overlaid on inputs to produce outputs.

        mode: str, :py:class:`list` ( str )
            Boundary conditions.  Multiple forms are accepted:

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
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        _kernel, _center, _mode = self._canonical_repr(self.dim_shape, kernel, center, mode)

        # Pad/Trim operators
        pad_width = self._compute_pad_width(_kernel, _center, _mode)
        self._pad = Pad(
            dim_shape=dim_shape,
            pad_width=pad_width,
            mode=_mode,
        )
        self._trim = Trim(
            dim_shape=self._pad.codim_shape,
            trim_width=pad_width,
        )

        # Kernels (These _Stencil() instances are not used as-is in apply/adjoint calls: their ._[kernel,center]
        # attributes are used directly there instead to bypass Numba serialization limits. These _Stencil() objects are
        # used however for all other Operator public methods.)
        # It is moreover advantageous to instantiate them once here to cache JIT-compile kernels upfront.
        self._st_fw = self._init_fw(_kernel, _center)
        self._st_bw = self._init_bw(_kernel, _center)

        self._dispatch_params = dict()  # Extra kwargs passed to _Stencil.apply()
        self._dtype = _kernel[0].dtype  # useful constant
        self._enable_warnings = bool(enable_warnings)

        # We know a crude Lipschitz bound by default. Since computing it takes (code) space,
        # the estimate is computed as a special case of estimate_lipschitz()
        self.lipschitz = self.estimate_lipschitz(__rule=True)

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        x = self._pad.apply(arr)
        y = self._stencil_chain(
            x=self._cast_warn(x),
            stencils=self._st_fw,
        )
        z = self._trim.apply(y)
        return z

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        x = self._trim.adjoint(arr)
        y = self._stencil_chain(
            x=self._cast_warn(x),
            stencils=self._st_bw,
        )
        z = self._pad.adjoint(y)
        return z

    def configure_dispatcher(self, **kwargs):
        """
        (Only applies if `kernel` is a CuPy array.)

        Configure stencil Dispatcher.

        See :py:meth:`~pyxu.operator._Stencil.apply` for accepted options.

        Example
        -------
        .. code-block:: python3

           import cupy as cp
           from pyxu.operator import Stencil

           x = cp.arange(10)

           op = Stencil(
               dim_shape=x.shape,
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

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            # Analytic upper bound from Young's convolution inequality:
            #     \norm{x \ast h}{2} \le \norm{x}{2}\norm{h}{1}
            #
            # -> L \le \norm{h}{1}
            kernels = [st._kernel for st in self._st_fw]
            kernel = functools.reduce(operator.mul, kernels, 1)
            L_st = np.linalg.norm(pxu.to_NUMPY(kernel).reshape(-1), ord=1)

            L_pad = self._pad.lipschitz
            L_trim = self._trim.lipschitz

            L = L_trim * L_st * L_pad  # upper bound
        else:
            L = super().estimate_lipschitz(**kwargs)
        return L

    def asarray(self, **kwargs) -> pxt.NDArray:
        # Stencil.apply() prefers precision provided at init-time.
        xp = pxu.get_array_module(self._st_fw[0]._kernel)
        _A = super().asarray(xp=xp, dtype=self._dtype)

        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pxrt.Width.DOUBLE.value)
        A = xp.array(pxu.to_NUMPY(_A), dtype=dtype)
        return A

    def trace(self, **kwargs) -> pxt.Real:
        if all(m == "constant" for m in self._pad._mode):
            # tr = (kernel center coefficient) * dim_size
            tr = functools.reduce(
                operator.mul,
                [st._kernel[tuple(st._center)] for st in self._st_fw],
                1,
            )
            tr *= self.dim_size
        else:
            # Standard algorithm, with computations restricted to precision supported by
            # Stencil.apply().
            kwargs.update(dtype=self._dtype)
            tr = super().trace(**kwargs)
        return float(tr)

    # Helper routines (public) ------------------------------------------------
    @property
    def kernel(self) -> KernelSpec:
        r"""
        Stencil kernel coefficients.

        Returns
        -------
        kern: ~pyxu.operator.Stencil.KernelSpec
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
        ctr: ~pyxu.operator._Stencil.IndexSpec
            Stencil central position.
        """
        if len(self._st_fw) == 1:
            ctr = self._st_fw[0]._center
        else:
            ctr = [st._center[d] for (d, st) in enumerate(self._st_fw)]
        return tuple(ctr)

    def visualize(self) -> str:
        r"""
        Show the :math:`D`-dimensional stencil kernel.

        The stencil's center is identified by surrounding parentheses.

        Example
        -------
        .. code-block:: python3

           S = Stencil(
               dim_shape=(5, 6),
               kernel=[
                   np.r_[3, 2, 1],
                   np.r_[2, -1, 3, 1],
               ],
               center=(1, 2),
           )
           print(S.visualize())  # [[6.0 -3.0  9.0  3.0]
                                 #  [4.0 -2.0 (6.0) 2.0]
                                 #  [2.0 -1.0  3.0 1.0]]
        """
        kernels = [st._kernel for st in self._st_fw]
        kernel = functools.reduce(operator.mul, kernels, 1)

        kernel = pxu.to_NUMPY(kernel).astype(str)
        kernel[self.center] = "(" + kernel[self.center] + ")"

        kern = np.array2string(kernel).replace("'", "")
        return kern

    # Helper routines (internal) ----------------------------------------------
    @staticmethod
    def _canonical_repr(dim_shape, kernel, center, mode):
        # Create canonical representations
        #   * `_kernel`: list[ndarray[float], ...]
        #   * `_center`: list[ndarray[int], ...]
        #   * `_mode`: list[str, ...]
        #
        # `dim_shape`` is already assumed in tuple-form.
        N = len(dim_shape)
        assert len(center) == N

        kernel = pxu.compute(kernel, traverse=True)
        try:
            # array input -> non-seperable filter
            pxu.get_array_module(kernel)
            assert kernel.ndim == N
            _kernel = [kernel]
            _center = [np.array(center, dtype=int)]
        except Exception:
            # sequence input -> seperable filter(s)
            assert len(kernel) == N  # one filter per dimension

            _kernel = [None] * N
            for i in range(N):
                sh = [1] * N
                sh[i] = -1
                _kernel[i] = kernel[i].reshape(sh)

            _center = np.zeros((N, N), dtype=int)
            _center[np.diag_indices(N)] = center

        _mode = Pad(  # get `mode` in canonical form
            (3,) * _kernel[0].ndim,
            pad_width=1,
            mode=mode,
        )._mode

        return _kernel, _center, _mode

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
    def _init_fw(_kernel, _center) -> list:
        # Initialize kernels used in apply().
        # The returned objects must have the following fields:
        # * _kernel: ndarray[float] (D,)
        # * _center: ndarray[int] (D,)
        _st_fw = [None] * len(_kernel)
        for i, (k_fw, c_fw) in enumerate(zip(_kernel, _center)):
            _st_fw[i] = _Stencil.init(kernel=k_fw, center=c_fw)
        return _st_fw

    @staticmethod
    def _init_bw(_kernel, _center) -> list:
        # Initialize kernels used in adjoint().
        # The returned objects must have the following fields:
        # * _kernel: ndarray[float] (D,)
        # * _center: ndarray[int] (D,)
        _st_bw = [None] * len(_kernel)
        _kernel, _center = Stencil._bw_equivalent(_kernel, _center)
        for i, (k_bw, c_bw) in enumerate(zip(_kernel, _center)):
            _st_bw[i] = _Stencil.init(kernel=k_bw, center=c_bw)
        return _st_bw

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

    def _stencil_chain(self, x: pxt.NDArray, stencils: list) -> pxt.NDArray:
        # Apply sequence of stencils to `x`.
        #
        # x: (..., M1,...,MD)
        # y: (..., M1,...,MD)

        # _Stencil() instances cannot be serialized by Dask, so we pass around _[kernel,center] directly.
        # _Stencil(kernel,center) was compiled in __init__() though, hence re-instantiating _Stencil() here is free.
        kernel = [st._kernel for st in stencils]
        center = [st._center for st in stencils]

        def _chain(x, kernel, center, dispatch_params):
            stencils = [_Stencil.init(k, c) for (k, c) in zip(kernel, center)]

            xp = pxu.get_array_module(x)
            if len(stencils) == 1:
                x = xp.require(x, requirements="C")
                y = x.copy()
            else:
                # [2023.04.17, Sepand]
                # In-place updates of `x` breaks thread-safety of Stencil().
                x, y = x.copy(), x.copy()

            for st in stencils:
                st.apply(x, y, **dispatch_params)
                x, y = y, x
            y = x
            return y

        ndi = pxd.NDArrayInfo.from_obj(x)
        if ndi == pxd.NDArrayInfo.DASK:
            stack_depth = (0,) * (x.ndim - self.dim_rank)
            y = x.map_overlap(
                _chain,
                depth=stack_depth + self._pad._pad_width,
                dtype=x.dtype,
                meta=x._meta,
                # extra _chain() kwargs -------------------
                kernel=kernel,
                center=center,
                dispatch_params=self._dispatch_params,
            )
        else:  # NUMPY/CUPY
            y = _chain(x, kernel, center, self._dispatch_params)
        return y

    def _cast_warn(self, arr: pxt.NDArray) -> pxt.NDArray:
        if arr.dtype == self._dtype:
            out = arr
        else:
            if self._enable_warnings:
                msg = "Computation may not be performed at the requested precision."
                warnings.warn(msg, pxw.PrecisionWarning)
            out = arr.astype(dtype=self._dtype)
        return out


Correlate = Stencil  #: Alias of :py:class:`~pyxu.operator.Stencil`.


class Convolve(Stencil):
    r"""
    Multi-dimensional JIT-compiled convolution.

    Inputs are convolved with the given kernel.

    Notes
    -----
    Given a :math:`D`-dimensional array :math:`x\in\mathbb{R}^{N_1 \times\cdots\times N_D}` and kernel
    :math:`k\in\mathbb{R}^{K_1 \times\cdots\times K_D}` with center :math:`(c_1, \ldots, c_D)`, the output of the
    convolution operator is an array :math:`y\in\mathbb{R}^{N_1 \times\cdots\times N_D}` given by:

    .. math::

       y[i_{1},\ldots,i_{D}]
       =
       \sum_{q_{1},\ldots,q_{D}=0}^{K_{1},\ldots,K_{D}}
       x[i_{1} - q_{1} + c_{1},\ldots,i_{D} - q_{D} + c_{D}]
       \,\cdot\,
       k[q_{1},\ldots,q_{D}].

    The convolution is implemented via :py:class:`~pyxu.operator.Stencil`.  To do so, the convolution kernel is
    transformed to the equivalent correlation kernel:

    .. math::

       y[i_{1},\ldots,i_{D}]
       =
       \sum_{q_{1},\ldots,q_{D}=0}^{K_{1},\ldots,K_{D}}
       &x[i_{1} + q_{1} - (K_{1} - c_{1}),\ldots,i_{D} + q_{D} - (K_{D} - c_{D})] \\
       &\cdot\,
       k[K_{1}-q_{1},\ldots,K_{D}-q_{D}].

    This corresponds to a correlation with a flipped kernel and center.

    .. Warning::

       For large, non-separable kernels, stencil compilation can be time-consuming. Depending on your computer's
       architecture, using the :py:class:~pyxu.operator.FFTConvolve operator might offer a more efficient solution.
       However, the performance improvement varies, so we recommend evaluating this alternative in your specific
       environment.

    Examples
    --------
    .. code-block:: python3

       import numpy as np
       from scipy.ndimage import convolve
       from pyxu.operator import Convolve

       x = np.array([
            [1, 2, 0, 0],
            [5, 3, 0, 4],
            [0, 0, 0, 7],
            [9, 3, 0, 0],
       ])
       k = np.array([
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
       ])
       op = Convolve(
           dim_shape=x.shape,
           kernel=k,
           center=(1, 1),
           mode="constant",
       )

       y_op = op.apply(x)
       y_sp = convolve(x, k, mode="constant", origin=0)  # np.allclose(y_op, y_sp) -> True
       # [[11  10   7   4],
       #  [10   3  11  11],
       #  [15  12  14   7],
       #  [12   3   7   0]]

    See Also
    --------
    :py:class:`~pyxu.operator.Stencil`
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        kernel: Stencil.KernelSpec,
        center: _Stencil.IndexSpec,
        mode: Pad.ModeSpec = "constant",
        enable_warnings: bool = True,
    ):
        r"""
        See :py:meth:`~pyxu.operator.Stencil.__init__` for a description of the arguments.
        """
        super().__init__(
            dim_shape=dim_shape,
            kernel=kernel,
            center=center,
            mode=mode,
            enable_warnings=enable_warnings,
        )

        # flip FW/BW kernels (& centers)
        self._st_fw, self._st_bw = self._st_bw, self._st_fw
