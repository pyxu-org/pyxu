import collections.abc as cabc
import functools
import operator
import typing as typ

import numpy as np

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
from pycsou.operator.linop.pad import Pad
from pycsou.operator.linop.select import Trim
from pycsou.operator.linop.stencil._stencil import _Stencil

__all__ = [
    "Stencil",
]


class Stencil(pyca.SquareOp):
    r"""
    Multi-dimensional JIT-compiled stencil.

    Stencils are a class of linear operators where output :math:`(i_{1},\ldots,i_{d})` is given by a
    weighted linear combination of inputs surrounding :math:`(i_{1},\ldots,i_{d})`.
    Notable examples include multi-dimensional convolution and correlation.

    Stencils can be evaluated efficiently on CPU/GPU architectures.

    Several boundary conditions are supported.
    Moreover boundary conditions may differ per axis.

    Implementation Notes
    --------------------
    * Numba is used behind the scenes to compile efficient machine code from a stencil kernel
      specification.
      This has 2 consequences:

      * :py:class:`~pycsou.operator.linop.stencil.stencil.Stencil` instances are **not
        arraymodule-agnostic**: they will only work with NDArrays belonging to the same array module
        as ``kernel``.
      * Compiled stencils are not **precision-agnostic**: they will only work on NDArrays with the
        same dtype as ``kernel``.
    * Stencil kernels can be specified in two forms:
      (See :py:meth:`~pycsou.operator.linop.stencil.stencil.Stencil.__init__` for details.)

      * A single non-seperable :math:`D`-dimensional kernel :math:`k[i_{1},\ldots,i_{D}]` of shape
        :math:`(k_{1},\ldots,k_{D})`.
      * A sequence of seperable :math:`1`-dimensional kernel(s) :math:`k_{d}[i]` of shapes
        :math:`(k_{1},),\ldots,(k_{D},)` such that :math:`k[i_{1},\ldots,i_{D}] = \Pi_{d=1}^{D}
        k_{d}[i_{d}]`.

    Example
    -------

    * **Moving average of a 1D signal**

      Let :math:`x[n]` denote a 1D signal.
      The weighted moving average

      .. math::

         y[n] = x[n-2] + 2 x[n-1] + 3 x[n]

      can be viewed as the output of the 3-point stencil of kernel :math:`h = [1, 2, 3]`.

      .. code-block:: python3

         import numpy as np
         from pycsou.operator.linop import Stencil

         x = np.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

         op = Stencil(
             arg_shape=x.shape,
             kernel=np.array([1., 2, 3]),
             center=(2,),  # h[2] applies on x[n]
         )

         y = op.apply(x)  # [0, 3, 8, 14, 20, 26, 32, 38, 44, 50]


    * **Image filtering 1**

      Let :math:`x[n, m]` denote a 2D image.
      The blurred image

      .. math::

         y[n, m] = \frac{1}{4} x[n-1,m-1] + \frac{1}{4} x[n-1,m+1] + \frac{1}{4} x[n+1,m-1] +
         \frac{1}{4} x[n+1,m+1]

      can be viewed as the output of the 9-point stencil

      .. math::

         h =
         \left[
         \begin{array}{ccc}
            \frac{1}{4} & 0 & \frac{1}{4} \\
            0 & 0 & 0 \\
            \frac{1}{4} & 0 & \frac{1}{4}
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

         c = 0.25
         op = Stencil(
             arg_shape=x.shape,
             kernel=np.array(
                 [[c, 0, c],
                  [0, 0, 0],
                  [c, 0, c]]),
             center=(1, 1),  # h[1, 1] applies on x[n, m]
         )

         y = op.apply(x.reshape(-1)).reshape(8, 8)
         # [[ 2.25,  4.5 ,  5.  ,  5.5 ,  6.  ,  6.5 ,  7.  ,  3.5 ]
         #  [ 4.5 ,  9.  , 10.  , 11.  , 12.  , 13.  , 14.  ,  7.  ]
         #  [ 8.5 , 17.  , 18.  , 19.  , 20.  , 21.  , 22.  , 11.  ]
         #  [12.5 , 25.  , 26.  , 27.  , 28.  , 29.  , 30.  , 15.  ]
         #  [16.5 , 33.  , 34.  , 35.  , 36.  , 37.  , 38.  , 19.  ]
         #  [20.5 , 41.  , 42.  , 43.  , 44.  , 45.  , 46.  , 23.  ]
         #  [24.5 , 49.  , 50.  , 51.  , 52.  , 53.  , 54.  , 27.  ]
         #  [12.25, 24.5 , 25.  , 25.5 , 26.  , 26.5 , 27.  , 13.5 ]]

    * **Image filtering 2**

      Following the example above, notice that :math:`y[n, m]` can be implemented more efficiently
      by factoring the 9-point stencil as a cascade of two 3-point stencils:

      .. math::

         h =
         \left[
         \begin{array}{ccc}
            \frac{1}{2} & 0 & \frac{1}{2}
         \end{array}
         \right]
         \left[
         \begin{array}{c}
            \frac{1}{2} \\ 0 \\ \frac{1}{2}
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

         c = 0.5
         op = Stencil(
             arg_shape=x.shape,
             kernel=[
                 np.array([c, 0, c]),  # h1: stencil along 1st axis (rows)
                 np.array([c, 0, c]),  # h2: stencil along 2nd axis (columns)
             ],
             center=(1, 1),  # h1[1] * h2[1] applies on x[n, m]
         )

         y = op.apply(x.reshape(-1)).reshape(8, 8)
         # [[ 2.25,  4.5 ,  5.  ,  5.5 ,  6.  ,  6.5 ,  7.  ,  3.5 ]
         #  [ 4.5 ,  9.  , 10.  , 11.  , 12.  , 13.  , 14.  ,  7.  ]
         #  [ 8.5 , 17.  , 18.  , 19.  , 20.  , 21.  , 22.  , 11.  ]
         #  [12.5 , 25.  , 26.  , 27.  , 28.  , 29.  , 30.  , 15.  ]
         #  [16.5 , 33.  , 34.  , 35.  , 36.  , 37.  , 38.  , 19.  ]
         #  [20.5 , 41.  , 42.  , 43.  , 44.  , 45.  , 46.  , 23.  ]
         #  [24.5 , 49.  , 50.  , 51.  , 52.  , 53.  , 54.  , 27.  ]
         #  [12.25, 24.5 , 25.  , 25.5 , 26.  , 26.5 , 27.  , 13.5 ]]
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
              is filtered by stencil `kernel[k]`.
        center: IndexSpec
            (i_1, ..., i_D) index of the stencil's center.

            `center` defines how a kernel is overlaid on inputs to produce outputs.

            .. math::

               y[i_{1},\ldots,i_{D}]
               =
               \sum_{q_{1},\ldots,q_{D}=0}^{k_{1},\ldots,k_{D}}
               x[i_{1} - c_{1} + q_{1},\ldots,i_{D} - c_{D} + q_{D}]
               \,\cdot\,
               k[q_{1},\ldots,q_{D}]
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

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        x = self._pad.apply(arr)
        x = x.reshape(-1, *self._pad._pad_shape)

        y = self._stencil_chain(x, self._st_fw)
        y = y.reshape(*arr.shape[:-1], -1)

        out = self._trim.apply(y)
        return out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        x = self._trim.adjoint(arr)
        x = x.reshape(-1, *self._pad._pad_shape)

        y = self._stencil_chain(x, self._st_bw)
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
        #
        # Caution: `x` is modified in-place.
        y = x.copy()
        for st in stencils:
            st.apply(x, y, **self._dispatch_params)
            x, y = y, x
        y = x
        return y

    def lipschitz(self, **kwargs) -> pyct.Real:
        if kwargs.get("recompute", False):
            if kwargs.get("algo", "svds") == "fro":
                # inform hutchpp() to run at specific precision
                kwargs.update(dtype=self._dtype)
            self._lipschitz = pyca.SquareOp.lipschitz(self, **kwargs)
        else:
            # An analytic upper bound:
            #     \norm{A x}{2}^{2}
            #  =  \sum_{n} [A x]_{n}^{2}
            #  =  \sum_{n} |<h_n, x>|^{2}
            # \le \sum_{n} \norm{h_n}{2}^{2} \norm{x}{2}^{2}
            # \le \norm{h}{2}^{2} \sum_{n} \norm{x}{2}^{2}
            # \le \norm{h}{2}^{2} N \norm{x}{2}^{2}
            #
            # -> L \le \sqrt{N} \norm{h}{2}
            kernels = [st._kernel for st in self._st_fw]
            kernel = functools.reduce(operator.mul, kernels, 1)
            L_st = np.linalg.norm(pycu.to_NUMPY(kernel).reshape(-1))
            L_st *= np.sqrt(self.dim)

            L_pad = self._pad.lipschitz()
            L_trim = self._trim.lipschitz()

            L_ub = L_trim * L_st * L_pad
            self._lipschitz = min(L_ub, self._lipschitz)
        return self._lipschitz

    def to_sciop(self, **kwargs):
        # Stencil.apply/adjoint() only support the precision provided at init-time.
        kwargs.update(dtype=self._dtype)
        op = pyca.SquareOp.to_sciop(self, **kwargs)
        return op

    def asarray(self, **kwargs) -> pyct.NDArray:
        # Stencil.apply() only supports the precision provided at init-time.
        xp = pycu.get_array_module(self._st_fw[0]._kernel)
        _A = super().asarray(xp=xp, dtype=self._dtype)

        xp = kwargs.get("xp", pycd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pycrt.getPrecision().value)
        A = xp.array(pycu.to_NUMPY(_A), dtype=dtype)
        return A

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
        return tr
