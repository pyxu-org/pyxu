"""
Low-level functions used to define user-facing stencils.
"""
import collections.abc as cabc
import itertools
import string

import numpy as np

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt


def _signature(params, returns) -> str:
    # Translate a signature of the form
    #     [in_1_spec, ..., in_N_spec] -> out_spec
    # to Numba's string representation.
    #
    # Parameters
    # ----------
    # params: list(spec)
    # returns: spec | None
    #
    # Returns
    # -------
    # sig: str
    #
    # Notes
    # -----
    # A parameter spec is characterized by the triplet
    #     (dtype[single/double], ndim[int], c_contiguous[bool])
    def fmt(spec) -> str:
        dtype, ndim, c_contiguous = spec

        _dtype_spec = {
            pxrt.Width.SINGLE: "float32",
            pxrt.Width.DOUBLE: "float64",
        }[pxrt.Width(dtype)]

        dim_spec = [":"] * ndim
        if c_contiguous and (ndim > 0):
            dim_spec[-1] = "::1"
        dim_spec = "[" + ",".join(dim_spec) + "]"

        _repr = _dtype_spec
        if ndim > 0:
            _repr += dim_spec
        return _repr

    sig = "".join(
        [
            "void" if (returns is None) else fmt(returns),
            "(",
            ", ".join(map(fmt, params)),
            ")",
        ]
    )
    return sig


class _Stencil:
    """
    Multi-dimensional JIT-compiled stencil. (Low-level function.)

    This low-level class creates a gu-vectorized stencil applicable on multiple inputs simultaneously.

    Create instances via factory method :py:meth:`~pyxu.operator._Stencil.init`.

    Example
    -------
    Correlate a stack of images `A` with a (3, 3) kernel such that:

    .. math::

       B[n, m] = A[n-1, m] + A[n, m-1] + A[n, m+1] + A[n+1, m]

    .. code-block:: python3

       import numpy as np
       from pyxu.operator.linop.stencil._stencil import _Stencil

       # create the stencil
       kernel = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]], dtype=np.float64)
       center = (1, 1)
       stencil = _Stencil.init(kernel, center)

       # apply it to the data
       rng = np.random.default_rng()
       A = rng.normal(size=(2, 3, 4, 30, 30))  # 24 images of size (30, 30)
       B = np.zeros_like(A)
       stencil.apply(A, B)  # (2, 3, 4, 30, 30)
    """

    IndexSpec = cabc.Sequence[pxt.Integer]

    @staticmethod
    def init(
        kernel: pxt.NDArray,
        center: IndexSpec,
    ):
        """
        Parameters
        ----------
        kernel: NDArray
            (k_1, ..., k_D) kernel coefficients.

            Only float32/64 kernels are supported.
        center: ~pyxu.operator._Stencil.IndexSpec
            (D,) index of the kernel's center.

        Returns
        -------
        st: ~pyxu.operator.linop.stencil._stencil._Stencil
            Rank-D stencil.
        """
        dtype = kernel.dtype
        if dtype not in {_.value for _ in pxrt.Width}:
            raise ValueError(f"Unsupported kernel precision {dtype}.")

        center = np.array(center, dtype=int)
        assert center.size == kernel.ndim
        assert np.all(0 <= center) and np.all(center < kernel.shape)

        N = pxd.NDArrayInfo
        ndi = N.from_obj(kernel)
        if ndi == N.NUMPY:
            klass = _Stencil_NP
        elif ndi == N.CUPY:
            klass = _Stencil_CP
        else:
            raise NotImplementedError

        st = klass(kernel, center)
        return st

    def apply(
        self,
        arr: pxt.NDArray,
        out: pxt.NDArray,
        **kwargs,
    ) -> pxt.NDArray:
        r"""
        Evaluate stencil on multiple inputs.

        Parameters
        ----------
        arr: NDArray
            (..., N_1, ..., N_D) data to process.
        out: NDArray
            (..., N_1, ..., N_D) array to which outputs are written.
        kwargs: dict
            Extra kwargs to configure `f_jit()`, the Dispatcher instance created by Numba.

            Only relevant for GPU stencils, with values:

            * blockspergrid: int
            * threadsperblock: int

            Default values are chosen if unspecified.

        Returns
        -------
        out: NDArray
            (..., N_1, ..., N_D) outputs.

        Notes
        -----
        * `arr` and `out` must have the same type/dtype as the kernel used during instantiation.
        * Index regions in `out` where the stencil is not fully supported are set to 0.
        * :py:meth:`~pyxu.operator._Stencil.apply` may raise ``CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`` when the number of
          GPU registers required exceeds resource limits.  There are 2 solutions to this problem:

            (1) Pass the `max_registers` kwarg to f_jit()'s decorator; or
            (2) `Limit the number of threads per block <https://stackoverflow.com/a/68659008>`_.

          (1) must be set at compile time; it is thus left unbounded.
          (2) is accessible through .apply(\*\*kwargs).
        """
        assert arr.dtype == out.dtype == self._kernel.dtype
        assert arr.shape == out.shape
        # assert arr.flags.c_contiguous and out.flags.c_contiguous
        # [2022.12.25, Sepand]
        #   Preferable to explicitly enforce C-contiguity for better performance, but then
        #   LinOp.to_sciop() is partially broken.

        K_dim = len(self._kernel.shape)
        arg_shape = arr.shape[-K_dim:]

        stencil = self._configure_dispatcher(arr.size, **kwargs)
        stencil(
            arr.reshape(-1, *arg_shape),
            out.reshape(-1, *arg_shape),
        )
        return out

    def __init__(
        self,
        kernel: pxt.NDArray,
        center: pxt.NDArray,
    ):
        self._kernel = kernel
        self._center = center
        self._code = self._gen_code()

        exec(self._code, locals())  # compile stencil
        self._dispatch = eval("f_jit")  # keep track of JIT Dispatcher

    def _gen_code(self) -> str:
        # Generate code which creates `f_jit()` after execution.
        raise NotImplementedError

    def _configure_dispatcher(self, pb_size: int, **kwargs) -> cabc.Callable:
        # Configure `f_jit()`, the Numba Dispatcher instance.
        #
        # Parameters
        # ----------
        # pb_size: int
        #     Number of stencil evaluations.
        # **kwargs: dict
        #
        # Returns
        # -------
        # f: callable
        #     Configured Numba Dispatcher.
        raise NotImplementedError


class _Stencil_NP(_Stencil):
    def _gen_code(self) -> str:
        _template = string.Template(
            r"""
import numba

@numba.stencil(
    func_or_mode="constant",
    cval=0,
)
def f_stencil(a):
    # rank-D kernel [specified as rank-(D+1) to work across stacked dimension].
    return ${stencil_spec}

@numba.jit(
    "${signature}",
    nopython=True,
    nogil=True,
    forceobj=False,
    parallel=True,
    error_model="numpy",
    cache=False,  # not applicable to dynamically-defined functions (https://github.com/numba/numba/issues/3501)
    fastmath=True,
    boundscheck=False,
)
def f_jit(arr, out):
    # arr: (N_stack, N_1, ..., N_D)
    # out: (N_stack, N_1, ..., N_D)
    f_stencil(arr, out=out)
"""
        )
        code = _template.substitute(
            signature=self.__sig_spec(),
            stencil_spec=self.__stencil_spec(),
        )
        return code

    def _configure_dispatcher(self, pb_size: int, **kwargs) -> cabc.Callable:
        # Nothing to do for CPU targets.
        return self._dispatch

    def __sig_spec(self) -> str:
        sig_spec = (self._kernel.dtype, self._kernel.ndim + 1, False)
        signature = _signature([sig_spec, sig_spec], None)
        return signature

    def __stencil_spec(self) -> str:
        f_fmt = {  # coef float-formatter
            pxrt.Width.SINGLE: "1.8e",
            pxrt.Width.DOUBLE: "1.16e",
        }[pxrt.Width(self._kernel.dtype)]

        entry = []
        _range = list(map(range, self._kernel.shape))
        for idx in itertools.product(*_range):
            idx_c = [i - c for (i, c) in zip(idx, self._center)]
            idx_c = ",".join(map(str, idx_c))

            cst = self._kernel[idx]
            if np.isclose(cst, 0):
                # no useless look-ups at runtime
                e = None
            elif np.isclose(cst, 1):
                # no multiplication required
                e = f"a[0,{idx_c}]"
            else:
                # general case
                e = f"({cst:{f_fmt}} * a[0,{idx_c}])"

            if e is not None:
                entry.append(e)

        spec = " + ".join(entry)
        return spec


class _Stencil_CP(_Stencil):
    def _gen_code(self) -> str:
        _template = string.Template(
            r"""
import numba.cuda

@numba.cuda.jit(
    device=True,
    inline=True,
    fastmath=True,
    opt=True,
    cache=False,
)
def unravel_offset(
    offset,  # int
    shape,  # (D+1,)
):
    # Compile-time-equivalent of np.unravel_index(offset, shape, order='C').
    ${unravel_spec}
    return idx


@numba.cuda.jit(
    device=True,
    inline=True,
    fastmath=True,
    opt=True,
    cache=False,
)
def full_overlap(
    idx,  # (D+1,)
    dimensions,  # (D+1,)
):
    # Computes intersection of:
    # * idx[0] < dimensions[0]
    # * 0 <= idx[1:] - kernel_center
    # * idx[1:] - kernel_center <= dimensions[1:] - kernel_width
    if not (idx[0] < dimensions[0]):
        # went beyond stack-dim limit
        return False

    kernel_width = ${kernel_width}
    kernel_center = ${kernel_center}
    for i, w, c, n in zip(
        idx[1:],
        kernel_width,
        kernel_center,
        dimensions[1:],
    ):
        if not (0 <= i - c <= n - w):
            # kernel partially overlaps ROI around `idx`
            return False

    # kernel fully overlaps ROI around `idx`
    return True


@numba.cuda.jit(
    func_or_sig="${signature}",
    device=False,
    inline=False,
    fastmath=True,
    opt=True,
    cache=False,  # not applicable to dynamically-defined functions (https://github.com/numba/numba/issues/3501)
    # max_registers=None,  # see .apply() notes
)
def f_jit(arr, out):
    # arr: (N_stack, N_1, ..., N_D)
    # out: (N_stack, N_1, ..., N_D)
    offset = numba.cuda.grid(1)
    if offset < arr.size:
        idx = unravel_offset(offset, arr.shape)
        if full_overlap(idx, arr.shape):
            out[idx] = ${stencil_spec}
        else:
            out[idx] = 0
"""
        )
        code = _template.substitute(
            kernel_center=str(tuple(self._center)),
            kernel_width=str(self._kernel.shape),
            signature=self.__sig_spec(),
            stencil_spec=self.__stencil_spec(),
            unravel_spec=self.__unravel_spec(),
        )
        return code

    def _configure_dispatcher(self, pb_size: int, **kwargs) -> cabc.Callable:
        # Set (`threadsperblock`, `blockspergrid`)
        assert set(kwargs.keys()) <= {
            "threadsperblock",
            "blockspergrid",
        }

        attr = self._kernel.device.attributes
        tpb = kwargs.get("threadsperblock", attr["MaxThreadsPerBlock"])
        bpg = kwargs.get("blockspergrid", (pb_size // tpb) + 1)
        return self._dispatch[bpg, tpb]

    def __sig_spec(self) -> str:
        sig_spec = (self._kernel.dtype, self._kernel.ndim + 1, False)
        signature = _signature([sig_spec, sig_spec], None)
        return signature

    def __stencil_spec(self) -> str:
        f_fmt = {  # coef float-formatter
            pxrt.Width.SINGLE: "1.8e",
            pxrt.Width.DOUBLE: "1.16e",
        }[pxrt.Width(self._kernel.dtype)]

        entry = []
        _range = list(map(range, self._kernel.shape))
        for idx in itertools.product(*_range):
            # create string of form "idx[1]+i1,...,idx[K]+iK"
            idx_c = [i - c for (i, c) in zip(idx, self._center)]
            idx_c = [f"idx[{i1}]{i2:+d}" for (i1, i2) in enumerate(idx_c, start=1)]
            idx_c = ",".join(idx_c)

            cst = self._kernel[idx]
            if np.isclose(cst, 0):
                # no useless look-ups at runtime
                e = None
            elif np.isclose(cst, 1):
                # no multiplication required
                e = f"arr[idx[0],{idx_c}]"
            else:
                # general case
                e = f"({cst:{f_fmt}} * arr[idx[0],{idx_c}])"

            if e is not None:
                entry.append(e)

        spec = " + ".join(entry)
        return spec

    def __unravel_spec(self) -> str:
        N = self._kernel.ndim + 1  # 1 stack-dim
        entry = []

        # left = offset
        e = "left = offset"
        entry.append(e)

        # blk = prod(shape)
        e = "blk = " + " * ".join([f"shape[{n}]" for n in range(N)])
        entry.append(e)

        for n in range(N):
            # blk //= shape[n]
            e = f"blk //= shape[{n}]"
            entry.append(e)

            # i{n} = left // blk
            e = f"i{n} = left // blk"
            entry.append(e)

            # left -= i{n} * blk
            e = f"left -= i{n} * blk"
            entry.append(e)

        # idx = (i0, ..., i{N})
        e = "idx = (" + ", ".join([f"i{n}" for n in range(N)]) + ")"
        entry.append(e)

        # indent each entry by 4, then concatenate
        for i in range(1, len(entry)):  # 1st line skipped
            entry[i] = "    " + entry[i]
        spec = "\n".join(entry)
        return spec
