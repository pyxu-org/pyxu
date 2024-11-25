"""
Low-level functions used to define user-facing stencils.
"""
import collections.abc as cabc
import itertools
import pathlib as plib
import string
import types

import numpy as np

import pyxu.info.config as pxcfg
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu


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
    Only NUMPY/CUPY arrays are accepted.

    Create instances via factory method :py:meth:`~pyxu.operator._Stencil.init`.

    Example
    -------
    Correlate a stack of images `A` with a (3, 3) kernel such that:

    .. math::

       B[n, m] = A[n-1, m] + A[n, m-1] + A[n, m+1] + A[n+1, m]

    .. code-block:: python3

       import numpy as np
       from pyxu.operator import _Stencil

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
            (k1,...,kD) kernel coefficients.

            Only float32/64 kernels are supported.
        center: ~pyxu.operator._Stencil.IndexSpec
            (D,) index of the kernel's center.

        Returns
        -------
        st: ~pyxu.operator._Stencil
            Rank-D stencil.
        """
        dtype = kernel.dtype
        if dtype not in {_.value for _ in pxrt.Width}:
            raise ValueError(f"Unsupported kernel precision {dtype}.")

        center = np.array(center, dtype=int)
        assert center.size == kernel.ndim
        assert np.all((0 <= center) & (center < kernel.shape))

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
            (..., M1,...,MD) data to process.
        out: NDArray
            (..., M1,...,MD) array to which outputs are written.
        kwargs: dict
            Extra kwargs to configure `f_jit()`, the Dispatcher instance created by Numba.

            Only relevant for GPU stencils, with values:

            * blockspergrid: int
            * threadsperblock: int

            Default values are chosen if unspecified.

        Returns
        -------
        out: NDArray
            (..., M1,...,MD) outputs.

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
        assert arr.flags.c_contiguous and out.flags.c_contiguous

        K_dim = len(self._kernel.shape)
        dim_shape = arr.shape[-K_dim:]

        stencil = self._configure_dispatcher(arr.size, **kwargs)
        stencil(
            # OK since NP/CP input constraint.
            arr.reshape(-1, *dim_shape),
            out.reshape(-1, *dim_shape),
        )
        return out

    def __init__(
        self,
        kernel: pxt.NDArray,
        center: pxt.NDArray,
    ):
        self._kernel = kernel
        self._center = center

        cached_module = self._gen_code()
        self._dispatch = cached_module.f_jit

    def _gen_code(self) -> types.ModuleType:
        # Compile Numba kernel `void f_jit(arr, out)`.
        #
        # The code is compiled only if unavailable beforehand.
        #
        # Returns
        # -------
        # jit_module: module
        #     A (loaded) python package containing method f_jit().
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
    def _gen_code(self) -> types.ModuleType:
        # Generate the code which should be compiled --------------------------
        sig_spec = (self._kernel.dtype, self._kernel.ndim + 1, True)
        signature = _signature((sig_spec,) * 2, None)

        template_file = plib.Path(__file__).parent / "_template_cpu.txt"
        with open(template_file, mode="r") as f:
            template = string.Template(f.read())
        code = template.substitute(
            signature=signature,
            stencil_spec=self.__stencil_spec(),
        )
        # ---------------------------------------------------------------------

        # Store/update cached version as needed.
        module_name = pxu.cache_module(code)
        pxcfg.cache_dir(load=True)  # make the Pyxu cache importable (if not already done)
        jit_module = pxu.import_module(module_name)
        return jit_module

    def _configure_dispatcher(self, pb_size: int, **kwargs) -> cabc.Callable:
        # Nothing to do for CPU targets.
        return self._dispatch

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
    def _gen_code(self) -> types.ModuleType:
        # Generate the code which should be compiled --------------------------
        sig_spec = (self._kernel.dtype, self._kernel.ndim + 1, True)
        signature = _signature((sig_spec,) * 2, None)

        template_file = plib.Path(__file__).parent / "_template_gpu.txt"
        with open(template_file, mode="r") as f:
            template = string.Template(f.read())
        code = template.substitute(
            kernel_center=str(tuple(self._center.tolist())),
            kernel_width=str(self._kernel.shape),
            signature=signature,
            stencil_spec=self.__stencil_spec(),
            unravel_spec=self.__unravel_spec(),
        )
        # ---------------------------------------------------------------------

        # Store/update cached version as needed.
        module_name = pxu.cache_module(code)
        pxcfg.cache_dir(load=True)  # make the Pyxu cache importable (if not already done)
        jit_module = pxu.import_module(module_name)
        return jit_module

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
