import string
import typing as typ

import numba
import numba.cuda
import numpy as np

import pycsou.util as pycu
import pycsou.util.ptype as pyct


def make_nd_stencil(coefficients: pyct.NDArray, center: pyct.NDArray) -> typ.Callable:
    r"""
    Create a multidimensional Just-In-Time (JIT) compiled stencil from a given set of coefficients.

    The stencil computation is based on Numba's stencils, which work through kernels that look like standard Python
    function definitions but with different semantics with respect to array indexing. Numba stencils allow clearer,
    more concise code and enable higher performance through parallelization of the stencil execution (see
    `Numba stencils <https://numba.pydata.org/numba-doc/latest/user/stencil.html>`_ for further reference).

    Parameters
    ----------
    coefficients: NDArray
        Kernel coefficients. Must have the same number of dimensions as the input array's shape (without the
        stacking dimensions).

    center: NDArray
        Index of the kernel's center. Must be a 1-dimensional array with one element per dimension in ``coefficients``.

    Returns
    -------
    stencil
        JIT-ted stencil.

    Examples
    ________
    .. code-block:: python3

        import dask.array as da
        import numpy as np

        from pycsou.math.stencil import make_nd_stencil

        D = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
        center = np.array([1, 1])

        # NUMPY
        img = np.random.normal(0, 10, size=(1000, 100, 100))
        stencil = make_nd_stencil(D, center)
        out = stencil(img)

        # BOUNDARY CONDITIONS DASK
        depth_post = D.shape - center - 1
        depth_pre = center
        depth = {0: 0}
        depth.update({i + 1: (depth_pre[i], depth_post[i]) for i in range(D.ndim)})
        boundary = {i: "none" for i in range(D.ndim + 1)}

        # DASK
        img_da = da.asarray(img, chunks=(100, 10, 10))
        out_da = img_da.map_overlap(stencil, depth=depth, boundary=boundary, dtype=D.dtype)

        # Need to handle equally the borders
        print(np.allclose(out[1:-1, 1:-1, 1:-1], out_da.compute()[1:-1, 1:-1, 1:-1]))

    See also
    ________
    :py:func:`~pycsou.math.stencil.make_nd_stencil_gpu`
    :py:class:`~pycsou.operator.linop.base._StencilOp
    """

    indices = np.indices(coefficients.shape).reshape(coefficients.ndim, -1).T - center[None, ...]
    coefficients = pycu.compute(coefficients).ravel()
    kernel_string = _create_kernel_string(coefficients, indices)
    exec(_stencil_string.substitute(kernel=kernel_string, dtype=coefficients.dtype), globals())
    return my_stencil


def make_nd_stencil_gpu(coefficients: pyct.NDArray, center: pyct.NDArray, func_name: str) -> typ.Callable:
    r"""
    Create a multidimensional Just-In-Time (JIT) compiled GPU stencil from a given set of coefficients.

    Numba supports a JIT compilation of stencil computations (see :py:func:`~pycsou.math.stencil.make_nd_stencil`)
    with CUDA on compatible GPU devices.

    Parameters
    ----------
    coefficients: NDArray
        Kernel coefficients. Must have the same number of dimensions as the input array's shape (without the
        stacking dimensions).

    center: NDArray
        Index of the kernel's center. Must be a 1-dimensional array with one element per dimension in ``coefficients``.

    Returns
    -------
    gpu-stencil
        JIT-ted CUDA kernel

    Examples
    ________
    .. code-block:: python3

        import cupy as cp
        import numpy as np

        from pycsou.math.stencil import make_nd_stencil, make_nd_stencil_gpu

        D = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
        center = np.array([1, 1])

        # NUMPY
        img = np.random.normal(0, 10, size=(1000, 100, 100))
        stencil = make_nd_stencil(D, center)
        out = stencil(img)



        # CUPY
        img_cp = cp.asarray(img)
        stencil_cp = make_nd_stencil_gpu(cp.asarray(D), center)
        out_cp = cp.zeros_like(img_cp)
        threadsperblock = (10, 10, 10)
        blockspergrid_x = math.ceil(img_cp.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(img_cp.shape[1] / threadsperblock[1])
        blockspergrid_z = math.ceil(img_cp.shape[2] / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        stencil_cp[blockspergrid, threadsperblock](img_cp, out_cp)
        print(np.allclose(out[1:-1, 1:-1, 1:-1], out_cp.get()[1:-1, 1:-1, 1:-1]))
        print("Done")

    See also
    ________
    :py:func:`~pycsou.math.stencil.make_nd_stencil`
    :py:class:`~pycsou.operator.linop.base._StencilOp`

    """
    # If kernel is 3D, then the computation has to loop over samples (numba cuda grid can have at maximum 3 dimensions)
    if coefficients.ndim > 3:
        raise ValueError("The Stencil operator does not support GPU kernels with more than 3D dims.")

    stacking_dim = 1 if coefficients.ndim < 3 else 0
    letters1 = list(string.ascii_lowercase)[: coefficients.ndim + stacking_dim]
    letters2 = list(string.ascii_lowercase)[coefficients.ndim + stacking_dim : 2 * (coefficients.ndim + stacking_dim)]

    indices = np.indices(coefficients.shape).reshape(coefficients.ndim, -1).T - center
    kernel_string = _create_kernel_string_gpu(letters1, coefficients.ravel(), indices, stacking_dim)
    if_statement = _create_if_statement_gpu(letters1, letters2, indices, coefficients, stacking_dim)

    exec(
        _stencil_string_gpu.substitute(
            func_name=func_name,
            letters1=", ".join(letters1),
            letters2=", ".join(letters2),
            len_letters=str(len(letters1)),
            if_statement=if_statement,
            kernel=kernel_string,
            dtype=f"{coefficients.dtype}["
            + ",".join(
                [
                    ":",
                ]
                * (coefficients.ndim + stacking_dim)
            )
            + "]",
        ),
        globals(),
    )
    return globals()[f"kernel_cupy_{func_name}"]


def _create_kernel_string(coefficients: pyct.NDArray, indices: pyct.NDArray) -> str:
    return " + ".join(
        [
            f"a[0, {', '.join([str(elem) for elem in ids_])}] * np.{coefficients.dtype}({str(coefficients[i])})"
            for i, ids_ in enumerate(indices)
        ]
    )


def _create_kernel_string_gpu(letters1: typ.List, coefficients: pyct.NDArray, indices: pyct.NDArray, stacking_dim: int):
    stacking_str = f"{letters1[0]}, " if stacking_dim else ""
    return f" + ".join(
        [
            f"arr[{stacking_str}{', '.join(['+'.join([letters1[e + stacking_dim], str(elem)]) for e, elem in enumerate(ids_)])}] * cp.{coefficients.dtype}({str(coefficients[i])})"
            for i, ids_ in enumerate(indices)
        ]
    )


def _create_if_statement_gpu(
    letters1: typ.List, letters2: typ.List, indices: pyct.NDArray, coefficients: pyct.NDArray, stacking_dim: int
):
    stacking_str = f"(0 <= {letters1[0]} < {letters2[0]}) and " if stacking_dim else ""
    return stacking_str + " and ".join(
        [
            f"({-np.min(indices, axis=0)[i]} <= {letters1[i + stacking_dim]} < ({letters2[i+ stacking_dim]} - {np.max(indices, axis=0)[i]}))"
            for i in range(coefficients.ndim)
        ]
    )


# Cache cannot be used with string defined functions (see https://github.com/numba/numba/issues/3501 for workaround)
_stencil_string = string.Template(
    r"""
@numba.njit(parallel=True, fastmath=True, nogil=True)
def my_stencil(arr):
    stencil = numba.stencil(
    lambda a: ${kernel},
    signature=["${dtype}(${dtype})"]
    )(arr)
    return stencil"""
)
_stencil_string_gpu = string.Template(
    r"""
import cupy as cp
#@numba.cuda.jit('void(${dtype},${dtype})') # fails for float32
@numba.cuda.jit
def kernel_cupy_${func_name}(arr, out):
    $letters1 = numba.cuda.grid(${len_letters}) # j, k
    $letters2 = arr.shape # n, m
    if $if_statement:
        out[$letters1] = ${kernel}"""
)
