import collections.abc as cabc
import functools
import itertools
import math
import typing as typ

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.operator.blocks as pycb
import pycsou.operator.linop as pycl
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

try:
    import scipy.ndimage._filters as scif
except ImportError:
    import scipy.ndimage.filters as scif

__all__ = [
    "_FiniteDifference",
    "_GaussianDerivative",
    "PartialDerivative",
    "Gradient",
    "Jacobian",
    "Hessian",
    "DirectionalDerivative",
    "DirectionalGradient",
    "DirectionalLaplacian",
    "DirectionalHessian",
    "StructureTensor",
]

ModeSpec = typ.Union[str, cabc.Sequence[str]]

KernelSpec = typ.Union[
    pyct.NDArray,  # (k1, ..., kD) non-seperable kernel
    cabc.Sequence[pyct.NDArray],  # [(k1,), ..., (kD,)] seperable kernels
]


def _BaseDifferential(
    kernel: KernelSpec,
    center: KernelSpec,
    arg_shape: pyct.NDArrayShape,
    mode: ModeSpec = "constant",
    gpu: bool = False,
    dtype: typ.Optional[pyct.DType] = None,
):
    r"""
    Helper base class for differential operators based on Numba stencils (see
    https://numba.pydata.org/numba-doc/latest/user/stencil.html).

    See Also
    --------
    :py:class:`~pycsou.operator.linop.stencil.stencil.Stencil`, :py:func:`~pycsou.math.stencil.make_nd_stencil`,
    :py:class:`~pycsou.operator.linop.diff._FiniteDifferences`,
    :py:class:`~pycsou.operator.linop.diff._GaussianDerivative`,
    :py:class:`~pycsou.operator.linop.diff.PartialDerivative`, :py:class:`~pycsou.operator.linop.diff.Gradient`,
    :py:class:`~pycsou.operator.linop.diff.Hessian`.

    Parameters
    ----------
    kernel: KernelSpec
            Stencil coefficients.
            Two forms are accepted:

            * NDArray of rank-:math:`D`: denotes a non-seperable stencil.
            * tuple[NDArray_1, ..., NDArray_D]: a sequence of 1D stencils such that is filtered by the stencil
              `kernel[d]` along the :math:`d`-th dimension.
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
    arg_shape: tuple
        Shape of the input array.
    mode: str | list(str)
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: pyct.DType
        Working precision of the linear operator.
    """

    if dtype is None:
        dtype = pycrt.getPrecision().value

    if gpu:
        assert pycd.CUPY_ENABLED
        import cupy as xp
    else:
        import numpy as xp

    if isinstance(kernel, cabc.Sequence):
        for i in range(len(kernel)):
            kernel[i] = xp.array(kernel[i], dtype=dtype)
    else:
        kernel = xp.array(kernel, dtype=dtype)

    return pycl.Stencil(arg_shape=arg_shape, kernel=kernel, center=center, mode=mode)


def _sanitize_init_kwargs(
    order: typ.Union[pyct.Integer, typ.Tuple[pyct.Integer, ...]],
    arg_shape: pyct.NDArrayShape,
    sampling: typ.Union[pyct.Integer, typ.Tuple[pyct.Integer, ...]],
    diff_method: str,
    diff_kwargs: dict,
    axis: typ.Union[pyct.Integer, typ.Tuple] = None,
) -> typ.Tuple[
    typ.Tuple[pyct.Integer, ...],
    typ.Tuple[pyct.Integer, ...],
    typ.Union[typ.Tuple[pyct.Real, ...], typ.Tuple[str, ...]],
    typ.Tuple[pyct.Integer, ...],
    typ.Tuple[pyct.Integer, ...],
]:
    r"""
    Ensures that inputs have the appropriate shape and values.
    """

    def _ensure_tuple(param, param_name: str) -> typ.Union[tuple[pyct.Integer, ...], tuple[str, ...]]:
        r"""
        Enforces the input parameters to be tuples of the same size as `arg_shape`.
        """
        if not isinstance(param, cabc.Sequence) or isinstance(param, str):
            param = (param,)
        assert (len(param) == 1) | (len(param) <= len(arg_shape)), (
            f"The length of {param_name} cannot be larger than the"
            f"number of dimensions ({len(arg_shape)}) defined by `arg_shape`"
        )
        return param

    order = _ensure_tuple(order, param_name="order")
    sampling = _ensure_tuple(sampling, param_name="sampling")
    if len(sampling) == 1:
        sampling = sampling * len(arg_shape)
    assert all([_ > 0 for _ in order]), "Order must be strictly positive"
    assert all([_ > 0 for _ in sampling]), "Sampling must be strictly positive"

    if diff_method == "fd":
        param1_name = "diff_type"
        param2_name = "accuracy"
        param1 = diff_kwargs.get("diff_type", "forward")
        param2 = diff_kwargs.get("accuracy", 1)
    elif diff_method == "gd":
        param1_name = "sigma"
        param2_name = "truncate"
        param1 = diff_kwargs.get("sigma", 1.0)
        param2 = diff_kwargs.get("truncate", 3.0)
    else:
        raise NotImplementedError

    _param1 = _ensure_tuple(param1, param_name=param1_name)
    _param2 = _ensure_tuple(param2, param_name=param2_name)

    if param1_name == "sigma":
        assert all([p >= 0 for p in _param1]), "Sigma must be strictly positive"
    if param2_name == "accuracy":
        assert all([p >= 0 for p in _param2]), "Accuracy must be positive"
    elif param2_name == "truncate":
        assert all([p > 0 for p in _param2]), "Truncate must be strictly positive"

    if len(order) != len(arg_shape):
        assert axis is not None, (
            "If `order` is not a tuple with size of arg_shape, then `axis` must be" " specified. Got `axis=None`"
        )
        axis = _ensure_tuple(axis, param_name="axis")
        assert len(axis) == len(order), "`axis` must have the same number of elements as `order`"

    else:
        if axis is not None:
            axis = _ensure_tuple(axis, param_name="axis")
            assert len(axis) == len(order), "`axis` must have the same number of elements as `order`"
        else:
            axis = tuple([i for i in range(len(arg_shape))])

    if not (len(_param1) == len(order)):
        assert len(_param1) == 1, (
            f"Parameter `{param1_name}` inconsistent with the number of elements in " "parameter `order`."
        )
        _param1 = _param1 * len(order)

    if not (len(_param2) == len(order)):
        assert len(_param2) == 1, (
            f"Parameter `{param2_name}` inconsistent with the number of elements in " "parameter `order`."
        )
        _param2 = _param2 * len(order)

    return (
        order,
        sampling,
        _param1,
        _param2,
        axis,
    )


def _create_kernel(arg_shape, axis, _fill_coefs) -> typ.Tuple[pyct.NDArray, pyct.NDArray]:
    r"""
    Creates kernel for stencil.
    """
    stencil_ids = [np.array([0])] * len(arg_shape)
    stencil_coefs = [np.array([1.0])] * len(arg_shape)
    center = np.zeros(len(arg_shape), dtype=int)

    # Create finite difference coefficients for each dimension
    for i, ax in enumerate(axis):
        stencil_ids[ax], stencil_coefs[ax], center[ax] = _fill_coefs(i)

    # Create a kernel composing all dimensions coefficients
    # kernel = np.zeros([np.ptp(ids) + 1 if ids else 1 for ids in stencil_ids])
    # for i, ax in enumerate(axis):
    #     slices = tuple(
    #         [slice(center[j], center[j] + 1) if j != ax else slice(None) for j in range(len(arg_shape))]
    #     )
    #     shape = [1 if j != ax else kernel.shape[ax] for j in range(len(arg_shape))]
    #     kernel[slices] += stencil_coefs[ax].reshape(shape)

    # return kernel, center
    return stencil_coefs, center


def _FiniteDifference(
    order: typ.Union[pyct.Integer, tuple[pyct.Integer, ...]],
    arg_shape: pyct.NDArrayShape,
    diff_type: typ.Union[str, tuple[str, ...]] = "forward",
    axis: typ.Union[pyct.Integer, tuple[pyct.Integer, ...], None] = None,
    accuracy: typ.Union[pyct.Integer, tuple[pyct.Integer, ...]] = 1,
    mode: ModeSpec = "constant",
    gpu: bool = False,
    dtype: typ.Optional[pyct.DType] = None,
    sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 1,
    return_linop: bool = True,
):
    r"""
    Finite difference base operator along a single dimension.

    This class is used by :py:class:`~pycsou.operator.linop.diff.PartialDerivative`,
    :py:class:`~pycsou.operator.linop.diff.Gradient` and :py:class:`~pycsou.operator.linop.diff.Hessian`.
    See :py:class:`~pycsou.operator.linop.diff.PartialDerivative.finite_difference` for documentation.

    See Also
    --------
    :py:class:`~pycsou.operator.linop.diff._BaseDifferential`, :py:class:`~pycsou.operator.linop.diff._GaussianDerivative`,
    :py:class:`~pycsou.operator.linop.diff.PartialDerivative`, :py:class:`~pycsou.operator.linop.diff.Gradient`,
    :py:class:`~pycsou.operator.linop.diff.Hessian`.

    Parameters
    ----------
    order: int | tuple
        Derivative order. If a single integer value is provided, then `axis` should be provided to indicate which
        dimension should be differentiated. If a tuple is provided, it should contain as many elements as `arg_shape`.
    arg_shape: tuple
        Shape of the input array.
    diff_type: str | tuple
        Type of finite differences ["forward", "backward", "central"]. Defaults to "forward".
    axis: int | tuple | None
        Axis to which apply the derivative. It maps the argument `order` to the specified dimensions of the input
        array. Defaults to None, assuming that the `order` argument has as many elements as dimensions of the input.
    accuracy: int | tuple
        Determines the number of points used to approximate the derivative with finite differences (see `Notes`).
        Defaults to 1. If an int is provided, the same `accuracy` is assumed for all dimensions.
        If a tuple is provided, it should have as many elements as `arg_shape`.
    mode: str | list(str)
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: pyct.DType
        Working precision of the linear operator.
    sampling: float | tuple
        Sampling step (i.e. distance between two consecutive elements of an array). Defaults to 1.
    return_linop: bool
        Whether to return a linear operator object (True) or a tuple with the finite differences kernel and its center.
    """
    diff_kwargs = {"diff_type": diff_type, "accuracy": accuracy}
    order, sampling, diff_type, accuracy, axis = _sanitize_init_kwargs(
        order=order,
        diff_method="fd",
        diff_kwargs=diff_kwargs,
        arg_shape=arg_shape,
        axis=axis,
        sampling=sampling,
    )

    def _compute_ids(order: pyct.Integer, diff_type: str, accuracy: pyct.Integer) -> list:
        """
        Computes the Finite difference indices according to the order, type and accuracy.
        """
        if diff_type == "central":
            n_coefs = 2 * ((order + 1) // 2) - 1 + accuracy
            ids = np.arange(-(n_coefs // 2), n_coefs // 2 + 1, dtype=int)
        else:
            n_coefs = order + accuracy
            if diff_type == "forward":
                ids = np.arange(0, n_coefs, dtype=int)
            elif diff_type == "backward":
                ids = np.arange(-n_coefs + 1, 1, dtype=int)
            else:
                raise ValueError(
                    f"Incorrect value for variable 'type'. 'type' should be ['forward', 'backward', "
                    f"'central'], but got {diff_type}."
                )
        return ids.tolist()

    def _compute_coefficients(stencil_ids: list, order: pyct.Integer, sampling: pyct.Real) -> pyct.NDArray:
        """
        Computes the finite difference coefficients based on the order and indices.
        """
        # vander doesn't allow precision specification
        stencil_mat = np.vander(
            np.array(stencil_ids),
            increasing=True,
        ).T.astype(pycrt.getPrecision().value)
        vec = np.zeros(len(stencil_ids), dtype=pycrt.getPrecision().value)
        vec[order] = math.factorial(order)
        coefs = np.linalg.solve(stencil_mat, vec)
        coefs /= sampling**order
        return coefs

    # FILL COEFFICIENTS
    def _fill_coefs(i: pyct.Integer) -> typ.Tuple[list, pyct.NDArray, pyct.Integer]:
        r"""
        Defines kernel elements.
        """
        stencil_ids = _compute_ids(order=order[i], diff_type=diff_type[i], accuracy=accuracy[i])
        stencil_coefs = _compute_coefficients(stencil_ids=stencil_ids, order=order[i], sampling=sampling[i])
        center = stencil_ids.index(0)
        return stencil_ids, stencil_coefs, center

    kernel, center = _create_kernel(arg_shape, axis, _fill_coefs)

    if return_linop:
        op = _BaseDifferential(kernel=kernel, center=center, arg_shape=arg_shape, mode=mode, gpu=gpu, dtype=dtype)
        op._name = "FiniteDifference"
        return op
    else:
        return kernel, center


def _GaussianDerivative(
    order: typ.Union[pyct.Integer, tuple[pyct.Integer, ...]],
    arg_shape: pyct.NDArrayShape,
    sigma: typ.Union[pyct.Real, tuple[pyct.Real, ...]],
    axis: typ.Union[pyct.Integer, tuple[pyct.Integer, ...], None] = None,
    truncate: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 3.0,
    mode: ModeSpec = "constant",
    gpu: bool = False,
    dtype: typ.Optional[pyct.DType] = None,
    sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 1,
    return_linop: bool = True,
):
    r"""
    Gaussian derivative base operator along a single dimension.

    This class is used by :py:class:`~pycsou.operator.linop.diff.PartialDerivative`,
    :py:class:`~pycsou.operator.linop.diff.Gradient` and :py:class:`~pycsou.operator.linop.diff.Hessian`.
    See :py:class:`~pycsou.operator.linop.diff.PartialDerivative.gaussian_derivative` for documentation.

    See Also
    --------
    :py:class:`~pycsou.operator.linop.diff._BaseDifferential`, :py:class:`~pycsou.operator.linop.diff._FiniteDifference`,
    :py:class:`~pycsou.operator.linop.diff.PartialDerivative`, :py:class:`~pycsou.operator.linop.diff.Gradient`,
    :py:class:`~pycsou.operator.linop.diff.Hessian`.

    Parameters
    ----------
    order: int | tuple
        Derivative order. If a single integer value is provided, then `axis` should be provided to indicate which
        dimension should be used for differentiation. If a tuple is provided, it should contain as many elements as
        number of dimensions in `axis`.
    arg_shape: tuple
        Shape of the input array.
    sigma: float | tuple
        Standard deviation of the Gaussian kernel.
    axis: int | tuple | None
        Axis to which apply the derivative. It maps the argument `order` to the specified dimensions of the input
        array. Defaults to None, assuming that the `order` argument has as many elements as dimensions of the input.
    truncate: float | tuple
        Truncate the filter at this many standard deviations.
        Defaults to 3.0.
    mode: str | list(str)
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: pyct.DType
        Working precision of the linear operator.
    sampling: float | tuple
        Sampling step (i.e., the distance between two consecutive elements of an array). Defaults to 1.
    return_linop: bool
        Whether to return a linear operator object (True) or a tuple with the finite differences kernel and its center.
    """
    diff_kwargs = {"sigma": sigma, "truncate": truncate}
    order, sampling, sigma, truncate, axis = _sanitize_init_kwargs(
        order=order,
        diff_method="gd",
        diff_kwargs=diff_kwargs,
        arg_shape=arg_shape,
        axis=axis,
        sampling=sampling,
    )

    def _fill_coefs(i: pyct.Integer) -> typ.Tuple[list, pyct.NDArray, pyct.Integer]:
        r"""
        Defines kernel elements.
        """
        # make the radius of the filter equal to `truncate` standard deviations
        sigma_pix = sigma[i] / sampling  # Sigma rescaled to pixel units
        radius = int(truncate[i] * float(sigma_pix) + 0.5)
        stencil_coefs = _gaussian_kernel1d(sigma=sigma_pix, order=order[i], sampling=sampling[i], radius=radius)
        stencil_ids = [i for i in range(-radius, radius + 1)]
        return stencil_ids, stencil_coefs, radius

    def _gaussian_kernel1d(
        sigma: pyct.Real, order: pyct.Integer, sampling: pyct.Real, radius: pyct.Integer
    ) -> pyct.NDArray:
        """
        Computes a 1-D Gaussian convolution kernel.
        Wraps scipy.ndimage.filters._gaussian_kernel1d
        It flips the output because the original kernel is meant for convolution instead of correlation.
        """
        coefs = np.flip(scif._gaussian_kernel1d(sigma, order, radius))
        coefs /= sampling**order
        return coefs

    kernel, center = _create_kernel(arg_shape, axis, _fill_coefs)
    if return_linop:
        op = _BaseDifferential(kernel=kernel, center=center, arg_shape=arg_shape, mode=mode, gpu=gpu, dtype=dtype)
        op._name = "GaussianDerivative"
        return op
    else:
        return kernel, center


class PartialDerivative:
    r"""
    Partial derivative operator based on `Numba stencils <https://numba.pydata.org/numba-doc/latest/user/stencil.html>`_.

    Notes
    -----
    This operator approximates the partial derivative of a :math:`D`-dimensional signal
    :math:`\mathbf{f} \in \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}`

    .. math::

        \frac{\partial^{n} \mathbf{f}}{\partial x_0^{n_0} \cdots \partial x_{D-1}^{n_{D-1}}} \in
        \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}

    where :math:`\frac{\partial^{n_i}}{\partial x_i^{n_i}}` is the :math:`n_i`-th order partial derivative along
    dimension :math:`i` and :math:`n = \prod_{i=0}^{D-1} n_{i}` is the total derivative order.

    Partial derivatives can be implemented with `finite differences
    <https://en.wikipedia.org/wiki/Finite_difference>`_ via the
    :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.finite_difference` constructor or with
    `Gaussian derivatives <https://www.crisluengo.net/archives/22/>`_ via the
    :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.gaussian_derivative` constructor.

    See Also
    --------
    :py:class:`~pycsou.operator.linop.diff.Gradient`, :py:class:`~pycsou.operator.linop.diff.Laplacian`,
    :py:class:`~pycsou.operator.linop.diff.Hessian`.
    """

    @staticmethod
    def finite_difference(
        order: tuple[pyct.Integer, ...],
        arg_shape: pyct.NDArrayShape,
        diff_type: typ.Union[str, tuple[str, ...]] = "forward",
        accuracy: typ.Union[pyct.Integer, tuple[pyct.Integer, ...]] = 1,
        mode: ModeSpec = "constant",
        gpu: bool = False,
        dtype: typ.Optional[pyct.DType] = None,
        sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 1,
    ) -> pyco.LinOp:
        r"""
        Compute partial derivatives for multidimensional signals using finite differences.

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array.
        order: tuple
            Derivative order for each dimension. The total order of the partial derivative is the sum of the elements of
            the tuple.
        diff_type: str | tuple
            Type of finite differences ['forward, 'backward, 'central']. Defaults to 'forward'. If a string is provided,
            the same `diff_type` is assumed for all dimensions. If a tuple is provided, it should have as many elements
            as `order`.
        accuracy: int | tuple
            Determines the number of points used to approximate the derivative with finite differences (see `Notes`).
            Defaults to 1. If an int is provided, the same `accuracy` is assumed for all dimensions.
            If a tuple is provided, it should have as many elements as `arg_shape`.
        mode: str | list(str)
            Boundary conditions.
            Multiple forms are accepted:

            * str: unique mode shared amongst dimensions.
              Must be one of:

              * 'constant' (default): zero-padding
              * 'wrap'
              * 'reflect'
              * 'symmetric'
              * 'edge'
            * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

            (See :py:func:`numpy.pad` for details.)
        gpu: bool
            Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
        dtype: pyct.DType
            Working precision of the linear operator.
        sampling: float | tuple
            Sampling step (i.e. distance between two consecutive elements of an array). Defaults to 1.

        Returns
        -------
        op: :py:class:`~pycsou.abc.operator.LinOp`
            Partial derivative

        Notes
        -----

        We explain here finite differences for one-dimensional signals; this operator performs finite differences for
        multidimensional signals along dimensions specified by ``order``.

        This operator approximates derivatives with `finite differences <https://en.wikipedia.org/wiki/Finite_difference>`_.
        It is inspired by the `Finite Difference Coefficients Calculator <https://web.media.mit.edu/~crtaylor/calculator.html>`_
        to construct finite-difference approximations for the desired *(i)* derivative order, *(ii)* approximation accuracy,
        and *(iii)* finite difference type. Three basic types of finite differences are supported, which lead to the
        following first-order (``order = 1``) operators with ``accuracy = 1`` and sampling step ``sampling = h`` for
        one-dimensional signals :

        - **Forward difference**:
            Approximates the continuous operator :math:`D_{F}f(x) = \frac{f(x+h) - f(x)}{h}` with the discrete operator

            .. math::

                \mathbf{D}_{F} f [n] = \frac{f[n+1] - f[n]}{h},

            whose kernel is :math:`d = \frac{1}{h}[-1, 1]` and center is (0, ).

        - **Backward difference**:
            Approximates the continuous operator :math:`D_{B}f(x) = \frac{f(x) - f(x-h)}{h}` with the discrete operator

            .. math::

                \mathbf{D}_{F} f [n] = \frac{f[n] - f[n-1]}{h},

            whose kernel is :math:`d = \frac{1}{h}[-1, 1]` and center is (1, ).

        - **Central difference**:
            Approximates the continuous operator :math:`D_{C}f(x) = \frac{f(x+h) - f(x-h)}{2h}` with the discrete
            operator

            .. math::

                \mathbf{D}_{F} f [n] = \frac{f[n+1] - f[n-1]}{2h},

            whose kernel is :math:`d = \frac{1}{h}[-\frac12, 0, \frac12]` and center is (1, ).

        .. warning::
            For forward and backward differences, higher-order operators correspond to the composition of first-order operators.
            This is not the case for central differences: the second-order continuous operator is given by
            :math:`D^2_{C}f(x) = \frac{f(x+h) - 2 f(x) + f(x-h)}{h}`, hence :math:`D^2_{C} \neq D_{C} \circ D_{C}`. The
            corresponding discrete operator is given by :math:`\mathbf{D}^2_{C} f [n] = \frac{f[n+1] - 2 f[n] + f[n-1]}{h}`,
            whose kernel is :math:`d = \frac{1}{h}[1, -2, 1]` and center is (1, ). We refer to `this paper
            <https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf>`_
            for more details.

        For a given derivative order :math:`N\in\mathbb{Z}^{+}` and accuracy :math:`a\in\mathbb{Z}^{+}`, the size
        :math:`N_s` of the stencil kernel :math:`d` used for finite differences is given by:

        - For central differences:
            :math:`N_s = 2 \lfloor\frac{N + 1}{2}\rfloor - 1 + a`

        - For forward and backward differences:
            :math:`N_s = N + a`

        For :math:`N_s` given support indices :math:`\{s_1, \ldots , s_{N_s} \} \subset \mathbb{Z}` and a derivative
        order :math:`N<N_s`, the stencil kernel :math:`d = [d_1, \ldots, d_{N_s}]` of the finite-difference approximation of the
        derivative is obtained by solving the following system of linear equations (see the `Finite Difference
        Coefficients Calculator <https://web.media.mit.edu/~crtaylor/calculator.html>`_ documentation):

        .. math::

            \left(\begin{array}{ccc}
            s_{1}^{0} & \cdots & s_{N_s}^{0} \\
            \vdots & \ddots & \vdots \\
            s_{1}^{N_s-1} & \cdots & s_{N_s}^{N_s-1}
            \end{array}\right)\left(\begin{array}{c}
            d_{1} \\
            \vdots \\
            d_{N_s}
            \end{array}\right)= \frac{1}{h^{N}}\left(\begin{array}{c}
            \delta_{0, N} \\
            \vdots \\
            \delta_{i, N} \\
            \vdots \\
            \delta_{N_s-1, N}
            \end{array}\right),

        where :math:`\delta_{i, j}` is the Kronecker delta.

        This class inherits its methods from :py:class:`~pycsou.operator.linop.stencil.stencil.Stencil`.

        Example
        -------

        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from pycsou.operator.linop.diff import PartialDerivative
           from pycsou.util.misc import peaks
           x = np.linspace(-2.5, 2.5, 25)
           xx, yy = np.meshgrid(x, x)
           image = peaks(xx, yy)
           arg_shape = image.shape  # Shape of our image
           # Specify derivative order at each direction
           df_dx = (1, 0) # Compute derivative of order 1 in first dimension
           d2f_dy2 = (0, 2) # Compute derivative of order 2 in second dimension
           d3f_dxdy2 = (1, 2) # Compute derivative of order 1 in first dimension and der. of order 2 in second dimension
           # Instantiate derivative operators
           diff1 = PartialDerivative.finite_difference(order=df_dx, arg_shape=arg_shape, diff_type="central")
           diff2 = PartialDerivative.finite_difference(order=d2f_dy2, arg_shape=arg_shape, diff_type="central")
           diff = PartialDerivative.finite_difference(order=d3f_dxdy2, arg_shape=arg_shape, diff_type="central")
           # Compute derivatives
           out1 = (diff1 * diff2)(image.reshape(1, -1)).reshape(arg_shape)
           out2 = diff(image.reshape(1, -1)).reshape(arg_shape)
           plt.figure()
           plt.imshow(image.T),
           plt.axis('off')
           plt.colorbar()
           plt.title('f(x,y)')
           plt.figure()
           plt.imshow(out1.T)
           plt.axis('off')
           plt.title(r'$\frac{\partial^{3} f(x,y)}{\partial x\partial y^{2}}$')
           plt.figure()
           plt.imshow(out2.T)
           plt.axis('off')
           plt.title(r'$\frac{\partial^{3} f(x,y)}{\partial x\partial y^{2}}$')
           # Test
           assert np.allclose(out1, out2)

        """

        axis = np.where(np.array(order) > 0)[0]
        order = tuple(np.array(order)[axis])
        order, sampling, diff_type, accuracy, axis = _sanitize_init_kwargs(
            order=order,
            param1=diff_type,
            param1_name="diff_type",
            param2=accuracy,
            param2_name="accuracy",
            arg_shape=arg_shape,
            axis=tuple(axis),
            sampling=sampling,
        )

        # Compute a kernel for each axis
        kernel = [np.array(1)] * len(arg_shape)
        center = np.zeros(len(arg_shape), dtype=int)
        for i in range(len(order)):
            if order[i] > 0:
                k, c = _FiniteDifference(
                    order=order[i],
                    arg_shape=arg_shape,
                    diff_type=diff_type[i],
                    axis=axis[i],
                    accuracy=accuracy[i],
                    mode=mode[i],
                    gpu=gpu,
                    dtype=dtype,
                    sampling=sampling[i],
                    return_linop=False,
                )
                kernel[axis[i]] = k[axis[i]]
                center[axis[i]] = c[axis[i]]

        return _BaseDifferential(kernel=kernel, center=center, arg_shape=arg_shape, mode=mode, gpu=gpu, dtype=dtype)

    @staticmethod
    def gaussian_derivative(
        arg_shape: pyct.NDArrayShape,
        order: tuple[pyct.Integer, ...],
        sigma: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 1.0,
        truncate: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 3.0,
        mode: ModeSpec = "constant",
        gpu: bool = False,
        dtype: typ.Optional[pyct.DType] = None,
        sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 1,
    ) -> pyco.LinOp:
        r"""
        Compute partial derivatives for multidimensional signals using gaussian derivatives.

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array.
        order: tuple
            Derivative order for each dimension. The total order of the partial derivative is the sum of the elements of
            the tuple.
        sigma: float | tuple
            Standard deviation for the Gaussian kernel. Defaults to 1.0.
            If a float is provided, the same `sigma` is assumed for all dimensions. If a tuple is provided, it should have as many elements as `order`.
        truncate: float | tuple
            Truncate the filter at this many standard deviations. Defaults to 3.0.
            If a float is provided, the same `truncate` is assumed for all dimensions. If a tuple is provided, it should have as many elements as `order`.
        mode: str | list(str)
            Boundary conditions.
            Multiple forms are accepted:

            * str: unique mode shared amongst dimensions.
              Must be one of:

              * 'constant' (default): zero-padding
              * 'wrap'
              * 'reflect'
              * 'symmetric'
              * 'edge'
            * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

            (See :py:func:`numpy.pad` for details.)
        gpu: bool
            Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
        dtype: pyct.DType
            Working precision of the linear operator.
        sampling: float | tuple
            Sampling step (i.e., the distance between two consecutive elements of an array). Defaults to 1.

        Returns
        -------
        op: :py:class:`~pycsou.abc.operator.LinOp`
            Partial derivative

        Notes
        -----
        We explain here Gaussian derivatives for one-dimensional signals; this operator performs partial Gaussian
        derivatives for multidimensional signals along dimensions specified by ``order``.

        A Gaussian derivative is an approximation of a derivative that consists in convolving the input function with a
        Gaussian function :math:`g` before applying a derivative. In the continuous domain, the :math:`N`-th order
        Gaussian derivative :math:`D^N_G` amounts to a convolution with the :math:`N`-th order derivative of :math:`g`:

        .. math::

            D^N_G f (x) = \frac{\mathrm{d}^N (f * g) }{\mathrm{d} x^N} (x) = f(x) * \frac{\mathrm{d}^N g}{\mathrm{d} x^N} (x).

        For discrete signals :math:`f[n]`, this operator is approximated by

        .. math::

            \mathbf{D}^N_G f [n] = f[n] *\frac{\mathrm{d}^N g}{\mathrm{d} x^N} \left(\frac{n}{h}\right),

        where :math:`h` is the spacing between samples and the operator :math:`*` is now a discrete convolution.

        .. warning::
            The operator :math:`\mathbf{D}_{G} \circ \mathbf{D}_{G}` is not directly related to
            :math:`\mathbf{D}_{G}^{2}`: Gaussian smoothing is performed twice in the case of the former, whereas it is
            performed only once in the case of the latter.

        Note that in contrast with finite differences (see
        :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.finite_difference`), Gaussian derivatives compute exact
        derivatives in the continuous domain, since Gaussians can be differentiated analytically. This derivative is
        then sampled in order to perform a discrete convolution.

        This class inherits its methods from :py:class:`~pycsou.operator.linop.stencil.stencil.Stencil`.

        Example
        -------

        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from pycsou.operator.linop.diff import PartialDerivative
           from pycsou.util.misc import peaks
           x = np.linspace(-2.5, 2.5, 25)
           xx, yy = np.meshgrid(x, x)
           image = peaks(xx, yy)
           arg_shape = image.shape  # Shape of our image
           # Specify derivative order at each direction
           df_dx = (1, 0) # Compute derivative of order 1 in first dimension
           d2f_dy2 = (0, 2) # Compute derivative of order 2 in second dimension
           d3f_dxdy2 = (1, 2) # Compute derivative of order 1 in first dimension and der. of order 2 in second dimension
           # Instantiate derivative operators
           diff1 = PartialDerivative.gaussian_derivative(order=df_dx, arg_shape=arg_shape, sigma=2.0)
           diff2 = PartialDerivative.gaussian_derivative(order=d2f_dy2, arg_shape=arg_shape, sigma=2.0)
           diff = PartialDerivative.gaussian_derivative(order=d3f_dxdy2, arg_shape=arg_shape, sigma=2.0)
           # Compute derivatives
           out1 = (diff1 * diff2)(image.reshape(1, -1)).reshape(arg_shape)
           out2 = diff(image.reshape(1, -1)).reshape(arg_shape)
           plt.figure()
           plt.imshow(image.T),
           plt.axis('off')
           plt.colorbar()
           plt.title('f(x,y)')
           plt.figure()
           plt.imshow(out1.T)
           plt.axis('off')
           plt.title(r'$\frac{\partial^{3} f(x,y)}{\partial x\partial y^{2}}$')
           plt.figure()
           plt.imshow(out2.T)
           plt.axis('off')
           plt.title(r'$\frac{\partial^{3} f(x,y)}{\partial x\partial y^{2}}$')
           # Test
           assert np.allclose(out1, out2)

        """

        axis = np.where(np.array(order) > 0)[0]
        order = tuple(np.array(order)[axis])
        order, sampling, sigma, truncate, axis = _sanitize_init_kwargs(
            order=order,
            param1=sigma,
            param1_name="diff_type",
            param2=truncate,
            param2_name="accuracy",
            arg_shape=arg_shape,
            axis=tuple(axis),
            sampling=sampling,
        )

        # Compute a kernel for each axis
        kernel = [np.array(1)] * len(arg_shape)
        center = np.zeros(len(arg_shape), dtype=int)
        for i in range(len(order)):
            if order[i] > 0:
                k, c = _GaussianDerivative(
                    order=order[i],
                    arg_shape=arg_shape,
                    sigma=sigma[i],
                    axis=axis[i],
                    truncate=truncate[i],
                    mode=mode[i],
                    gpu=gpu,
                    dtype=dtype,
                    sampling=sampling[i],
                    return_linop=False,
                )
                kernel[axis[i]] = k[axis[i]]
                center[axis[i]] = c[axis[i]]

        return _BaseDifferential(kernel=kernel, center=center, arg_shape=arg_shape, mode=mode, gpu=gpu, dtype=dtype)


def _make_unravelable(op, arg_shape=None):
    def unravel(self, arr):
        return arr.reshape(*arr.shape[:-1], -1, *self.arg_shape)

    def ravel(self, arr):
        return arr.reshape(*arr.shape[: -1 - len(self.arg_shape)], -1)

    if arg_shape is not None:
        setattr(op, "arg_shape", arg_shape)

    setattr(op, "unravel", functools.partial(unravel, op))
    setattr(op, "ravel", functools.partial(ravel, op))
    return op


class _BaseVecDifferential:
    r"""
    Helper class for Gradient and Hessian.

    Defines a method for computing and stacking partial derivatives.

    See Also
    --------
    :py:class:`~pycsou.operator.linop.diff.Gradient`, :py:class:`~pycsou.operator.linop.diff.Laplacian`,
    :py:class:`~pycsou.operator.linop.diff.Hessian`.
    """

    @staticmethod
    def _stack_diff_ops(
        arg_shape,
        directions,
        diff_method,
        order,
        param1,
        param2,
        mode: ModeSpec = "constant",
        gpu: bool = False,
        dtype: typ.Optional[pyct.DType] = None,
        sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 1,
        parallel: bool = False,
    ):
        _param1 = np.empty(len(arg_shape), dtype=object)
        _param2 = np.empty(len(arg_shape), dtype=object)
        _param1[:] = param1
        _param2[:] = param2
        if isinstance(mode, str):
            mode = (mode,)
        if isinstance(mode, cabc.Sequence):
            if len(mode) != len(arg_shape):
                assert len(mode) == 1
                mode = mode * len(arg_shape)

        else:
            raise ValueError("mode has to be a string or a tuple")
        dif_op = []
        for i in range(0, len(directions)):
            _order = np.zeros_like(arg_shape)
            _order[directions[i]] = order[i]
            _directions = np.array(directions[i]).reshape(-1)
            param1 = _param1[_directions].tolist()
            param2 = _param2[_directions].tolist()
            if diff_method == "fd":
                dif_op.append(
                    PartialDerivative.finite_difference(
                        arg_shape=arg_shape,
                        order=tuple(_order),
                        diff_type=param1,
                        accuracy=param2,
                        mode=mode,
                        gpu=gpu,
                        dtype=dtype,
                        sampling=sampling,
                    )
                )

            elif diff_method == "gd":
                dif_op.append(
                    PartialDerivative.gaussian_derivative(
                        arg_shape=arg_shape,
                        order=tuple(_order),
                        sigma=param1,
                        truncate=param2,
                        mode=mode,
                        gpu=gpu,
                        dtype=dtype,
                        sampling=sampling,
                    )
                )
        return _make_unravelable(pycb.vstack(dif_op, parallel=parallel), arg_shape)

    @staticmethod
    def _check_directions_and_order(
        arg_shape, directions
    ) -> typ.Tuple[typ.Union[tuple[pyct.Integer, ...], tuple[tuple[pyct.Integer, ...], ...]], bool]:
        def _check_directions(_directions):
            assert all(0 <= _ <= (len(arg_shape) - 1) for _ in _directions), (
                "Direction values must be between 0 and " "the number of dimensions in `arg_shape`."
            )

        if not isinstance(directions, cabc.Sequence):
            # This corresponds to [mode 0] in `Notes`
            directions = [directions, directions]
            _check_directions(directions)
            directions = (directions,)
        else:
            if isinstance(directions, str):
                # This corresponds to [mode 3] in Hessian `Notes`
                assert directions == "all", (
                    f"Value for `directions` not implemented. The accepted directions types are"
                    f"int, tuple or a str with the value `all`."
                )
                directions = tuple(
                    list(_) for _ in itertools.combinations_with_replacement(np.arange(len(arg_shape)).astype(int), 2)
                )
            elif not isinstance(directions[0], cabc.Sequence):
                # This corresponds to [mode 2] in Hessian  `Notes`
                assert len(directions) == 2, (
                    "If `directions` is a tuple, it should contain two elements, corresponding "
                    "to the i-th an j-th elements (dx_i and dx_j)"
                )
                directions = list(directions)
                _check_directions(directions)
                directions = (directions,)
            else:
                # This corresponds to [mode 3] in Hessian `Notes`
                for direction in directions:
                    _check_directions(direction)

        _directions = [
            list(direction) if (len(np.unique(direction)) == len(direction)) else np.unique(direction).tolist()
            for direction in directions
        ]

        _order = [3 - len(np.unique(direction)) for direction in directions]

        return _directions, _order


def Gradient(
    arg_shape: pyct.NDArrayShape,
    directions: typ.Optional[typ.Union[pyct.Integer, tuple[pyct.Integer, ...]]] = None,
    diff_method: str = "fd",
    mode: ModeSpec = "constant",
    gpu: bool = False,
    dtype: typ.Optional[pyct.DType] = None,
    sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 1,
    parallel: bool = False,
    **diff_kwargs,
):
    r"""
    Gradient operator based on `Numba stencils <https://numba.pydata.org/numba-doc/latest/user/stencil.html>`_.
    
    Notes
    -----

    This operator stacks the first-order partial derivatives of a :math:`D`-dimensional signal
    :math:`\mathbf{f} \in \mathbb{R}^{N_{0} \times \cdots \times N_{D-1}}` along each dimension:

    .. math::

        \boldsymbol{\nabla} \mathbf{f} = \begin{bmatrix}
        \frac{\partial \mathbf{f}}{\partial x_0} \\
        \vdots \\
        \frac{\partial \mathbf{f}}{\partial x_{D-1}}
        \end{bmatrix} \in \mathbb{R}^{D \times N_{0} \times \cdots \times N_{D-1}}

    The gradient can be approximated by `finite differences <https://en.wikipedia.org/wiki/Finite_difference>`_ via the
    :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.finite_difference` constructor or by the 
    `Gaussian derivative <https://www.crisluengo.net/archives/22/>`_ via
    :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.gaussian_derivative` constructor.
    
    Parameters
    ----------
    arg_shape: tuple
        Shape of the input array.
    directions: int | tuple | None
        Gradient directions. Defaults to `None`, which computes the gradient for all directions.
    diff_method: str ['gd', 'fd']
        Method used to approximate the derivative. Must be one of:

        * 'fd' (default): finite differences
        * 'gd': Gaussian derivative
    mode: str | list(str)
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: pyct.DType
        Working precision of the linear operator.
    sampling: float | tuple
        Sampling step (i.e., the distance between two consecutive elements of an array). Defaults to 1.
    parallel: bool
        If ``true``, use Dask to evaluate the different partial derivatives in parallel.
    diff_kwargs: dict
        Keyword arguments to parametrize partial derivatives (see
        :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.finite_difference` and
        :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.gaussian_derivative`)

    Returns
    -------
    op: :py:class:`~pycsou.abc.operator.LinOp`
        Gradient

    Example
    -------

    .. plot::
       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.operator.linop.diff import Gradient
       from pycsou.util.misc import peaks
       # Define input image
       n = 1000
       x = np.linspace(-3, 3, n)
       xx, yy = np.meshgrid(x, x)
       image = peaks(xx, yy)
       nsamples = 2
       arg_shape = image.shape # (1000, 1000)
       images = np.tile(image, (nsamples, 1, 1)).reshape(nsamples, -1)
       print(images.shape) # (2, 1000000)
       # Instantiate gradient operator
       grad = Gradient(arg_shape=arg_shape, sigma=1.0)
       # Compute gradients
       outputs = grad(images)
       print(outputs.shape) # (2, 2000000)
       # Plot
       df_dx = grad.unravel(outputs)[:, 0]
       df_dy = grad.unravel(outputs)[:, 1]
       plt.figure()
       plt.imshow(images[0].reshape(arg_shape))
       plt.colorbar()
       plt.title("Image")
       plt.axis("off")
       plt.figure()
       plt.imshow(df_dx[0])
       plt.colorbar()
       plt.title(r"$\partial f/ \partial x$")
       plt.axis("off")
       plt.figure()
       plt.imshow(df_dy[0])
       plt.colorbar()
       plt.title(r"$\partial f/ \partial y$")
       plt.axis("off")

    See Also
    --------
    :py:func:`~pycsou.operator.linop.diff.PartialDerivative`, :py:func:`~pycsou.operator.linop.diff.Jacobian`

    """

    directions = tuple([i for i in range(len(arg_shape))]) if directions is None else directions

    order, sampling, param1, param2, _ = _sanitize_init_kwargs(
        order=(1,) * len(directions),
        arg_shape=arg_shape,
        sampling=sampling,
        diff_method=diff_method,
        diff_kwargs=diff_kwargs,
    )
    return _BaseVecDifferential._stack_diff_ops(
        arg_shape=arg_shape,
        directions=directions,
        diff_method=diff_method,
        order=order,
        param1=param1,
        param2=param2,
        mode=mode,
        gpu=gpu,
        dtype=dtype,
        sampling=sampling,
        parallel=parallel,
    )


def Jacobian(
    arg_shape: pyct.NDArrayShape,
    n_channels: pyct.Integer,
    diff_method: str = "fd",
    mode: ModeSpec = "constant",
    gpu: bool = False,
    dtype: typ.Optional[pyct.DType] = None,
    sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 1,
    parallel: bool = False,
    **diff_kwargs,
):
    r"""
    Jacobian operator based on `Numba stencils <https://numba.pydata.org/numba-doc/latest/user/stencil.html>`_.

    Notes
    -----

    This operator computes the first-order partial derivatives of a :math:`D`-dimensional vector-valued signal of
    :math:`C` variables :math:`\mathbf{f} = [\mathbf{f}_{0}, \ldots, \mathbf{f}_{C-1}]` with
    :math:`\mathbf{f}_{c} \in \mathbb{R}^{N_{0} \times \cdots \times N_{D-1}}`.

    The Jacobian of :math:`\mathbf{f}` is computed via the gradient as follows:

    .. math::

        \mathbf{J} \mathbf{f} = \begin{bmatrix}
        (\boldsymbol{\nabla} \mathbf{f}_{0})^{\top} \\
        \vdots \\
        (\boldsymbol{\nabla} \mathbf{f}_{C-1})^{\top} \\
        \end{bmatrix} \in \mathbb{R}^{C \times D \times N_0 \times \cdots \times N_{D-1}}

    Parameters
    ----------
    arg_shape: tuple
        Shape of the input array.
    n_channels: int
        Number of channels or variables of the input vector-valued signal. The Jacobian with `n_channels==1` yields the
        gradient.
    diff_method: str ['gd', 'fd']
        Method used to approximate the derivative. Must be one of:

        * 'fd' (default): finite differences
        * 'gd': Gaussian derivative
    mode: str | list(str)
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: pyct.DType
        Working precision of the linear operator.
    sampling: float | tuple
        Sampling step (i.e., the distance between two consecutive elements of an array). Defaults to 1.
    parallel: bool
        If ``true``, use Dask to evaluate the different partial derivatives in parallel.
    diff_kwargs: dict
        Keyword arguments to parametrize partial derivatives (see
        :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.finite_difference` and
        :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.gaussian_derivative`)

    Returns
    -------
    op: :py:class:`~pycsou.abc.operator.LinOp`
        Jacobian

    Example
    -------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.operator.linop.diff import Jacobian
       from pycsou.util.misc import peaks
       x = np.linspace(-2.5, 2.5, 25)
       xx, yy = np.meshgrid(x, x)
       image = np.tile(peaks(xx, yy), (3, 1, 1))
       jac = Jacobian(arg_shape=image.shape[1:], n_channels=image.shape[0])
       out = jac.unravel(jac(image.ravel()))
       fig, axes = plt.subplots(2, 3, figsize=(15, 10))
       for i in range(2):
           for j in range(3):
               axes[i, j].imshow(out[i, j].T, cmap=["Reds", "Greens", "Blues"][j])
               axes[i, j].set_title(f"$\partial I_{{{['R', 'G', 'B'][j]}}}/\partial{{{['x', 'y'][i]}}}$")
       plt.suptitle("Jacobian")

    See Also
    --------
    :py:func:`~pycsou.operator.linop.diff.Gradient`, :py:func:`~pycsou.operator.linop.diff.PartialDerivative`

    """
    init_kwargs = dict(
        arg_shape=arg_shape,
        directions=None,
        diff_method=diff_method,
        mode=mode,
        gpu=gpu,
        dtype=dtype,
        sampling=sampling,
        parallel=parallel,
        **diff_kwargs,
    )

    grad = Gradient(**init_kwargs)

    op = pycb.block_diag(
        [
            grad,
        ]
        * n_channels
    )
    op._name = "Jacobian"
    return _make_unravelable(op, (n_channels, *arg_shape))


def Hessian(
    arg_shape: pyct.NDArrayShape,
    directions: typ.Union[
        str, tuple[pyct.Integer, pyct.Integer], tuple[tuple[pyct.Integer, pyct.Integer], ...]
    ] = "all",
    diff_method: str = "fd",
    mode: ModeSpec = "constant",
    gpu: bool = False,
    dtype: typ.Optional[pyct.DType] = None,
    sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 1,
    parallel: bool = False,
    **diff_kwargs,
):
    r"""
    Hessian operator based on `Numba stencils <https://numba.pydata.org/numba-doc/latest/user/stencil.html>`_.

    Notes
    -----

    The Hessian matrix or Hessian of a :math:`D`-dimensional signal :math:`\mathbf{f} \in \mathbb{R}^{N_0 \times \cdots
    \times N_{D-1}}` is the square matrix of second-order partial derivatives:

    .. math::

        \mathbf{H} \mathbf{f} = \begin{bmatrix}
        \dfrac{ \partial^{2}\mathbf{f} }{ \partial x_{0}^{2} } &  \dfrac{ \partial^{2}\mathbf{f} }{ \partial x_{0}\,\partial x_{1} } & \cdots & \dfrac{ \partial^{2}\mathbf{f} }{ \partial x_{0} \, \partial x_{D-1} } \\
        \dfrac{ \partial^{2}\mathbf{f} }{ \partial x_{1} \, \partial x_{0} } & \dfrac{ \partial^{2}\mathbf{f} }{ \partial x_{2}^{2} } & \cdots & \dfrac{ \partial^{2}\mathbf{f} }{\partial x_{1} \,\partial x_{D-1}} \\
        \vdots & \vdots & \ddots & \vdots \\
        \dfrac{ \partial^{2}\mathbf{f} }{ \partial x_{D-1} \, \partial x_{0} } & \dfrac{ \partial^{2}\mathbf{f} }{ \partial x_{D-1} \, \partial x_{1} } & \cdots & \dfrac{ \partial^{2}\mathbf{f} }{ \partial x_{D-1}^{2}}
        \end{bmatrix}

    The Hessian can be approximated by `finite differences <https://en.wikipedia.org/wiki/Finite_difference>`_ via the
    :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.finite_difference` constructor or by the `Gaussian derivative <https://www.crisluengo.net/archives/22/>`_ via
    :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.gaussian_derivative` constructor.

    The Hessian being symmetric, only the upper triangular part at most needs to be computed. Due to the (possibly)
    large size of the full Hessian, 4 different options are handled:

    * [mode 0] ``directions`` is an integer, e.g.:
      ``directions=0`` :math:`\rightarrow \partial^{2}\mathbf{f}/\partial x_{0}^{2}`.
    * [mode 1] ``directions`` is tuple of length 2, e.g.:
      ``directions=(0,1)`` :math:`\rightarrow  \partial^{2}\mathbf{f}/\partial x_{0}\partial x_{1}`.
    * [mode 2]  ``directions`` is tuple of tuples, e.g.:
      ``directions=((0,0), (0,1))`` :math:`\rightarrow  \left(\frac{ \partial^{2}\mathbf{f} }{ \partial x_{0}^{2} },
      \frac{ \partial^{2}\mathbf{f} }{ \partial x_{0}\partial x_{1} }\right)`.
    * [mode 3] ``directions = ''all''`` (default), computes the Hessian for all directions, i.e.:
      :math:`\rightarrow  \left(\frac{ \partial^{2}\mathbf{f} }{ \partial x_{0}^{2} }, \frac{ \partial^{2}\mathbf{f} }
      { \partial x_{0}\partial x_{1} }, \, \ldots , \, \frac{ \partial^{2}\mathbf{f} }{ \partial x_{D-1}^{2} }\right)`.

    The shape of the output :py:class:`~pycsou.abc.operator.LinOp` depends on the number of computed directions; by
    default (all directions), we have
    :math:`\mathbf{H} \mathbf{f} \in \mathbb{R}^{\frac{D(D-1)}{2} \times N_0 \times \cdots \times N_{D-1}}`.

    Parameters
    ----------

    arg_shape: tuple
        Shape of the input array.
    directions: int | tuple | None
        Hessian directions. Defaults to `all`, which computes the Hessian for all directions.
    diff_method: str ['gd', 'fd']
        Method used to approximate the derivative. Must be one of:

        * 'fd' (default): finite differences
        * 'gd': Gaussian derivative
    mode: str | list(str)
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: pyct.DType
        Working precision of the linear operator.
    sampling: float | tuple
        Sampling step (i.e., the distance between two consecutive elements of an array). Defaults to 1.
    parallel: bool
        If ``true``, use Dask to evaluate the different partial derivatives in parallel.
    diff_kwargs: dict
        Keyword arguments to parametrize partial derivatives (see
        :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.finite_difference` and
        :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.gaussian_derivative`)

    Returns
    -------
    op: :py:class:`~pycsou.abc.operator.LinOp`
        Hessian

    Example
    -------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.operator.linop.diff import Hessian, PartialDerivative
       from pycsou.util.misc import peaks
       # Define input image
       n = 1000
       x = np.linspace(-3, 3, n)
       xx, yy = np.meshgrid(x, x)
       image = peaks(xx, yy)
       nsamples = 2
       arg_shape = image.shape  # (1000, 1000)
       images = np.tile(image, (nsamples, 1, 1)).reshape(nsamples, -1)
       print(images.shape)  # (2, 1000000)
       # Instantiate Hessian operator
       directions = "all"
       hessian = Hessian(arg_shape=arg_shape, directions=directions)
       # Compute Hessian
       outputs = hessian(images)
       print(outputs.shape)  # (2, 3000000)
       # Plot
       outputs = hessian.unravel(outputs)
       print(outputs.shape)  # (2, 3, 1000, 1000)
       d2f_dx2 = outputs[:, 0]
       d2f_dxdy = outputs[:, 1]
       d2f_dy2 = outputs[:, 2]
       plt.figure()
       plt.imshow(images[0].reshape(arg_shape))
       plt.colorbar()
       plt.title("Image")
       plt.axis("off")
       plt.figure()
       plt.imshow(d2f_dx2[0].reshape(arg_shape))
       plt.colorbar()
       plt.title(r"$\partial^{2} f/ \partial x^{2}$")
       plt.axis("off")
       plt.figure()
       plt.imshow(d2f_dxdy[0].reshape(arg_shape))
       plt.colorbar()
       plt.title(r"$\partial^{2} f/ \partial x\partial y$")
       plt.axis("off")
       plt.figure()
       plt.imshow(d2f_dy2[0].reshape(arg_shape))
       plt.colorbar()
       plt.title(r"$\partial^{2} f/ \partial y^{2}$")
       plt.axis("off")

    See Also
    --------
    :py:class:`~pycsou.operator.linop.diff.PartialDerivative`, :py:class:`~pycsou.operator.linop.diff.Gradient`,
    :py:class:`~pycsou.operator.linop.diff.Laplacian`.
    """

    order, sampling, param1, param2, _ = _sanitize_init_kwargs(
        order=(1,) * len(arg_shape),
        diff_method=diff_method,
        diff_kwargs=diff_kwargs,
        arg_shape=arg_shape,
        sampling=sampling,
    )
    directions, order = _BaseVecDifferential._check_directions_and_order(arg_shape, directions)
    return _BaseVecDifferential._stack_diff_ops(
        arg_shape=arg_shape,
        directions=directions,
        diff_method=diff_method,
        order=order,
        param1=param1,
        param2=param2,
        mode=mode,
        gpu=gpu,
        dtype=dtype,
        sampling=sampling,
        parallel=parallel,
    )


def DirectionalDerivative(
    arg_shape: pyct.NDArrayShape,
    which: pyct.Integer,
    directions: pyct.NDArray,
    diff_method: str = "fd",
    mode: ModeSpec = "constant",
    gpu: bool = False,
    dtype: typ.Optional[pyct.DType] = None,
    sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 1,
    parallel: bool = False,
    **diff_kwargs,
):
    r"""
    Directional derivative operator based on `Numba stencils <https://numba.pydata.org/numba-doc/latest/user/stencil.html>`_.

    Parameters
    ----------
    arg_shape: tuple
        Shape of the input array.
    which: int
        Which directional derivative (restricted to 1: First or 2: Second, see ``Notes``).
    directions: NDArray
        Single direction (array of size :math:`(D,)`) or spatially-varying directions
        (array of size :math:`(D, N_0, \ldots, N_{D-1})`)
    diff_method: str ['gd', 'fd']
        Method used to approximate the derivative. Must be one of:

        * 'fd' (default): finite differences
        * 'gd': Gaussian derivative
    mode: str | list(str)
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: pyct.DType
        Working precision of the linear operator.
    sampling: float | tuple
        Sampling step (i.e., the distance between two consecutive elements of an array). Defaults to 1.
    diff_kwargs: dict
        Keyword arguments to parametrize partial derivatives (see
        :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.finite_difference` and
        :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.gaussian_derivative`)

    Returns
    -------
    op: :py:class:`~pycsou.abc.operator.LinOp`
            Directional derivative

    Notes
    -----
    The first-order ``DirectionalDerivative`` of a :math:`D`-dimensional signal :math:`\mathbf{f} \in
    \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}` applies a derivative along the direction specified by a constant
    unitary vector :math:`\mathbf{v} \in \mathbb{R}^D`:

    .. math::

        \boldsymbol{\nabla}_\mathbf{v} \mathbf{f} = \sum_{i=0}^{D-1} v_i \frac{\partial \mathbf{f}}{\partial x_i} \in
        \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}

    or along spatially-varying directions :math:`\mathbf{v} = [\mathbf{v}_0, \ldots , \mathbf{v}_{D-1}]^\top \in
    \mathbb{R}^{D \times N_0 \times \cdots \times N_{D-1} }` where each direction :math:`\mathbf{v}_{\cdot, i_0, \ldots
    , i_{D-1}} \in \mathbb{R}^D` for any :math:`0 \leq i_d \leq N_d-1` with :math:`0 \leq d \leq D-1` is a unitary vector:

    .. math::

        \boldsymbol{\nabla}_\mathbf{v} \mathbf{f} = \sum_{i=0}^{D-1} \mathbf{v}_i \odot
        \frac{\partial \mathbf{f}}{\partial x_i} \in \mathbb{R}^{N_0 \times \cdots \times N_{D-1}},

    where :math:`\odot` denotes the Hadamard (elementwise) product.

    Note that choosing :math:`\mathbf{v}= \mathbf{e}_d \in \mathbb{R}^D` (the :math:`d`-th canonical basis vector)
    amounts to the first-order :py:func:`~pycsou.operator.linop.diff.PartialDerivative` operator applied along axis
    :math:`d`.

    High-order directional derivatives :math:`\boldsymbol{\nabla}^N_\mathbf{v}` are obtained by composing the
    first-order directional derivative :math:`\boldsymbol{\nabla}_\mathbf{v}` :math:`N` times.

    .. warning::
        - :py:func:`~pycsou.operator.linop.diff.DirectionalDerivative` instances are **not arraymodule-agnostic**: they
          will only work with NDArrays belonging to the same array module as ``directions``. Inner
          computations may recast input arrays when the precision of ``directions`` does not match the user-requested
          precision.
        - ``directions`` are always normalized to be unit vectors.

    Example
    -------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.operator.linop.diff import DirectionalDerivative
       from pycsou.util.misc import peaks
       x = np.linspace(-2.5, 2.5, 25)
       xx, yy = np.meshgrid(x, x)
       z = peaks(xx, yy)
       directions = np.zeros(shape=(2, z.size))
       directions[0, : z.size // 2] = 1
       directions[1, z.size // 2 :] = 1
       dop = DirectionalDerivative(arg_shape=z.shape, which=1, directions=directions)
       out = dop(z.reshape(1, -1))
       dop2 = DirectionalDerivative(arg_shape=z.shape, which=2, directions=directions)
       out2 = dop2(z.reshape(1, -1))
       plt.figure()
       h = plt.pcolormesh(xx, yy, z, shading="auto")
       plt.quiver(x, x, directions[1].reshape(xx.shape), directions[0].reshape(xx.shape))
       plt.colorbar(h)
       plt.title("Signal and directions of derivatives")
       plt.figure()
       h = plt.pcolormesh(xx, yy, out.reshape(xx.shape), shading="auto")
       plt.colorbar(h)
       plt.title("First-order directional derivatives")
       plt.figure()
       h = plt.pcolormesh(xx, yy, out2.reshape(xx.shape), shading="auto")
       plt.colorbar(h)
       plt.title("Second-order directional derivative")

    See Also
    --------
    :py:func:`~pycsou.operator.linop.diff.Gradient`, :py:func:`~pycsou.operator.linop.diff.DirectionalGradient`
    """

    diff = Gradient(
        arg_shape=arg_shape,
        directions=None,
        diff_method=diff_method,
        mode=mode,
        gpu=gpu,
        dtype=dtype,
        sampling=sampling,
        parallel=parallel,
        **diff_kwargs,
    )

    xp = pycu.get_array_module(directions)
    directions = directions / xp.linalg.norm(directions, axis=0, keepdims=True)

    if directions.ndim == 1:
        dop = pycl.DiagonalOp(xp.tile(directions, arg_shape + (1,)).transpose().ravel())
    else:
        dop = pycl.DiagonalOp(directions.ravel())

    sop = pycl.Sum(arg_shape=(len(arg_shape),) + arg_shape, axis=0)
    op = sop * dop * diff

    if which == 2:
        op = -op.gram()  # -op.T * op

    op._name = "DirectionalDerivative"
    return _make_unravelable(op, arg_shape=arg_shape)


def DirectionalGradient(
    arg_shape: pyct.NDArrayShape,
    directions: list,
    diff_method: str = "fd",
    mode: ModeSpec = "constant",
    gpu: bool = False,
    dtype: typ.Optional[pyct.DType] = None,
    sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 1,
    parallel: bool = False,
    **diff_kwargs,
):
    r"""
    Directional gradient operator based on `Numba stencils <https://numba.pydata.org/numba-doc/latest/user/stencil.html>`_.

    Parameters
    ----------
    arg_shape: tuple
        Shape of the input array.
    directions: list
        List of directions, either constant (array of size :math:`(D,)`) or spatially-varying (array of size
        :math:`(D, N_0, \ldots, N_{D-1})`)
    diff_method: str ['gd', 'fd']
        Method used to approximate the derivative. Must be one of:

        * 'fd' (default): finite differences
        * 'gd': Gaussian derivative
    mode: str | list(str)
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: pyct.DType
        Working precision of the linear operator.
    sampling: float | tuple
        Sampling step (i.e., the distance between two consecutive elements of an array). Defaults to 1.
    diff_kwargs: dict
        Keyword arguments to parametrize partial derivatives (see
        :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.finite_difference` and
        :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.gaussian_derivative`)

    Returns
    -------
    op: :py:class:`~pycsou.abc.operator.LinOp`
            Directional gradient

    Notes
    -----
    The ``DirectionalGradient`` of a :math:`D`-dimensional signal :math:`\mathbf{f} \in
    \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}` stacks the directional derivatives of :math:`\mathbf{f}` along a list
    of :math:`m` directions :math:`\mathbf{v}_i` for :math:`1 \leq i \leq m`:

    .. math::

        \boldsymbol{\nabla}_{\mathbf{v}_1, \ldots ,\mathbf{v}_m} \mathbf{f} = \begin{bmatrix}
             \boldsymbol{\nabla}_{\mathbf{v}_1} \\
             \vdots\\
             \boldsymbol{\nabla}_{\mathbf{v}_m}\\
            \end{bmatrix} \mathbf{f} \in \mathbb{R}^{m \times N_0 \times \cdots \times N_{D-1}},

    where :math:`\boldsymbol{\nabla}_{\mathbf{v}_i}` is the first-order directional derivative along :math:`\mathbf{v}_i`
    implemented with :py:func:`~pycsou.operator.linop.diff.DirectionalDerivative`, with :math:`\mathbf{v}_i \in
    \mathbb{R}^D` or :math:`\mathbf{v}_i \in \mathbb{R}^{D \times N_0 \times \cdots \times N_{D-1}}`.

    Note that choosing :math:`m=D` and :math:`\mathbf{v}_i = \mathbf{e}_i \in \mathbb{R}^D` (the :math:`i`-th
    canonical basis vector) amounts to the :py:func:`~pycsou.operator.linop.diff.Gradient` operator.

    Example
    -------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.operator.linop.diff import DirectionalGradient
       from pycsou.util.misc import peaks
       x = np.linspace(-2.5, 2.5, 25)
       xx, yy = np.meshgrid(x, x)
       z = peaks(xx, yy)
       directions1 = np.zeros(shape=(2, z.size))
       directions1[0, :z.size // 2] = 1
       directions1[1, z.size // 2:] = 1
       directions2 = np.zeros(shape=(2, z.size))
       directions2[1, :z.size // 2] = -1
       directions2[0, z.size // 2:] = -1
       arg_shape = z.shape
       Dop = DirectionalGradient(arg_shape=arg_shape, directions=[directions1, directions2])
       out = Dop(z.reshape(1, -1))
       plt.figure()
       h = plt.pcolormesh(xx, yy, z, shading='auto')
       plt.quiver(x, x, directions1[1].reshape(arg_shape), directions1[0].reshape(xx.shape))
       plt.quiver(x, x, directions2[1].reshape(arg_shape), directions2[0].reshape(xx.shape), color='red')
       plt.colorbar(h)
       plt.title(r'Signal $\mathbf{f}$ and directions of derivatives')
       plt.figure()
       h = plt.pcolormesh(xx, yy, out[:, :z.size].reshape(arg_shape), shading='auto')
       plt.colorbar(h)
       plt.title(r'$\nabla_{\mathbf{v}_0} \mathbf{f}$')
       plt.figure()
       h = plt.pcolormesh(xx, yy, out[:, z.size:].reshape(arg_shape), shading='auto')
       plt.colorbar(h)
       plt.title(r'$\nabla_{\mathbf{v}_1} \mathbf{f}$')

    See Also
    --------
    :py:func:`~pycsou.operator.linop.diff.Gradient`, :py:func:`~pycsou.operator.linop.diff.DirectionalDerivative`
    """

    dir_deriv = []
    for direction in directions:
        dir_deriv.append(
            DirectionalDerivative(
                arg_shape=arg_shape,
                which=1,
                directions=direction,
                diff_method=diff_method,
                mode=mode,
                gpu=gpu,
                dtype=dtype,
                sampling=sampling,
                parallel=parallel,
                **diff_kwargs,
            )
        )
    op = pycb.vstack(dir_deriv)
    op._name = "DirectionalGradient"
    return _make_unravelable(op, arg_shape=arg_shape)


def DirectionalLaplacian(
    arg_shape: pyct.NDArrayShape,
    directions: list,
    weights: typ.Iterable = None,
    diff_method: str = "fd",
    mode: ModeSpec = "constant",
    gpu: bool = False,
    dtype: typ.Optional[pyct.DType] = None,
    sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 1,
    parallel: bool = False,
    **diff_kwargs,
):
    r"""
    Directional Laplacian operator based on `Numba stencils <https://numba.pydata.org/numba-doc/latest/user/stencil.html>`_.

    Sum of the second directional derivatives of a multi-dimensional array (at least two dimensions are required)
    along multiple ``directions`` for each entry of the array.

    Parameters
    ----------
    arg_shape: tuple
        Shape of the input array.
    directions: list
        List of directions, either constant (array of size :math:`(D,)`) or spatially-varying (array of size
        :math:`(D, N_0, \ldots, N_{D-1})`)
    weights: iterable (optional)
        List of optional positive weights with which each second directional derivative operator is multiplied.
    diff_method: str ['gd', 'fd']
        Method used to approximate the derivative. Must be one of:

        * 'fd' (default): finite differences
        * 'gd': Gaussian derivative
    mode: str | list(str)
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: pyct.DType
        Working precision of the linear operator.
    sampling: float | tuple
            Sampling step (i.e., the distance between two consecutive elements of an array). Defaults to 1.
    diff_kwargs: dict
        Keyword arguments to parametrize partial derivatives (see
        :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.finite_difference` and
        :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.gaussian_derivative`)

    Returns
    -------
    op: :py:class:`~pycsou.abc.operator.LinOp`
        Directional Laplacian

    Notes
    -----

    The ``DirectionalLaplacian`` of a :math:`D`-dimensional signal :math:`\mathbf{f} \in
    \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}` sums the second-order directional derivatives of :math:`\mathbf{f}`
    along a list of :math:`m` directions :math:`\mathbf{v}_i` for :math:`1 \leq i \leq m`:

    .. math::

        \boldsymbol{\Delta}_{\mathbf{v}_1, \ldots ,\mathbf{v}_m} \mathbf{f} = \sum_{i=1}^m
        \boldsymbol{\nabla}^2_{\mathbf{v}_i} \mathbf{f} \in \mathbb{R}^{N_0 \times \cdots \times N_{D-1}},

    where :math:`\boldsymbol{\nabla}^2_{\mathbf{v}_i}` is the second-order directional derivative along
    :math:`\mathbf{v}_i` implemented with :py:func:`~pycsou.operator.linop.diff.DirectionalDerivative`.

    Note that choosing :math:`m=D` and :math:`\mathbf{v}_i = \mathbf{e}_i \in \mathbb{R}^D` (the :math:`i`-th
    canonical basis vector) amounts to the :py:func:`~pycsou.operator.linop.diff.Laplacian` operator.

    Example
    -------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.operator.linop.diff import DirectionalLaplacian
       from pycsou.util.misc import peaks
       x = np.linspace(-2.5, 2.5, 25)
       xx, yy = np.meshgrid(x, x)
       z = peaks(xx, yy)
       directions1 = np.zeros(shape=(2, z.size))
       directions1[0, :z.size // 2] = 1
       directions1[1, z.size // 2:] = 1
       directions2 = np.zeros(shape=(2, z.size))
       directions2[1, :z.size // 2] = -1
       directions2[0, z.size // 2:] = -1
       arg_shape = z.shape
       Dop = DirectionalLaplacian(arg_shape=arg_shape, directions=[directions1, directions2])
       out = Dop(z.reshape(1, -1))
       plt.figure()
       h = plt.pcolormesh(xx, yy, z, shading='auto')
       plt.quiver(x, x, directions1[1].reshape(arg_shape), directions1[0].reshape(xx.shape))
       plt.quiver(x, x, directions2[1].reshape(arg_shape), directions2[0].reshape(xx.shape), color='red')
       plt.colorbar(h)
       plt.title('Signal and directions of derivatives')
       plt.figure()
       h = plt.pcolormesh(xx, yy, out.reshape(arg_shape), shading='auto')
       plt.colorbar(h)
       plt.title('Directional Laplacian')

    See Also
    --------
    :py:func:`~pycsou.operator.linop.diff.Laplacian`, :py:func:`~pycsou.operator.linop.diff.DirectionalDerivative`
    """

    if weights is None:
        weights = [1.0] * len(directions)
    else:
        if len(weights) != len(directions):
            raise ValueError("The number of weights and directions provided differ.")
    op = pycl.NullOp(shape=(np.prod(arg_shape), np.prod(arg_shape)))
    for weight, direction in zip(weights, directions):
        op += weight * DirectionalDerivative(
            arg_shape=arg_shape,
            which=2,
            directions=direction,
            diff_method=diff_method,
            mode=mode,
            gpu=gpu,
            dtype=dtype,
            sampling=sampling,
            parallel=parallel,
            **diff_kwargs,
        )
    op._name = "DirectionalLaplacian"
    return _make_unravelable(op, arg_shape=arg_shape)


def DirectionalHessian(
    arg_shape: pyct.NDArrayShape,
    directions: list,
    diff_method="gd",
    mode: ModeSpec = "constant",
    gpu: bool = False,
    dtype: typ.Optional[pyct.DType] = None,
    sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 1,
    parallel: bool = False,
    **diff_kwargs,
):
    r"""
    Directional Hessian operator based on `Numba stencils <https://numba.pydata.org/numba-doc/latest/user/stencil.html>`_.

    Parameters
    ----------
    arg_shape: tuple
        Shape of the input array.
    directions: list
        List of directions, either constant (array of size :math:`(D,)`) or spatially-varying (array of size
        :math:`(D, N_0, \ldots, N_{D-1})`)
    diff_method: str ['gd', 'fd']
        Method used to approximate the derivative. Must be one of:

        * 'fd' (default): finite differences
        * 'gd': Gaussian derivative
    mode: str | list(str)
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: pyct.DType
        Working precision of the linear operator.
    sampling: float | tuple
            Sampling step (i.e., the distance between two consecutive elements of an array). Defaults to 1.
    diff_kwargs: dict
        Keyword arguments to parametrize partial derivatives (see
        :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.finite_difference` and
        :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.gaussian_derivative`)

    Returns
    -------
    op: :py:class:`~pycsou.abc.operator.LinOp`
            Directional Hessian

    Notes
    -----

    The ``DirectionalHessian`` of a :math:`D`-dimensional signal :math:`\mathbf{f} \in
    \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}` stacks the second-order directional derivatives of :math:`\mathbf{f}`
    along a list of :math:`m` directions :math:`\mathbf{v}_i` for :math:`1 \leq i \leq m`:

    .. math::

        \mathbf{H}_{\mathbf{v}_1, \ldots ,\mathbf{v}_m} \mathbf{f} = \begin{bmatrix}
         \boldsymbol{\nabla}^2_{\mathbf{v}_0} & \cdots & \boldsymbol{\nabla}_{\mathbf{v}_0} \boldsymbol{\nabla}_{\mathbf{v}_{m-1}} \\
        \vdots & \ddots & \vdots \\
        \boldsymbol{\nabla}_{\mathbf{v}_{m-1}} \boldsymbol{\nabla}_{\mathbf{v}_0} & \cdots & \boldsymbol{\nabla}^2_{\mathbf{v}_{m-1}}
        \end{bmatrix} \mathbf{f},

    where :math:`\boldsymbol{\nabla}_{\mathbf{v}_i}` is the first-order directional derivative along
    :math:`\mathbf{v}_i` implemented with :py:func:`~pycsou.operator.linop.diff.DirectionalDerivative`.

    However, due to the symmetry of the Hessian, only the upper triangular part is computed in practice:

    .. math::

        \mathbf{H}_{\mathbf{v}_1, \ldots ,\mathbf{v}_m} \mathbf{f} = \begin{bmatrix}
        \boldsymbol{\nabla}^2_{\mathbf{v}_0}\\
        \boldsymbol{\nabla}_{\mathbf{v}_0} \boldsymbol{\nabla}_{\mathbf{v}_{1}} \\
        \vdots \\
        \boldsymbol{\nabla}^2_{\mathbf{v}_{m-1}}
        \end{bmatrix} \mathbf{f} \in \mathbb{R}^{\frac{m (m-1)}{2} \times N_0 \times \cdots \times N_{D-1}}

    Note that choosing :math:`m=D` and :math:`\mathbf{v}_i = \mathbf{e}_i \in \mathbb{R}^D` (the :math:`i`-th
    canonical basis vector) amounts to the :py:func:`~pycsou.operator.linop.diff.Hessian` operator.


    Example
    -------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.operator.linop.diff import DirectionalHessian
       from pycsou.util.misc import peaks
       x = np.linspace(-2.5, 2.5, 25)
       xx, yy = np.meshgrid(x, x)
       z = peaks(xx, yy)
       directions1 = np.zeros(shape=(2, z.size))
       directions1[0, :z.size // 2] = 1
       directions1[1, z.size // 2:] = 1
       directions2 = np.zeros(shape=(2, z.size))
       directions2[1, :z.size // 2] = -1
       directions2[0, z.size // 2:] = -1
       arg_shape = z.shape
       d_hess = DirectionalHessian(arg_shape=arg_shape, directions=[directions1, directions2])
       out = d_hess.unravel(d_hess(z.reshape(1, -1)))
       plt.figure()
       h = plt.pcolormesh(xx, yy, z, shading='auto')
       plt.quiver(x, x, directions1[1].reshape(arg_shape), directions1[0].reshape(xx.shape))
       plt.quiver(x, x, directions2[1].reshape(arg_shape), directions2[0].reshape(xx.shape), color='red')
       plt.colorbar(h)
       plt.title(r'Signal $\mathbf{f}$ and directions of derivatives')
       plt.figure()
       h = plt.pcolormesh(xx, yy, out[0, 0].reshape(arg_shape), shading='auto')
       plt.colorbar(h)
       plt.title(r'$\nabla^2_{\mathbf{v}_0} \mathbf{f}$')
       plt.figure()
       h = plt.pcolormesh(xx, yy, out[0, 1].reshape(arg_shape), shading='auto')
       plt.colorbar(h)
       plt.title(r'$\nabla_{\mathbf{v}_0} \nabla_{\mathbf{v}_{1}} \mathbf{f}$')
       plt.figure()
       h = plt.pcolormesh(xx, yy, out[0, 2].reshape(arg_shape), shading='auto')
       plt.colorbar(h)
       plt.title(r'$\nabla^2_{\mathbf{v}_1} \mathbf{f}$')

    See Also
    --------
    :py:func:`~pycsou.operator.linop.diff.Hessian`, :py:func:`~pycsou.operator.linop.diff.DirectionalDerivative`
    """

    dir_deriv = []
    for i, dir1 in enumerate(directions):
        for dir2 in directions[i:]:
            dir_deriv.append(
                -DirectionalDerivative(
                    arg_shape=arg_shape,
                    which=1,
                    directions=dir1,
                    diff_method=diff_method,
                    mode=mode,
                    gpu=gpu,
                    dtype=dtype,
                    sampling=sampling,
                    parallel=parallel,
                    **diff_kwargs,
                ).T
                * DirectionalDerivative(
                    arg_shape=arg_shape,
                    which=1,
                    directions=dir2,
                    diff_method=diff_method,
                    mode=mode,
                    gpu=gpu,
                    dtype=dtype,
                    sampling=sampling,
                    parallel=parallel,
                    **diff_kwargs,
                ).T
            )

    op = pycb.vstack(dir_deriv)
    op._name = "DirectionalHessian"
    return _make_unravelable(op, arg_shape=arg_shape)


class StructureTensor(pyco.DiffMap):
    r"""
    Structure tensor operator based on `Numba stencils <https://numba.pydata.org/numba-doc/latest/user/stencil.html>`_.
    
    Notes
    -----
    The Structure Tensor, also known as the second-order moment tensor or the inertia tensor, is a matrix derived from
    the gradient of a function. It describes the distribution of the gradient (i.e., its prominent directions) in a
    specified neighbourhood around a point, and the degree to which those directions are coherent.
    The structure tensor of a :math:`D`-dimensional signal
    :math:`\mathbf{f} \in \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}` can be written as:
    
    .. math::

        \mathbf{S}_\sigma \mathbf{f} = \mathbf{g}_{\sigma} * \nabla\mathbf{f} (\nabla\mathbf{f})^{\top} = \mathbf{g}_{\sigma} *
        \begin{bmatrix}
        \left( \dfrac{ \partial\mathbf{f} }{ \partial x_{0} } \right)^2 &  \dfrac{ \partial^{2}\mathbf{f} }{ \partial x_{0}\,\partial x_{1} } & \cdots & \dfrac{ \partial\mathbf{f} }{ \partial x_{0} } \dfrac{ \partial\mathbf{f} }{ \partial x_{D-1} } \\
        \dfrac{ \partial\mathbf{f} }{ \partial x_{1} } \dfrac{ \partial\mathbf{f} }{ \partial x_{0} } & \left( \dfrac{ \partial\mathbf{f} }{ \partial x_{1} }\right)^2 & \cdots & \dfrac{ \partial\mathbf{f} }{ \partial x_{1} } \dfrac{ \partial\mathbf{f} }{ \partial x_{D-1} } \\
        \vdots & \vdots & \ddots & \vdots \\
        \dfrac{ \partial\mathbf{f} }{ \partial x_{D-1} } \dfrac{ \partial\mathbf{f} }{ \partial x_{0} } & \dfrac{ \partial\mathbf{f} }{ \partial x_{D-1} } \dfrac{ \partial\mathbf{f} }{ \partial x_{1} } & \cdots & \left( \dfrac{ \partial\mathbf{f} }{ \partial x_{D-1}} \right)^2
        \end{bmatrix},

    where :math:`\mathbf{g}_{\sigma} \in \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}` is a discrete Gaussian filter
    with standard variation :math:`\sigma` with which a convolution is performed elementwise.

    However, due to the symmetry of the structure tensor, only the upper triangular part is computed in practice:

    .. math::

        \mathbf{H}_{\mathbf{v}_1, \ldots ,\mathbf{v}_m} \mathbf{f} = \mathbf{g}_{\sigma} * \begin{bmatrix}
        \left( \dfrac{ \partial\mathbf{f} }{ \partial x_{0} } \right)^2 \\
        \dfrac{ \partial^{2}\mathbf{f} }{ \partial x_{0}\,\partial x_{1} } \\
        \vdots \\
        \left( \dfrac{ \partial\mathbf{f} }{ \partial x_{D-1}} \right)^2
        \end{bmatrix} \mathbf{f} \in \mathbb{R}^{\frac{D (D-1)}{2} \times N_0 \times \cdots \times N_{D-1}}

    Example
    -------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.operator.linop.diff import StructureTensor
       from pycsou.util.misc import peaks

       # Define input image
       n = 1000
       x = np.linspace(-3, 3, n)
       xx, yy = np.meshgrid(x, x)
       image = peaks(xx, yy)
       nsamples = 2
       arg_shape = image.shape  # (1000, 1000)
       images = np.tile(image, (nsamples, 1, 1)).reshape(nsamples, -1)
       print(images.shape)  # (2, 1000000)
       # Instantiate structure tensor operator
       structure_tensor = StructureTensor(arg_shape=arg_shape)

       outputs = structure_tensor(images)
       print(outputs.shape)  # (2, 3000000)
       # Plot
       outputs = structure_tensor.unravel(outputs)
       print(outputs.shape)  # (2, 3, 1000, 1000)
       plt.figure()
       plt.imshow(images[0].reshape(arg_shape))
       plt.colorbar()
       plt.title("Image")
       plt.axis("off")

       plt.figure()
       plt.imshow(outputs[0][0].reshape(arg_shape))
       plt.colorbar()
       plt.title(r"$\hat{S}_{xx}$")
       plt.axis("off")

       plt.figure()
       plt.imshow(outputs[0][1].reshape(arg_shape))
       plt.colorbar()
       plt.title(r"$\hat{S}_{xy}$")
       plt.axis("off")
       
       plt.figure()
       plt.imshow(outputs[0][2].reshape(arg_shape))
       plt.colorbar()
       plt.title(r"$\hat{S}_{yy}$")
       plt.axis("off")
       
    See Also
    --------
    :py:class:`~pycsou.operator.linop.diff.PartialDerivative`, :py:class:`~pycsou.operator.linop.diff.Gradient`,
    :py:class:`~pycsou.operator.linop.diff.Hessian`.
    """

    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        diff_method="fd",
        smooth_sigma: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 1.0,
        smooth_truncate: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 3.0,
        mode: ModeSpec = "constant",
        gpu: bool = False,
        dtype: typ.Optional[pyct.DType] = None,
        sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]] = 1,
        parallel: bool = False,
        **diff_kwargs,
    ):
        self.arg_shape = arg_shape
        size = int(np.prod(arg_shape))
        ndim = len(arg_shape)
        ntriu = (ndim * (ndim + 1)) // 2
        super().__init__(shape=(ntriu * size, size))
        self.directions = tuple(
            list(_) for _ in itertools.combinations_with_replacement(np.arange(len(arg_shape)).astype(int), 2)
        )

        order, sampling, smooth_sigma, smooth_truncate, axis = _sanitize_init_kwargs(
            order=(1,) * len(arg_shape),
            param1=smooth_sigma,
            param1_name="smooth_sigma",
            param2=smooth_truncate,
            param2_name="smooth_truncate",
            arg_shape=arg_shape,
            sampling=sampling,
        )

        if diff_method == "fd":
            diff_kwargs.update({"diff_type": diff_kwargs.pop("diff_type", "central")})
            self.grad = Gradient.finite_difference(
                arg_shape=arg_shape,
                mode=mode,
                gpu=gpu,
                dtype=dtype,
                sampling=sampling,
                parallel=parallel,
                **diff_kwargs,
            )
        elif diff_method == "gd":
            self.grad = Gradient.gaussian_derivative(
                arg_shape=arg_shape,
                mode=mode,
                gpu=gpu,
                dtype=dtype,
                sampling=sampling,
                parallel=parallel,
                **diff_kwargs,
            )

        if smooth_sigma:
            kernel = [
                np.array([1]),
            ] * len(arg_shape)
            center = np.zeros(len(arg_shape), dtype=int)
            for i in range(len(arg_shape)):
                radius = int(smooth_truncate[i] * float(smooth_sigma[i]) + 0.5)
                kernel[axis[i]] = np.flip(scif._gaussian_kernel1d(smooth_sigma[i], 0, radius))
                center[axis[i]] = radius

            self.smooth = _BaseDifferential(
                kernel=kernel, center=center, arg_shape=arg_shape, mode=mode, gpu=gpu, dtype=dtype
            )
        else:
            self.smooth = pycl.IdentityOp(dim=np.prod(arg_shape).item())

    def unravel(self, arr):
        return arr.reshape(-1, *arr.shape[:-1], *self.arg_shape).swapaxes(0, 1)

    def ravel(self, arr):
        return arr.swapaxes(0, 1).reshape(*arr.shape[: -1 - len(self.arg_shape)], -1)

    def apply(self, arr):
        xp = pycu.get_array_module(arr)
        sh = arr.shape[:-1]
        grad = self.grad.unravel(self.grad(arr))
        return xp.concatenate(
            [self.smooth((grad[:, i] * grad[:, j]).reshape(*sh, -1)) for i, j in self.directions]
        ).reshape(arr.shape[0], -1)
