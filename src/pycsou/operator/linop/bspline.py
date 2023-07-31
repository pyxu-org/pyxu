r"""
This B-spline module provides the basic tools to solve continuous-domain inverse problems by parametrizing the signal of
interest in the B-spline basis and optimizing over its B-spline coefficients. This continuous-domain approach is
particularly relevant when the forward model does not admit a straightforward discretization, such as nonuniform
sampling (see example below, [FuncSphere]_, and [BSplineTV]_).

B-splines are basis functions for the space of splines (piecewise polynomials) with predefined knots, which means that
*any* spline can be represented as a linear combination of B-splines. The other key feature of B-splines is that they
are the splines with the shortest support, which is paramount for numerical applications (see [BSplineDeBoor]_). In
particular, splines can be evaluated at any point using only a very small subset of their B-spline coefficients, and
B-splines lead to well-conditioned optimization problems (see [BSplineTV]_).

Below is an example of an image-reconstruction problem parametrized in the B-spline basis. The forward model is a
sampling operator on a grid, but where the grid is not uniform. This can for example occur in imaging applications where
it is difficult to impose a uniform sampling pattern due to hardware precision limitations, such as scanning
transmission X-ray microscopy (STXM) imaging. To demonstrate the necessity of the B-spline-based continuous model, we
compare it with a purely discrete model where the non-uniformity of the sampling pattern is not
accounted for, which leads to much worse reconstruction results. In both cases, we use total-variation (TV)
regularization :math:`||\nabla \cdot||_{2, 1}`, where :math:`\nabla` is the gradient operator and
:math:`||\cdot ||_{2, 1}` is the mixed :math:`L_{2,1}` norm (isotropic TV). In the B-spline model, the continuous
gradient is evaluated exactly on the pixel grid, whereas it can only be approximated using finite differences in the
discrete model.

.. code-block:: python3

    import matplotlib.pyplot as plt
    import scipy.interpolate as sci
    import numpy as np
    import pycsou.operator.linop.bspline as bsp
    from pycsou.util.misc import star_like_sample
    import pycsou.operator as pycop
    import pycsou.opt.solver as pycsol
    import pycsou.opt.stop as pycstop
    from pycsou.operator.func.norm import L21Norm

    # Generate 2D image
    N = 128
    grid_rec = np.arange(N)
    image = star_like_sample(N, 8, 20, 3, 0.7)

    # Interpolate image with B-splines (continuous-domain ground truth
    degree = 3  # Cubic B-splines
    spl = sci.RectBivariateSpline(grid_rec, grid_rec, image, kx=degree, ky=degree, s=0)
    knots, _ = spl.get_knots()
    c_im = spl.get_coeffs()  # Image B-spline coefficients

    # Sampling locations
    rng = np.random.default_rng(seed=0)  # For reproducibility
    sigma_dev = (grid_rec[1] - grid_rec[0]) / 2  # Std of deviation from uniform grid
    samp_dev_x = rng.standard_normal(len(grid_rec)) * sigma_dev  # Deviation from uniform grid in x
    samp_dev_y = rng.standard_normal(len(grid_rec)) * sigma_dev  # Deviation from uniform grid in y
    samp_dev_x[0], samp_dev_x[-1] = abs(samp_dev_x[0]), - abs(samp_dev_x[-1])  # Ensure that sampling is within the ROI
    samp_dev_y[0], samp_dev_y[-1] = abs(samp_dev_y[0]), - abs(samp_dev_y[-1])  # Ensure that sampling is within the ROI
    grid_samp_x, grid_samp_y = np.sort(grid_rec + samp_dev_x), np.sort(grid_rec + samp_dev_y)  # Sampling grid

    # Nonuniform sampling operator
    sampling_op = bsp.BSplineSampling([grid_samp_x, grid_samp_y], knots, degree, ndim=2)
    # Measured data (nonuniform samples of the ground truth)
    sigma_noise = 5*1e-2
    y = sampling_op(c_im) + rng.standard_normal(sampling_op.codim) * sigma_noise

    # Continuous-domain optimization problem formulation in B-spline basis

    # Data fidelity term
    f = 1 / 2 * pycop.SquaredL2Norm(dim=sampling_op.codim).asloss(y) * sampling_op
    f.diff_lipschitz(tight=True, tol=1e-1)

    # Regularization term to enforce periodicity of reconstruction
    g = bsp.BSplinePeriodicIndFunc(knots, degree, ndim=2)

    # TV regularization term
    K = bsp.BSplineGradientGrid(eval_grid=grid_rec, knots=knots, degrees=degree, ndim=2)
    K.lipschitz()
    h = L21Norm(arg_shape=(2, N, N))
    lamb = 2*1e-2  # Regularization parameter

    # Stopping criterion
    tol = 1e-4
    stop_crit = pycstop.RelError(eps=tol, var="x") & pycstop.RelError(eps=tol, var="z")

    # Solve optimization problem
    solver = pycsol.CV(f=f, g=g, h=lamb * h, K=K, verbosity=1000)
    solver.fit(x0=y, stop_crit=stop_crit)  # Solve problem
    c_opt = solver.solution()  # Coefficients of reconstructed spline
    rec_op = bsp.BSplineSampling(grid_rec, knots, degree, ndim=2)  # Spline-coefficients-to-image operator
    image_rec = rec_op(c_opt).reshape(image.shape)  # Reconstructed image
    snr = 20 * np.log10(np.linalg.norm(image) / np.linalg.norm(image - image_rec))

    # Discrete optimization problem formulation (without accounting for irregular sampling)
    # All operators are discretized versions of the continuous ones
    f_d = 1 / 2 * pycop.SquaredL2Norm(dim=sampling_op.codim).asloss(y)  # Forward model is identity
    K_d = pycop.Gradient(image.shape)
    K_d.lipschitz()
    h_d = L21Norm(arg_shape=(2, N, N))

    # Solve optimization problem
    solver_d = pycsol.CV(f=f_d, h=lamb * h_d, K=K_d, verbosity=1000)
    solver_d.fit(x0=y, stop_crit=stop_crit)  # Solve problem
    image_rec_d = solver_d.solution().reshape(image.shape)  # Reconstructed image
    snr_d = 20 * np.log10(np.linalg.norm(image) / np.linalg.norm(image - image_rec_d))

    # Plots
    vmin, vmax = 0, 1

    plt.figure()
    plt.imshow(image, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.colorbar()
    plt.title('Original image')

    plt.figure()
    plt.imshow(y.reshape(image.shape), vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.colorbar()
    plt.title('Noisy nonuniform data')

    plt.figure()
    plt.imshow(image_rec, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.colorbar()
    plt.title(f'Reconstructed image with continuous model (SNR = {snr:.2f} dB)')

    plt.figure()
    plt.imshow(image_rec_d, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.colorbar()
    plt.title(f'Reconstructed image with discrete model (SNR = {snr_d:.2f} dB)')

Below is the same example with Hessian-Schatten regularization :math:`||\mathbf{H} \cdot||_{1, *} =
\sum_{i=1}^N ||\mathbf{H}_i \cdot ||_{*}`, where :math:`\mathbf{H} = (\mathbf{H}_0, \ldots, \mathbf{H}_{N-1})` is the
Hessian operator where :math:`\mathbf{H}_i` is the Hessian at the :math:`i`-th pixel and :math:`||\cdot ||_*` is the
1-Schatten norm for matrices (or nuclear norm). This type of regularization was first introduced in [HessianSchatten]_.
In the B-spline model, the continuous Hessian is evaluated exactly on the pixel grid, whereas it can only be
approximated using finite differences in the discrete model.

Notes:

* This example is meant to demonstrate how to implement the Hessian-Schatten norm with Pyxu (the Schatten norm could be
  generalized to the :math:`p \neq 1` cases with minimal effort). However, this is not be the best test case for
  Hessian-Schatten regularization, as the best SNR is achieved with very low regularization. Hessian-Schatten
  regularization tends to promote piewise-linear reconstructions with few linear regions. It also works well for images
  with filament-like structures, whose Hessian have a sparse SVD decomposition (i.e., a single non-zero singular value
  in 2D images).
* The Condat-Vu algorithm converges very slowly in these examples; a stricter stopping criterion leads to
  significantly better (and slower) reconstructions.

.. code-block:: python3

    import matplotlib.pyplot as plt
    import scipy.interpolate as sci
    import numpy as np
    import pycsou.operator.linop.bspline as bsp
    from pycsou.util.misc import star_like_sample
    import pycsou.operator as pycop
    import pycsou.opt.solver as pycsol
    import pycsou.opt.stop as pycstop

    class NuclearNorm(pyca.ProxFunc):

        def __init__(self, arg_shape):
            ndim = len(arg_shape)
            dim = int(np.prod(arg_shape) * ndim * (ndim + 1) / 2)
            super().__init__(shape=(1, dim))
            self.ndim = ndim
            self._arg_shape = arg_shape

        def apply(self, arr):
            return np.linalg.norm(self._reshape_to_mat(arr), ord="nuc", axis=(-2, -1))

        def prox(self, arr, tau):
            u, s, vh = np.linalg.svd(a=self._reshape_to_mat(arr), full_matrices=False, hermitian=True)
            s_prox = np.fmax(0, np.fabs(s) - tau) * np.sign(s)  # Soft thresholding of singular values
            out = (u * s_prox[..., None, :] @ vh).reshape((*arr.shape[:-1], np.prod(self._arg_shape), self.ndim, self.ndim))
            return np.swapaxes(out[..., *np.triu_indices(self.ndim)], -1, -2).ravel()

        def _reshape_to_mat(self, arr):
            arr_mat = np.zeros((*arr.shape[:-1], np.prod(self._arg_shape), self.ndim, self.ndim))
            arr = arr.reshape((*arr.shape[:-1], -1, np.prod(self._arg_shape)))
            idx_i, idx_j = np.triu_indices(self.ndim)
            for idx in range(int(self.ndim * (self.ndim - 1) / 2)):
                i, j = idx_i[idx], idx_j[idx]
                arr_mat[..., i, j] = arr[..., idx, :]
                arr_mat[..., j, i] = arr[..., idx, :]
            return arr_mat

    # Generate 2D image
    N = 128
    grid_rec = np.arange(N)
    image = star_like_sample(N, 8, 20, 3, 0.7)

    # Interpolate image with B-splines (continuous-domain ground truth
    degree = 3  # Cubic B-splines
    spl = sci.RectBivariateSpline(grid_rec, grid_rec, image, kx=degree, ky=degree, s=0)
    knots, _ = spl.get_knots()
    c_im = spl.get_coeffs()  # Image B-spline coefficients

    # Sampling locations
    rng = np.random.default_rng(seed=0)  # For reproducibility
    sigma_dev = (grid_rec[1] - grid_rec[0]) / 2  # Std of deviation from uniform grid
    samp_dev_x = rng.standard_normal(len(grid_rec)) * sigma_dev  # Deviation from uniform grid in x
    samp_dev_y = rng.standard_normal(len(grid_rec)) * sigma_dev  # Deviation from uniform grid in y
    samp_dev_x[0], samp_dev_x[-1] = abs(samp_dev_x[0]), - abs(samp_dev_x[-1])  # Ensure that sampling is within the ROI
    samp_dev_y[0], samp_dev_y[-1] = abs(samp_dev_y[0]), - abs(samp_dev_y[-1])  # Ensure that sampling is within the ROI
    grid_samp_x, grid_samp_y = np.sort(grid_rec + samp_dev_x), np.sort(grid_rec + samp_dev_y)  # Sampling grid

    # Nonuniform sampling operator
    sampling_op = bsp.BSplineSampling([grid_samp_x, grid_samp_y], knots, degree, ndim=2)
    # Measured data (nonuniform samples of the ground truth)
    sigma_noise = 5*1e-2
    y = sampling_op(c_im) + rng.standard_normal(sampling_op.codim) * sigma_noise

    # Continuous-domain optimization problem formulation in B-spline basis

    # Data fidelity term
    f = 1 / 2 * pycop.SquaredL2Norm(dim=sampling_op.codim).asloss(y) * sampling_op
    f.diff_lipschitz(tight=True, tol=1e-1)

    # Regularization term to enforce periodicity of reconstruction
    g = bsp.BSplinePeriodicIndFunc(knots, degree, ndim=2)

    # Hessian-Schatten regularization term
    K = bsp.BSplineHessianGrid(eval_grid=grid_rec, knots=knots, degrees=degree, ndim=2)
    K.lipschitz()
    h = NuclearNorm(arg_shape=(N, N))
    lamb = 1e-8  # Regularization parameter

    # Stopping criterion
    tol = 1e-4
    stop_crit = pycstop.RelError(eps=tol, var="x") & pycstop.RelError(eps=tol, var="z")

    # Solve optimization problem
    solver = pycsol.CV(f=f, g=g, h=lamb * h, K=K, verbosity=1000)
    solver.fit(x0=y, stop_crit=stop_crit)  # Solve problem
    c_opt = solver.solution()  # Coefficients of reconstructed spline
    rec_op = bsp.BSplineSampling(grid_rec, knots, degree, ndim=2)  # Spline-coefficients-to-image operator
    image_rec = rec_op(c_opt).reshape(image.shape)  # Reconstructed image
    snr = 20 * np.log10(np.linalg.norm(image) / np.linalg.norm(image - image_rec))

    # Discrete optimization problem formulation (without accounting for irregular sampling)
    # All operators are discretized versions of the continuous ones
    f_d = 1 / 2 * pycop.SquaredL2Norm(dim=sampling_op.codim).asloss(y)  # Forward model is identity
    K_d = pycop.Hessian(image.shape)
    K_d.lipschitz()
    h_d = NuclearNorm(arg_shape=(N, N))

    # Solve optimization problem
    solver_d = pycsol.CV(f=f_d, h=lamb * h_d, K=K_d, verbosity=1000)
    solver_d.fit(x0=y, stop_crit=stop_crit)  # Solve problem
    image_rec_d = solver_d.solution().reshape(image.shape)  # Reconstructed image
    snr_d = 20 * np.log10(np.linalg.norm(image) / np.linalg.norm(image - image_rec_d))

    # Plots
    vmin, vmax = 0, 1

    plt.figure()
    plt.imshow(image, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.colorbar()
    plt.title('Original image')

    plt.figure()
    plt.imshow(y.reshape(image.shape), vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.colorbar()
    plt.title('Noisy nonuniform data')

    plt.figure()
    plt.imshow(image_rec, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.colorbar()
    plt.title(f'Reconstructed image with continuous model (SNR = {snr:.2f} dB)')

    plt.figure()
    plt.imshow(image_rec_d, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.colorbar()
    plt.title(f'Reconstructed image with discrete model (SNR = {snr_d:.2f} dB)')

"""

import typing as typ

import dask.array as da
import numpy as np
import scipy.interpolate as sci
import scipy.sparse as sp

import pycsou.abc as pyca
import pycsou.operator.blocks as pycb
import pycsou.operator.func.norm as pycn
import pycsou.operator.linop as pycl
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

__all__ = [
    "BSplineSampling",
    "BSplinePeriodicIndFunc",
    "uniform_knots",
    "BSplineGradientGrid",
    "BSplineLaplacianGrid",
    "BSplineHessianGrid",
    "BSplineInnos1D",
]


def _array_module_to_sparse(array_module: pycd.NDArrayInfo) -> pycd.SparseArrayInfo:
    r"""
    Returns the suitable sparse array module that is compatible with the input array module.
    """
    if array_module == pycd.NDArrayInfo.CUPY:
        sparse_module = pycd.SparseArrayInfo.CUPY_SPARSE
    elif array_module == pycd.NDArrayInfo.DASK:
        sparse_module = pycd.SparseArrayInfo.PYDATA_SPARSE
    else:
        sparse_module = pycd.SparseArrayInfo.SCIPY_SPARSE
    return sparse_module


def _to_sparse_backend(
    spmat: pyct.SparseArray, sparse_module: pycd.SparseArrayInfo = pycd.SparseArrayInfo.SCIPY_SPARSE
):
    r"""
    Returns sparse array module compatible with the current array module.
    """
    if sparse_module == pycd.SparseArrayInfo.CUPY_SPARSE:
        spmat = pycd.SparseArrayInfo.CUPY_SPARSE.module().csr_matrix(spmat)
    elif sparse_module == pycd.SparseArrayInfo.PYDATA_SPARSE:
        spmat = pycd.SparseArrayInfo.PYDATA_SPARSE.module().GCXS(spmat)
    return spmat


def _convert_to_list(*args, ndim: pyct.Integer = 1):
    r"""
    Converts a collection of objects that are either lists of length ndim or single objects into a tuple of lists.
    Single objects are converted to lists of ndim instances of this object. ndim is either provided or inferred from
    the length of the provided lists.
    """
    out = ()

    # Check that all lists have the same length and set ndim
    is_list = False
    for x in args:
        if isinstance(x, list):
            if is_list or ndim > 1:
                assert ndim == len(x)
            is_list = True
            ndim = len(x)

    # Convert single objects to lists
    for x in args:
        if isinstance(x, list):
            out = (*out, x)
        else:
            out = (*out, [x for _ in range(ndim)])
    return *out, ndim


def _BSplineDerivative(
    knots: typ.Union[np.ndarray, list[np.ndarray]],
    degrees: typ.Union[pyct.Integer, list[pyct.Integer]],
    deriv_orders: typ.Union[pyct.Integer, list[pyct.Integer]],
    ndim: pyct.Integer = 1,
    array_module: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY,
) -> pyca.LinOp:
    r"""
    Returns a :py:class:`~pycsou.abc.operator.LinOp` that takes as input coefficients of a tensor-product B-spline and
    returns the coefficients of its derivative.

    Parameters
    ----------
    knots: ndarray or list[ndarray]
        Knots of the spline (list if different for each dimension).
    degrees: int or list[int]
        Degree of the spline (list if different for each dimension).
    deriv_orders: int or list[int]
        Order of differentiation (list if different for each dimension).
    ndim: int, optional
        Number of dimensions, `i.e.`, of variables of the spline (can be inferred from previous parameters if a list
        is provided).
    array_module: :py:class:`~pycsou.util.deps.NDArrayInfo`, optional
        Supported input ndarray API for the output operator (default is Numpy).
    """
    knots, degrees, deriv_orders, ndim = _convert_to_list(knots, degrees, deriv_orders, ndim=ndim)

    sparse_module = _array_module_to_sparse(array_module)
    for i in range(ndim):
        if deriv_orders[i] > 0:
            bspline = sci.BSpline(
                t=knots[i], c=np.eye(knots[i].size - degrees[i] - 1, dtype=pycrt.getPrecision().value), k=degrees[i]
            )
            op = sp.csr_array(
                bspline.derivative(nu=deriv_orders[i]).c[: bspline.t.size - bspline.k - 1 - deriv_orders[i]]
            )
            # For some reason, scipy doesn't return the correct number of coefficients of the derivative.
            op = _to_sparse_backend(op, sparse_module)
            if array_module == pycd.NDArrayInfo.DASK:
                op = da.from_array(op)
            op = pyca.LinOp.from_array(op.astype(pycrt.getPrecision().value))
        else:
            op = pycl.IdentityOp(dim=knots[i].size - degrees[i] - 1)
        if i == 0:
            op_ndim = op
        else:
            op_ndim = pycl.kron(op_ndim, op)
    return op_ndim


# @pycrt.enforce_precision('samples', o=False) #
def BSplineSampling(
    eval_grid: typ.Union[np.ndarray, list[np.ndarray]],
    knots: typ.Union[np.ndarray, list[np.ndarray]],
    degrees: typ.Union[pyct.Integer, list[pyct.Integer]],
    deriv_orders: typ.Union[pyct.Integer, list[pyct.Integer]] = 0,
    ndim: pyct.Integer = 1,
    array_module: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY,
) -> pyca.LinOp:
    r"""
    Returns a linear operator that takes as input :math:`D`-variate tensor-product B-spline coefficients and that
    outputs the samples of the derivative of that spline on a predefined grid.

    Parameters
    ----------
    eval_grid: ndarray or list[ndarray]
        Evaluation grid (list if different for each dimension) with shape :math:`(m_d,)`.
    knots: ndarray or list[ndarray]
        Knots of the spline (list if different for each dimension) with shape :math:`(n_d + k_d + 1,)`.
    degrees: int or list[int]
        Degree of the spline (list if different for each dimension).
    deriv_orders: int or list[int], optional
        Order of differentiation (list if different for each dimension). Default is 0.
    ndim: int, optional
        Number of variables :math:`D` of the spline (can be inferred from previous parameters if a list is provided).
    array_module: :py:class:`~pycsou.util.deps.NDArrayInfo`, optional
        Supported input ndarray API for the output operator (default is Numpy).

    Returns
    -------
    op: :py:class:`~pycsou.abc.operator.LinOp`, shape :math:`\left(\prod_{d=0}^{D-1} m_d, \prod_{d=0}^{D-1} n_d \right)`.
        Sampling operator.

    Notes
    -----
    A :math:`D`-variate tensor-product spline in the B-spline basis is given by

    .. math::

        S(\mathbf{x}) = \sum_{j_0=0}^{n_0-1} \ldots \sum_{j_{D-1}=0}^{n_{D-1}-1} c_{j_0, \ldots, j_{D-1}}
        \prod_{d=0}^{D-1} B_{j_d, k_d; \mathbf{t}[d]}(x_d)

    for any :math:`\mathbf{x} \in \mathbb{R}^D`, where:

        * :math:`\mathbf{c} \in \mathbb{R}^{n_0 \times \ldots \times n_{D-1}}` is the :math:`D`-dimensional array of
          B-spline coefficients;

        * :math:`\mathbf{t}[d]` = ``knots[d]`` are the knots of :math:`S` for the :math:`d`-th dimension;

        * :math:`k_d` = ``degrees[d]`` is the degree of :math:`S` for the :math:`d`-th dimension.

    The univariate B-spline basis functions
    :math:`B_{j_d, k_d; \mathbf{t}[d]}` with :math:`0 \leq d \leq D-1` are defined as in the documentation of Scipy's
    :py:class:`~scipy.interpolate.BSpline` class.

    The output operator ``op`` takes a flattened ndarray of B-spline coefficients :math:`\mathbf{c}` with shape
    :math:`\left(\prod_{d=0}^{D-1} n_d, \right)` as input, and outputs the flattened ndarray

    .. math::

        \Big( \partial_{x_0}^{\nu_0} \ldots \partial_{x_{D-1}}^{\nu_{D-1}} S \big((\mathbf{x}[0])_{j_0}, \ldots,
        (\mathbf{x}[D-1])_{j_{D-1}} \big) \Big)_{0 \leq j_0 \leq m_0, \ \ldots, \ 0 \leq j_{D-1} \leq m_{D-1}}

    with shape :math:`\left(\prod_{d=0}^{D-1} m_d, \right)`, where:

        * :math:`\nu_d` = ``deriv_orders[d]`` is the order of differentiation with respect to the :math:`d`-th variable;

        * :math:`\mathbf{x}[d]` = ``eval_grid[d]``:math:`\in \mathbb{R}^{m_d}` is the evaluation grid for the
          :math:`d`-th dimension.

    The implementation of the output operator ``op`` leverages the following sparse matrix APIs:

        * Scipy :py:class:`~scipy.sparse.spmatrix` for Numpy input ndarrays;

        * Sparse :py:class:`~sparse.SparseArray` for Dask input ndarrays;

        * Cupy :py:class:`~cupyx.scipy.sparse.spmatrix` for Cupy input ndarrays.

    Examples
    --------

    .. plot::

        import matplotlib.pyplot as plt
        import scipy.interpolate as sci
        import numpy as np
        from pycsou.operator.linop.bspline import BSplineSampling
        from pycsou.util.misc import peaks

        # Generate coarse-scale 2D image
        grid_coarse = np.linspace(-2.5, 2.5, 50)
        xx, yy = np.meshgrid(grid_coarse, grid_coarse)
        image_coarse = peaks(xx, yy)

        # Interpolate image with B-splines
        degree = 3  # Cubic B-splines
        spl = sci.RectBivariateSpline(grid_coarse, grid_coarse, image_coarse, kx=degree, ky=degree, s=0)
        knots, _ = spl.get_knots()
        shape_coeffs = (len(knots) - degree - 1,) * 2
        c = spl.get_coeffs().reshape(shape_coeffs)  # 2D array of tensor-product B-spline coefficients

        # Assert that sampling operator is consistent with Scipy
        im_coarse_op = BSplineSampling(grid_coarse, knots, degree, deriv_orders=0, ndim=2)
        assert np.allclose(spl(grid_coarse, grid_coarse, grid=True).ravel(), im_coarse_op(c.ravel()))

        # Resample B-spline on finer grid
        grid_fine = np.linspace(grid_coarse[0], grid_coarse[-1], 100)
        shape_fine = (len(grid_fine),) * 2  # Shape of the fine-scale image
        im_fine_op = BSplineSampling(grid_fine, knots, degree, deriv_orders=0, ndim=2)
        image_fine = im_fine_op(c.ravel()).reshape(shape_fine)

        # Partial derivative of the image along first axis
        im_dx_op = BSplineSampling(grid_fine, knots, degree, deriv_orders=[1, 0], ndim=2)
        image_dx = im_dx_op(c.ravel()).reshape(shape_fine)

        # Plots

        plt.figure()
        plt.imshow(image_coarse)
        plt.axis('off')
        plt.colorbar()
        plt.title(r'Coarse-scale image $f(x,y)$')

        plt.figure()
        plt.imshow(image_fine)
        plt.axis('off')
        plt.colorbar()
        plt.title(r'Fine-scale image $f(x,y)$')

        plt.figure()
        plt.imshow(image_dx)
        plt.axis('off')
        plt.colorbar()
        plt.title(r'$\partial_{x_0} f(x,y)$')

    """
    eval_grid, knots, degrees, deriv_orders, ndim = _convert_to_list(eval_grid, knots, degrees, deriv_orders, ndim=ndim)
    sparse_module = _array_module_to_sparse(array_module)

    for i in range(ndim):
        t = knots[i][deriv_orders[i] : -deriv_orders[i] or None]  # Knots of derivative for i-th dimension
        k = degrees[i] - deriv_orders[i]  # Degree of derivative
        spmat = _to_sparse_backend(sci.BSpline.design_matrix(x=eval_grid[i], t=t, k=k), sparse_module)
        if array_module == pycd.NDArrayInfo.DASK:
            spmat = da.from_array(spmat)
        op = pyca.LinOp.from_array(spmat.astype(pycrt.getPrecision().value))
        if i == 0:
            op_ndim = op
        else:
            op_ndim = pycl.kron(op_ndim, op)
    return op_ndim * _BSplineDerivative(knots, degrees, deriv_orders, ndim, array_module)


def _precond_nonuniform_knots(
    knots: typ.Union[np.ndarray, list[np.ndarray]],
    degrees: typ.Union[pyct.Integer, list[pyct.Integer]],
    ndim: pyct.Integer = 1,
    array_module: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY,
) -> tuple[pyca.LinOp, pyca.LinOp, pyct.NDArray]:
    r"""
    Preconditioning operator for inverse problems formulated over B-splines with nonuniform knots.

    Parameters
    ----------
    knots: ndarray or list[ndarray]
        Knots of the spline (list if different for each dimension) with shape :math:`(n_d + k_d + 1,)`.
    degrees: int or list[int]
        Degree of the spline (list if different for each dimension).
    ndim: int, optional
        Number of variables :math:`D` of the spline (can be inferred from previous parameters if a list is provided).
    array_module: :py:class:`~pycsou.util.deps.NDArrayInfo`, optional
        Supported input ndarray API for the output operators (default is Numpy).

    Returns
    -------
    precond: :py:class:`~pycsou.abc.operator.LinOp`, shape :math:`\left(\prod_{d=0}^{D-1} n_d, \prod_{d=0}^{D-1} n_d \right)`.
        Preconditioning operator.

    precond_inv :py:class:`~pycsou.abc.operator.LinOp`, shape :math:`\left(\prod_{d=0}^{D-1} n_d, \prod_{d=0}^{D-1} n_d \right)`.
        Inverse preconditioning operator.

    weights: ndarray, shape :math:`\left(\prod_{d=0}^{D-1} n_d, \right)`
        Preconditioning weights.

    Notes
    -----
    The preconditioning operator for a univariate nonuniform spline of degree :math:`k` with knots :math:`\mathbf{t}`
    and :math:`n` B-spline coefficients is the diagonal operator :math:`\mathbf{P}` with weights

    .. math::
        :label: eq:precond

        \mathrm{diag}(\mathbf{P})_i = \frac{k}{\mathbf{t}_{k+i} - \mathbf{t}_{i}}, \qquad 0 \leq i \leq n - 1

    (see :py:func:`~pycsou.operator.linop.bspline.BSplineSampling` for notations). For a multivariate spline,
    :math:`\mathbf{P}` is the Kronecker product of the preconditioning operators associated to each of its dimensions,
    `i.e.`,

    .. math::
        \mathbf{P} = \mathbf{P}^0 \otimes \cdots \otimes \mathbf{P}^{D-1},

    where :math:`\mathbf{P}^d` is the operator :math:numref:`eq:precond` with degree :math:`k=k_d`, knots
    :math:`\mathbf{t} = \mathbf{t}[d]`, and dimension :math:`n=n_d`.

    This preconditioning operator is useful for inverse problems formulated over B-spline coefficients
    :math:`\mathbf{c}`; replacing :math:`\mathbf{c}` with :math:`\mathbf{P}\tilde{\mathbf{c}}` in the
    problem formulation and optimizing over :math:`\tilde{\mathbf{c}}` will improve the numerical conditioning of the
    problem. The spline coefficients can then be recovered via
    :math:`\mathbf{c}^\ast = \mathbf{P} \tilde{\mathbf{c}}^\ast`, where :math:`\tilde{\mathbf{c}}^\ast` is the obtained
    solution of the preconditioned problem.

    See Also
    --------
    :py:func:`~pycsou.operator.linop.bspline.BSplineSampling`,
    :py:class:`~pycsou.operator.linop.bspline.BSplinePeriodicIndFunc`

    """
    xp = array_module.module()
    knots, degrees, ndim = _convert_to_list(knots, degrees, ndim=ndim)
    for i in range(ndim):
        n = knots[i].size - degrees[i] - 1
        weights = xp.asarray(degrees[i] / (knots[i][degrees[i] : -1] - knots[i][:n]), dtype=pycrt.getPrecision().value)
        precond = pycl.DiagonalOp(weights)
        precond_inv = pycl.DiagonalOp(1 / weights)
        if i == 0:
            precond_ndim, precond_inv_ndim, weights_ndim = precond, precond_inv, weights
        else:
            precond_ndim = pycl.kron(precond_ndim, precond)
            precond_inv_ndim = pycl.kron(precond_inv_ndim, precond_inv)
            weights_ndim = xp.kron(weights_ndim, weights)
    return precond_ndim, precond_inv_ndim, weights_ndim


@pycrt.enforce_precision(["knots", "precond_weights"], o=False)
def _periodic_constraints(
    knots: np.ndarray, degree: pyct.Integer, precond_weights: pyct.NDArray = 1
) -> tuple[pyca.LinOp, pyca.LinOp]:
    r"""
    Computes the linear operator of constraints on univariate B-spline coefficients for this spline to be periodic.
    A spline is periodic if and only if this operator applied to its B-spline coefficients outputs zero.

    Parameters
    ----------
    knots: ndarray
        Knots of the spline.
    degree: int
        Degree of the spline.
    precond_weights: ndarray, optional
        Preconditioning weights for nonuniform spline knots (see private
        :py:func:`~pycsou.operator.linop.bspline.precond_nonuniform_knots` function). Default is 1 (no preconditioning).

    Returns
    -------
    A: :py:class:`~pycsou.abc.operator.LinOp`
        Linear operator of constraints.
    P: :py:class:`~pycsou.abc.operator.LinOp`
        Projector onto the kernel of A (useful for the proximal operator of
        :py:class:`~pycsou.operator.linop.bspline.BSplinePeriodicIndFunc`).
    """
    xp = pycu.get_array_module(knots)
    t = np.asarray(knots, dtype=pycrt.getPrecision().value)
    interval = (t[degree], t[-degree - 1])  # Bounds of the period
    w = np.asarray(precond_weights, dtype=pycrt.getPrecision().value)
    rows = []
    for i in range(degree):
        if i == 0:  # s(0) = s(T)
            dm = sci.BSpline.design_matrix(x=np.array(interval), t=t, k=degree).todense(order="C")
            row_vec = dm[0] - dm[-1]
        else:  # s^(i)(0) = s^(i)(T)
            t_der = knots[i:-i] if i > 0 else knots
            sDk = _BSplineDerivative(
                knots=knots, degrees=degree, deriv_orders=i, array_module=pycd.NDArrayInfo.from_obj(knots)
            )
            dm = sci.BSpline.design_matrix(x=np.array(interval), t=t_der, k=degree - i).todense(order="C")
            dmd = dm[0] - dm[-1]
            row_vec = sDk.adjoint(dmd)
        rows.append(row_vec * w)

    A = np.stack(rows, axis=0)
    Adag = np.linalg.pinv(A)
    P = np.eye(A.shape[-1], dtype=pycrt.getPrecision().value) - Adag @ A
    A, P = xp.asarray(A, dtype=pycrt.getPrecision().value), xp.asarray(P, dtype=pycrt.getPrecision().value)
    return pyca.LinOp.from_array(A), pyca.LinOp.from_array(P)


class BSplinePeriodicIndFunc(pycn.ShiftLossMixin, pyca.ProxFunc):
    r"""
    Indicator function of periodic splines with smooth junctions applied to tensor-product B-splines coefficients.

    Notes
    _____
    Indicator function of the kernel of the linear operator :math:`\mathbf{A}` of constraints over tensor-product
    B-splines coefficient that enforce periodicity with smooth junctions. A tensor-product B-spline :math:`S` is
    periodic along its :math:`d`-th dimension with smooth junctions if its consecutive derivatives are
    :math:`T`-periodic at the boundary points :math:`t[d]_{k_d}` and :math:`t[d]_{-k_d}` with
    :math:`T = t[d]_{-k_d} - t[d]_{k_d}`, `i.e.`,

    .. math::
        S(x_1, \ldots, t[d]_{k_d}, \ldots, x_D) &= S(x_1, \ldots, t[d]_{-k_d}, \ldots, x_D) \\
        & \vdots \\
        S^{(k_d - 1)}(x_1, \ldots, t[d]_{k_d}, \ldots, x_D) &= S^{(k_d - 1)}(x_1, \ldots, t[d]_{-k_d}, \ldots, x_D)

    :math:`\forall \mathbf{x} \in \mathbb{R}^D` (see :py:func:`~pycsou.operator.linop.bspline.BSplineSampling` for
    notations). These linear constraints are implemented via the operator :math:`\mathbf{A}` so that :math:`S` is
    periodic with smooth junctions if and only if its B-spline coefficients :math:`\mathbf{c}` satisfy

    .. math::
        \mathbf{A}\mathbf{c} = \mathbf{0}.

    The indicator function of the kernel of :math:`\mathbf{A}` if defined as

    .. math::
        i_{ \ker{A} }( \mathbf{c} ) =
        \begin{cases}
        0 & \text{if }  \mathbf{A} \mathbf{c} = \mathbf{0}, \\
        + \infty & \text{otherwise};
        \end{cases}

    it can be used as a cost term in an optimization problem to enforce the periodicity of the reconstructed spline.

    The period(s) of :math:`S` are automatically determined by its knots :math:`\mathbf{t}[d]` and its degrees
    :math:`\mathbf{k}`: they correspond to the base intervals :math:`[t[d]_{k_d}, t[d]_{-k_d}]` of :math:`S` for every
    dimension :math:`d` where :math:`S` is periodic.

    By default, the operator :math:`\mathbf{A}` encodes the periodicity of :math:`S` over all its variables. However,
    it may only do so for a subset of its variables.

    Examples
    ________
    See the :py:mod:`~pycsou.operator.linop.bspline` module documentation for an example of how to use this functional
    within an image-reconstruction problem. In the following example, we illustrate its definition and its proximal
    operator, which projects spline coefficients onto the space of coefficients that represent periodic splines.

    .. plot::

        import matplotlib.pyplot as plt
        import scipy.interpolate as sci
        import numpy as np
        import pycsou.operator.linop.bspline as bsp

        # Construct a non-periodic spline
        degree = 3  # Cubic splines
        knots = bsp.uniform_knots(10, 0, 1, degree)  # Spline knots
        N = len(knots) - degree - 1  # Number of spline coefficients
        c = np.arange(N) ** 2  # Spline coefficients (quadratic function)
        spl = sci.BSpline(t=knots, c=c, k=degree, extrapolate='periodic')  # Non-periodic spline

        # Indicator function for periodic splines
        per_find_func = bsp.BSplinePeriodicIndFunc(knots, degree)
        assert per_find_func(c) == np.inf  # Assert that the spline is non-periodic

        # Project spline coefficients to get periodic spline using proximal operator of indicator function
        c_per = per_find_func.prox(c, tau=1)  # The value of tau is irrelevant
        assert per_find_func(c_per) == 0  # Assert that the underlying spline is periodic

        # Periodic spline object
        spl_per = sci.BSpline(t=knots, c=c_per, k=degree, extrapolate='periodic')

        # Plots
        grid = np.linspace(-0.5, 1.5, 1000)

        plt.figure()
        plt.plot(grid, spl(grid))
        plt.axvline(0, color='k')
        plt.axvline(1, color='k')
        plt.title('Original non-periodic spline')

        plt.figure()
        plt.plot(grid, spl_per(grid))
        plt.axvline(0, color='k')
        plt.axvline(1, color='k')
        plt.title('Periodized spline (with smooth junctions)')

    See Also
    --------
    :py:func:`~pycsou.operator.linop.bspline.BSplineSampling`,
    :py:func:`~pycsou.operator.linop.bspline._precond_nonuniform_knots`
    """

    def __init__(
        self,
        knots: typ.Union[np.ndarray, list[np.ndarray]],
        degrees: typ.Union[pyct.Integer, list[pyct.Integer]],
        is_periodic: typ.Union[bool, list[bool]] = True,
        precond_weights: typ.Union[pyct.NDArray, list[pyct.NDArray]] = 1,
        ndim: pyct.Integer = 1,
    ):
        r"""
        Parameters
        ----------
        knots: ndarray or list[ndarray]
            Knots of the spline (list if different for each dimension).
        degrees: int or list[int]
            Degree of the spline (list if different for each dimension).
        is_periodic: bool or list[str], optional
            Specify along which dimension the spline is periodic (list if different for each dimension).
        precond_weights: ndarray or list[ndarray], optional
            Preconditioning weights for nonuniform spline knots (see
            :py:func:`~pycsou.operator.linop.bspline._precond_nonuniform_knots`); list if different for each
            dimension. Default is 1 (no preconditioning).
        ndim: int, optional
            Number of dimensions, `i.e.`, of variables of the spline (can be inferred from previous parameters if a list
            is provided).
        """

        knots, degrees, is_periodic, precond_weights, ndim = _convert_to_list(
            knots, degrees, is_periodic, precond_weights, ndim=ndim
        )
        self.ndim = ndim
        _A_list, _P_list, shape_in = [], [], ()
        for i in range(ndim):
            # Number of spline coefficients for the corresponding dimension
            shape_in = (*shape_in, knots[i].size - degrees[i] - 1)
            # Only periodic boundary conditions leads to constraints for the corresponding dimension
            if is_periodic[i]:
                _A, _P = _periodic_constraints(knots=knots[i], degree=degrees[i], precond_weights=precond_weights[i])
                _A_list.append(_A)
                _P_list.append(_P)
            else:
                # If non-periodic, there are no constraints and projector is identity
                _A_list.append(pycl.NullOp(shape=(0, shape_in[-1])))
                _P_list.append(pycl.IdentityOp(dim=shape_in[-1]))
        self._A, self._P = _A_list, _P_list
        self.shape_in = shape_in
        self._lipschitz = np.inf
        super(BSplinePeriodicIndFunc, self).__init__(shape=(1, np.prod(shape_in)))

    @pycrt.enforce_precision("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        out = xp.zeros((*arr.shape[:-1], 1), dtype=arr.dtype)
        arr = arr.reshape((*arr.shape[:-1], *self.shape_in))
        for i in range(self.ndim):
            # Dimension i leads to N_1 * ... * N_{i-1} * degrees[i] * N_{i+1} * ... * N_{ndim} constraints
            tmp = xp.tensordot(self._A[i].asarray(xp=xp), arr, axes=([1], [-self.ndim + i]))
            tmp = xp.moveaxis(tmp, 0, -self.ndim + i)
            tmp = xp.all(xp.isclose(tmp, 0), axis=tuple(range(-self.ndim, 0)))
            # If not all constraints are satisfied, then the indicator is infinity (otherwise it is zero)
            out[xp.logical_not(tmp)] = np.inf
        return out

    @pycrt.enforce_precision("arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        # The proximal operator of the indicator function of the kernel of A is the
        # orthogonal projector onto the kernel of A. The value of tau is irrelevant.
        xp = pycu.get_array_module(arr)
        sh_out = (*arr.shape[:-1], self.dim)
        arr = arr.reshape((*arr.shape[:-1], *self.shape_in))
        for i in range(self.ndim):
            # Apply projectors sequentially. Since they commute, this projects onto the intersection of the sets.
            arr = xp.tensordot(self._P[i].asarray(xp=xp).astype(arr.dtype), arr, axes=([1], [-self.ndim + i]))
            arr = xp.moveaxis(arr, 0, -self.ndim + i)
        return arr.reshape(sh_out)


def uniform_knots(nb_of_knots: pyct.Integer, t_min: pyct.Real, t_max: pyct.Real, degree: pyct.Integer) -> np.ndarray:
    r"""
    Returns uniform array of knots based on limit points of a base interval. Includes boundary points outside
    the base interval so that a spline with these knots has the correct base interval (see Scipy's
    :py:class:`~scipy.interpolate.BSpline` class).

    Parameters
    ----------
    nb_of_knots: int
        Number of knots in base interval (including limit points).
    t_min: float
        Starting point of the base interval.
    t_max: float
        End point of the base interval.
    degree: int
        Degree of the spline.

    Returns
    -------
    knots: :py:class:`~numpy.ndarray`
        Array of uniform knots.
    """
    h = (t_max - t_min) / (nb_of_knots - 1)
    return np.linspace(t_min - degree * h, t_max + degree * h, nb_of_knots + 2 * degree)


# Multidimensional operators evaluated on a grid

# @pycrt.enforce_precision('eval_grid', o=False)
def BSplineGradientGrid(
    eval_grid: typ.Union[np.ndarray, list[np.ndarray]],
    knots: typ.Union[np.ndarray, list[np.ndarray]],
    degrees: typ.Union[pyct.Integer, list[pyct.Integer]],
    ndim: pyct.Integer = 1,
    array_module: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY,
) -> pyca.LinOp:
    r"""
    Returns a linear operator that takes as input :math:`D`-variate tensor-product B-spline coefficients and that
    outputs the gradient of the spline evaluated on a predefined grid.

    Parameters
    ----------
    eval_grid: ndarray or list[ndarray]
        Evaluation grid (list if different for each dimension) with shape :math:`(m_d,)`.
    knots: ndarray or list[ndarray]
        Knots of the spline (list if different for each dimension) with shape :math:`(n_d + k_d + 1,)`.
    degrees: int or list[int]
        Degree of the spline (list if different for each dimension).
    ndim: int, optional
        Number of variables :math:`D` of the spline (can be inferred from previous parameters if a list is provided).
    array_module: :py:class:`~pycsou.util.deps.NDArrayInfo`, optional
        Supported input ndarray API for the output operator (default is Numpy).

    Returns
    -------
    op: :py:class:`~pycsou.abc.operator.LinOp`, shape :math:`\left(D \times \prod_{d=0}^{D-1} m_d, \prod_{d=0}^{D-1} n_d \right)`.
        Gradient operator.

    Notes
    -----
    The output operator ``op`` takes a flattened ndarray of B-spline coefficients :math:`\mathbf{c}` with shape
    :math:`\left(\prod_{d=0}^{D-1} n_d, \right)` as input, and outputs the flattened ndarray

    .. math::

        \Big( \partial_{x_d} S \big((\mathbf{x}[0])_{j_0}, \ldots,(\mathbf{x}[D-1])_{j_{D-1}} \big) \Big)_
        {0 \leq d \leq D-1, \ 0 \leq j_0 \leq m_0, \ \ldots, \ 0 \leq j_{D-1} \leq m_{D-1}}

    with shape :math:`\left(D \times \prod_{d=0}^{D-1} m_d, \right)` (see
    :py:func:`~pycsou.operator.linop.bspline.BSplineSampling` for notations).

    This operator is implemented using sparse matrices via the :py:func:`~pycsou.operator.linop.bspline.BSplineSampling`
    function.

    Examples
    --------

    .. plot::

        import matplotlib.pyplot as plt
        import scipy.interpolate as sci
        import numpy as np
        from pycsou.operator.linop.bspline import BSplineGradientGrid
        from pycsou.util.misc import peaks

        # Generate 2D image
        grid = np.linspace(-2.5, 2.5, 50)
        xx, yy = np.meshgrid(grid, grid)
        image = peaks(xx, yy)

        # Interpolate image with B-splines
        degree = 3  # Cubic B-splines
        spl = sci.RectBivariateSpline(grid, grid, image, kx=degree, ky=degree, s=0)
        knots, _ = spl.get_knots()
        c = spl.get_coeffs()  # B-spline coefficients of the image

        # Compute gradient of continuous image
        grad_op = BSplineGradientGrid(grid, knots, degree, ndim=2)  # Gradient operator
        grad_im = grad_op(c).reshape((2, *image.shape))  # Gradient of image

        # Plots

        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.colorbar()
        plt.title(r'$f(x,y)$')

        plt.figure()
        plt.imshow(grad_im[0, :, :])
        plt.axis('off')
        plt.colorbar()
        plt.title(r'$\partial_{x_0} f(x,y)$')

        plt.figure()
        plt.imshow(grad_im[1, :, :])
        plt.axis('off')
        plt.colorbar()
        plt.title(r'$\partial_{x_1} f(x,y)$')

    See Also
    --------
    :py:func:`~pycsou.operator.linop.bspline.BSplineSampling`,
    :py:func:`~pycsou.operator.linop.bspline.BSplineLaplacianGrid`,
    :py:func:`~pycsou.operator.linop.bspline.BSplineHessianGrid`,
    :py:func:`~pycsou.operator.linop.diff.Gradient`,
    :py:func:`~pycsou.operator.linop.diff.DirectionalGradient`
    """
    eval_grid, knots, degrees, ndim = _convert_to_list(eval_grid, knots, degrees, ndim=ndim)
    grad_op_list = []
    deriv_orders = [0 for _ in range(ndim)]
    for i in range(ndim):
        deriv_orders[i] = 1
        grad_op_list.append(
            BSplineSampling(
                eval_grid=eval_grid,
                knots=knots,
                degrees=degrees,
                deriv_orders=deriv_orders,
                ndim=ndim,
                array_module=array_module,
            )
        )
        deriv_orders[i] = 0
    return pycb.vstack(grad_op_list)


# @pycrt.enforce_precision('eval_grid', o=False)
def BSplineLaplacianGrid(
    eval_grid: typ.Union[np.ndarray, list[np.ndarray]],
    knots: typ.Union[np.ndarray, list[np.ndarray]],
    degrees: typ.Union[pyct.Integer, list[pyct.Integer]],
    ndim: pyct.Integer = 1,
    array_module: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY,
) -> pyca.LinOp:
    r"""
    Returns a linear operator that takes as input :math:`D`-variate tensor-product B-spline coefficients and that
    outputs the Laplacian of the spline evaluated on a predefined grid.

    Parameters
    ----------
    eval_grid: ndarray or list[ndarray]
        Evaluation grid (list if different for each dimension) with shape :math:`(m_d,)`.
    knots: ndarray or list[ndarray]
        Knots of the spline (list if different for each dimension) with shape :math:`(n_d + k_d + 1,)`.
    degrees: int or list[int]
        Degree of the spline (list if different for each dimension).
    ndim: int, optional
        Number of variables :math:`D` of the spline (can be inferred from previous parameters if a list is provided).
    array_module: :py:class:`~pycsou.util.deps.NDArrayInfo`, optional
        Supported input ndarray API for the output operator (default is Numpy).

    Returns
    -------
    op: :py:class:`~pycsou.abc.operator.LinOp`, shape
        :math:`\left(\prod_{d=0}^{D-1} m_d, \prod_{d=0}^{D-1} n_d \right)`.
        Laplacian operator.

    Notes
    -----
    The output operator ``op`` takes a flattened ndarray of B-spline coefficients :math:`\mathbf{c}` with shape
    :math:`\left(\prod_{d=0}^{D-1} n_d, \right)` as input, and outputs the flattened ndarray

    .. math::

        \left( \sum_{d=0}^{D-1} \partial_{x_d}^2 S \big((\mathbf{x}[0])_{j_0}, \ldots,(\mathbf{x}[D-1])_{j_{D-1}} \big)
        \right)_{0 \leq j_0 \leq m_0, \ \ldots, \ 0 \leq j_{D-1} \leq m_{D-1}}

    with shape :math:`\left(\prod_{d=0}^{D-1} m_d, \right)` (see
    :py:func:`~pycsou.operator.linop.bspline.BSplineSampling` for notations).

    This operator is implemented using sparse matrices via the :py:func:`~pycsou.operator.linop.bspline.BSplineSampling`
    function.

    Examples
    --------

    .. plot::

        import matplotlib.pyplot as plt
        import scipy.interpolate as sci
        import numpy as np
        from pycsou.operator.linop.bspline import BSplineLaplacianGrid
        from pycsou.util.misc import peaks

        # Generate 2D image
        grid = np.linspace(-2.5, 2.5, 50)
        xx, yy = np.meshgrid(grid, grid)
        image = peaks(xx, yy)

        # Interpolate image with B-splines
        degree = 3  # Cubic B-splines
        spl = sci.RectBivariateSpline(grid, grid, image, kx=degree, ky=degree, s=0)
        knots, _ = spl.get_knots()
        c = spl.get_coeffs()  # B-spline coefficients of the image

        # Compute Laplacian of continuous image
        laplacian_op = BSplineLaplacianGrid(grid, knots, degree, ndim=2)  # Laplacian operator
        laplacian_im = laplacian_op(c).reshape(image.shape)  # Laplacian of image

        # Plots

        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.colorbar()
        plt.title(r'$f(x,y)$')

        plt.figure()
        plt.imshow(laplacian_im)
        plt.axis('off')
        plt.colorbar()
        plt.title(r'$\Delta f(x,y)$')

    See Also
    --------
    :py:func:`~pycsou.operator.linop.bspline.BSplineSampling`,
    :py:func:`~pycsou.operator.linop.bspline.BSplineGradientGrid`,
    :py:func:`~pycsou.operator.linop.bspline.BSplineHessianGrid`,
    :py:func:`~pycsou.operator.linop.diff.DirectionalLaplacian`
    """
    eval_grid, knots, degrees, ndim = _convert_to_list(eval_grid, knots, degrees, ndim=ndim)
    deriv_orders = [0 for _ in range(ndim)]
    for i in range(ndim):
        deriv_orders[i] = 2
        sampling_op = BSplineSampling(
            eval_grid=eval_grid,
            knots=knots,
            degrees=degrees,
            deriv_orders=deriv_orders,
            ndim=ndim,
            array_module=array_module,
        )
        if i == 0:
            out = sampling_op
        else:
            out += sampling_op
        deriv_orders[i] = 0
    return out


def BSplineHessianGrid(
    eval_grid: typ.Union[np.ndarray, list[np.ndarray]],
    knots: typ.Union[np.ndarray, list[np.ndarray]],
    degrees: typ.Union[pyct.Integer, list[pyct.Integer]],
    ndim: pyct.Integer = 1,
    array_module: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY,
) -> pyca.LinOp:
    r"""
    Returns a linear operator that takes as input :math:`D`-variate tensor-product B-spline coefficients and that
    outputs the Hessian of the spline evaluated on a predefined grid.

    Parameters
    ----------
    eval_grid: ndarray or list[ndarray]
        Evaluation grid (list if different for each dimension) with shape :math:`(m_d,)`.
    knots: ndarray or list[ndarray]
        Knots of the spline (list if different for each dimension) with shape :math:`(n_d + k_d + 1,)`.
    degrees: int or list[int]
        Degree of the spline (list if different for each dimension).
    ndim: int, optional
        Number of variables :math:`D` of the spline (can be inferred from previous parameters if a list is provided).
    array_module: :py:class:`~pycsou.util.deps.NDArrayInfo`, optional
        Supported input ndarray API for the output operator (default is Numpy).

    Returns
    -------
    op: :py:class:`~pycsou.abc.operator.LinOp`, shape
        :math:`\left( \frac{D(D+1)}2 \times \prod_{d=0}^{D-1} m_d, \prod_{d=0}^{D-1} n_d \right)`.
        Hessian operator.

    Notes
    -----
    The output operator ``op`` takes a flattened ndarray of B-spline coefficients :math:`\mathbf{c}` with shape
    :math:`\left(\prod_{d=0}^{D-1} n_d, \right)` as input, and outputs the flattened ndarray

    .. math::

        \Big( \mathbf{H}(j_0, \ldots, j_{D-1})_k \Big)_{0 \leq k \leq \frac{D(D+1)}2 - 1, \ 0 \leq j_0 \leq m_0, \
        \ldots, \ 0 \leq j_{D-1} \leq m_{D-1}}

    with shape :math:`\left(\frac{D (D+1)}2 \times \prod_{d=0}^{D-1} m_d, \right)`. The vector
    :math:`\mathbf{H}(j_0, \ldots, j_{D-1})` corresponds to the stacked upper triangular elements of the Hessian matrix
    of the spline :math:`S`, which is given by

    .. math::

        \mathbf{H}(j_0, \ldots, j_{D-1}) = \Big( \partial_{x_0} \partial_{x_0} S, \ldots,
        \partial_{x_0} \partial_{x_{D-1}} S, \partial_{x_1} \partial_{x_1} S, \ldots,
        \partial_{x_{D-1}} \partial_{x_{D-1}} S \Big)

    where the notation :math:`\partial_{x_d} \partial_{x_{d'}} S` is a shorthand for
    :math:`\partial_{x_d} \partial_{x_{d'}} S \big((\mathbf{x}[0])_{j_0}, \ldots,(\mathbf{x}[D-1])_{j_{D-1}} \big)`
    (see :py:func:`~pycsou.operator.linop.bspline.BSplineSampling` for notations).

    This operator is implemented using sparse matrices via the :py:func:`~pycsou.operator.linop.bspline.BSplineSampling`
    function.

    Examples
    --------

    .. plot::

        import matplotlib.pyplot as plt
        import scipy.interpolate as sci
        import numpy as np
        from pycsou.operator.linop.bspline import BSplineHessianGrid
        from pycsou.util.misc import peaks

        # Generate 2D image
        grid = np.linspace(-2.5, 2.5, 50)
        xx, yy = np.meshgrid(grid, grid)
        image = peaks(xx, yy)

        # Interpolate image with B-splines
        degree = 3  # Cubic B-splines
        spl = sci.RectBivariateSpline(grid, grid, image, kx=degree, ky=degree, s=0)
        knots, _ = spl.get_knots()
        c = spl.get_coeffs()  # B-spline coefficients of the image

        # Compute Hessian of continuous image
        hessian_op = BSplineHessianGrid(grid, knots, degree, ndim=2)  # Hessian operator
        hessian_im = hessian_op(c).reshape((3, *image.shape))  # Hessian of image

        # Plots

        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.colorbar()
        plt.title(r'$f(x,y)$')

        plt.figure()
        plt.imshow(hessian_im[0, :, :])
        plt.axis('off')
        plt.colorbar()
        plt.title(r'$\partial_{x_0} \partial_{x_0} f(x,y)$')

        plt.figure()
        plt.imshow(hessian_im[1, :, :])
        plt.axis('off')
        plt.colorbar()
        plt.title(r'$\partial_{x_0} \partial_{x_1} f(x,y)$')

        plt.figure()
        plt.imshow(hessian_im[2, :, :])
        plt.axis('off')
        plt.colorbar()
        plt.title(r'$\partial_{x_1} \partial_{x_1} f(x,y)$')

    See Also
    --------
    :py:func:`~pycsou.operator.linop.bspline.BSplineSampling`,
    :py:func:`~pycsou.operator.linop.bspline.BSplineGradientGrid`,
    :py:func:`~pycsou.operator.linop.bspline.BSplineLaplacianGrid`,
    :py:func:`~pycsou.operator.linop.diff.Hessian`,
    :py:func:`~pycsou.operator.linop.diff.DirectionalHessian`
    """
    eval_grid, knots, degrees, ndim = _convert_to_list(eval_grid, knots, degrees, ndim=ndim)
    hess_op_list = []
    deriv_orders = [0 for _ in range(ndim)]
    for i in range(ndim):
        deriv_orders[i] = 1
        for j in range(ndim - i):
            deriv_orders[j + i] += 1
            hess_op_list.append(
                BSplineSampling(
                    eval_grid=eval_grid,
                    knots=knots,
                    degrees=degrees,
                    deriv_orders=deriv_orders,
                    ndim=ndim,
                    array_module=array_module,
                )
            )
            deriv_orders[j + i] -= 1
        deriv_orders[i] = 0
    return pycb.vstack(hess_op_list)


def _is_valid_knots(knots: np.ndarray, degree: pyct.Integer, bc_type: str = None):
    r"""
    Determines whether a sequence of knots is valid to be able to compute innovations of a spline with these knots.
    """
    internal_knots = knots[degree + 1 : -(degree + 1)]
    if len(internal_knots) != len(np.unique(internal_knots)):
        return False
    if bc_type == "zero":
        if knots[degree] == knots[degree + 1] or knots[-degree] == knots[-(degree + 1)]:
            return False
    return True


class _PiecewiseCstInnos1D(pyca.LinOp):
    r"""
    Linear operator that takes B-spline coefficients of a 1D piecewise-constant spline and outputs its innovations.
    """

    def __init__(self, dim, bc_type: str = None):
        if bc_type is None:
            bc_type = "not-a-knot"
        if bc_type == "not-a-knot":
            codim = dim - 1
        elif bc_type == "periodic":
            codim = dim
        elif bc_type == "zero":
            codim = dim + 1
        else:
            raise ValueError("Unsupported boundary condition")
        self.bc_type = bc_type
        super().__init__(shape=(codim, dim))

    @pycrt.enforce_precision("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        zero = arr.dtype.type(0)  # To have correct dtype at output of xp.diff
        if self.bc_type == "not-a-knot":
            arrdiff = xp.diff(arr, axis=-1)
        elif self.bc_type == "periodic":
            arrdiff = xp.diff(arr, prepend=zero, axis=-1)
        elif self.bc_type == "zero":
            arrdiff = xp.diff(arr, prepend=zero, append=zero, axis=-1)
        return arrdiff

    @pycrt.enforce_precision("arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        zero = arr.dtype.type(0)  # To have correct dtype at output of xp.diff
        if self.bc_type == "not-a-knot":
            arrdiff = -xp.diff(arr, prepend=zero, append=zero, axis=-1)
        elif self.bc_type == "periodic":
            arrdiff = -xp.diff(arr, append=zero, axis=-1)
        elif self.bc_type == "zero":
            arrdiff = -xp.diff(arr, axis=-1)
        return arrdiff


# Below is a multidimensional-dimensional implementation of _PiecewiseCstInnos. We have chosen to support only on the 1D
# case since the multidimensional case is not well understood, but we leave it here for future reference.

# class _PiecewiseCstInnos(pyca.LinOp):
#     r"""
#     Linear operator that takes B-spline coefficients of a piecewise-constant spline and outputs its innovations.
#     """
#
#     def __init__(self, shape_in: tuple[pyct.Integer, ...], bc_types: typ.Union[str, list[str]] = None):
#         self.shape_in = shape_in
#         self.ndim = len(shape_in)
#         bc_types, ndim = _convert_to_list(bc_types, ndim=self.ndim)
#         assert ndim == self.ndim
#         bc_types = ["not-a-knot" if bc is None else bc for bc in bc_types]
#         shape_out = ()
#         for i in range(self.ndim):
#             bc = bc_types[i]
#             if bc == "not-a-knot":
#                 shape_out = (*shape_out, shape_in[i] - 1)
#             elif bc == "periodic":
#                 shape_out = (*shape_out, shape_in[i])
#             elif bc == "zero":
#                 shape_out = (*shape_out, shape_in[i] + 1)
#             else:
#                 raise ValueError("Unsupported boundary condition")
#         self.shape_out = shape_out
#         self.bc_types = bc_types
#         super().__init__(shape=(np.prod(shape_out), np.prod(shape_in)))
#
#     @pycrt.enforce_precision("arr")
#     def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
#         xp = pycu.get_array_module(arr)
#         arrdiff = arr.reshape((*arr.shape[:-1], *self.shape_in))
#         zero = arr.dtype.type(0)  # To have correct dtype at output of xp.diff
#         for i in range(self.ndim):
#             if self.bc_types[i] == "not-a-knot":
#                 arrdiff = xp.diff(arrdiff, axis=-self.ndim + i)
#             elif self.bc_types[i] == "periodic":
#                 arrdiff = xp.diff(arrdiff, prepend=zero, axis=-self.ndim + i)
#             elif self.bc_types[i] == "zero":
#                 arrdiff = xp.diff(arrdiff, prepend=zero, append=zero, axis=-self.ndim + i)
#         return arrdiff.reshape((*arr.shape[:-1], self.codim))
#
#     @pycrt.enforce_precision("arr")
#     def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
#         xp = pycu.get_array_module(arr)
#         arrdiff = arr.reshape((*arr.shape[:-1], *self.shape_out))
#         zero = arr.dtype.type(0)  # To have correct dtype at output of xp.diff
#         for i in range(len(self.shape_in)):
#             if self.bc_types[i] == "not-a-knot":
#                 arrdiff = -xp.diff(arrdiff, prepend=zero, append=zero, axis=-self.ndim + i)
#             elif self.bc_types[i] == "periodic":
#                 arrdiff = -xp.diff(arrdiff, append=zero, axis=-self.ndim + i)
#             elif self.bc_types[i] == "zero":
#                 arrdiff = -xp.diff(arrdiff, axis=-self.ndim + i)
#
#         return arrdiff.reshape((*arr.shape[:-1], self.dim))


def BSplineInnos1D(
    knots: np.ndarray,
    degree: pyct.Integer,
    bc_type: str = None,
    array_module: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY,
) -> pyca.LinOp:
    r"""
    Returns linear operator that takes as input B-spline coefficients of a 1D spline and returns its innovations within
    the base interval.

    Parameters
    ----------
    knots: ndarray
        Knots of the spline.
    degree: int
        Degree of the spline.
    bc_type: str, optional
        Type of boundary conditions for the spline. The following conditions are supported:

        * ``"not-a-knot"`` (default): The spline has no knots at the limit points, which amounts
          to constant extrapolation. This is equivalent to ``None``.

        * ``"periodic"``: The spline has a knot at the starting point (identified with the end
          point with periodic boundary conditions).

        * ``"zero"``: The spline is zero outside the base interval, and has knots at both limit points.
    array_module: :py:class:`~pycsou.util.deps.NDArrayInfo`, optional
        Supported input ndarray API for the output operator (default is Numpy).

    Returns
    -------
    op: :py:class:`~pycsou.abc.operator.LinOp`
        Linear operator of innovations.

    Notes
    -----
    The `innovation` of a spline :math:`S` of degree :math:`k` is the :math:`(k+1)`-th derivative of :math:`S`, which
    in common cases is a Dirac comb of the form

    .. math::
        :label: eq:inno

        w(x) = S^{(k+1)}(x) = \sum_{n=1}^N a_n \delta(x - x_n),

    where the :math:`x_n` are knots of :math:`S` and the :math:`a_n` are the `amplitudes` of these knots.

    Computing the innovation of a spline is useful to solve continuous-domain inverse problems with higher-order
    total-variation regularization, which is known to promote spline solutions with few knots (see `Examples` section
    and [2]).

    The output operator ``op``, represented by a matrix :math:`\mathbf{L}`, computes the innovation of :math:`S` by
    taking its B-spline coefficients :math:`\mathbf{c}` as input (see
    :py:func:`~pycsou.operator.linop.bspline.BSplineSampling` for notations), `i.e.`,

    .. math::

        \mathbf{L} \mathbf{c} = \mathbf{a}.

    The matrix :math:`\mathbf{L}` is band-diagonal with :math:`k+1` diagonals; it represents a discrete convolution in
    the case where :math:`S` has uniform knots (see [2]). The vector :math:`\mathbf{a}` then represents the
    amplitudes of :math:`S` at the knots:

        * :math:`t_{k+1}, \ldots, t_{-(k+1)}` if ``bc_type`` = ``"not-a-knot"``;

        * :math:`t_{k}, \ldots, t_{-(k+1)}` if ``bc_type`` = ``"periodic"``;

        * :math:`t_{k}, \ldots, t_{-k}` if ``bc_type`` = ``"zero"``.

    The operator ``op`` is typically used in combination with an
    :py:class:`~pycsou.operator.func.norm.L1Norm` due to the relation

    .. math::

        \Vert S^{(k+1)} \Vert_{\mathcal{M}} = \Vert \mathbf{L} \mathbf{c} \Vert_1,

    where :math:`\Vert \cdot \Vert_{\mathcal{M}}` is the total-variation norm for measures (see [BSplineRepThm]_ and
    [BSplineTV]_).

    The operator ``op`` is implemented via a sparse matrix API (see the `Notes` section of the
    :py:func:`~pycsou.operator.linop.bspline.BSplineSampling` function).

    **Remark**
    This function requires all the `internal` knots of :math:`S` to have multiplicity 1, `i.e.`,
    :math:`t_{k+1}, \ldots, t_{-(k+1)}` must be pairwise distinct. This condition is equivalent to the innovation
    :math:`w` being a Dirac comb as in Eq. :math:numref:`eq:inno` (see [BSplineDeBoor]_).

    Examples
    --------
    Consider the problem of interpolating :math:`M` 1D data points :math:`(x_m, y_m)_{1 \leq m \leq M}` with a
    sparse spline or degree :math:`k`, `i.e.`, a spline with few knots. To achieve this, we formulate the following
    optimization problem with higher-order total-variation regularization

    .. math::

        \min_{S} \frac12 \sum_{m = 1} (S(x_m) - y_m) ^2 + \lambda \Vert S^{(k+1)} \Vert_{\mathcal{M}},

    where the regularization parameter :math:`\lambda > 0` controls the tradeoff between data fidelity and the sparsity
    of the spline. Following [2], we discretize this problem in the B-spline basis and optimize over the B-spline
    coefficients, which yields

    .. math::

        \min_{\mathbf{c} \in \mathbb{R}^n} \frac12 \Vert \mathbf{G} \mathbf{c} - \mathbf{y} \Vert_2^2 + \lambda
        \Vert \mathbf{L} \mathbf{c} \Vert_1,

    where :math:`\mathbf{G} \in \mathbb{R}^{M \times n}` is the design matrix of the spline at the evaluation points
    :math:`x_1, \ldots, x_M` (see Scipy's :py:meth:`~scipy.interpolate.BSpline.design_matrix` method) and
    :math:`\mathbf{L}` is the :py:func:`~pycsou.operator.linop.bspline.BSplineInnos1D` operator. The reconstructed
    signal is then the spline :math:`S` whose B-spline coefficients are the solution to this problem.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy as sp

        import pycsou.operator.linop.bspline as bsp
        import pycsou.operator as pycop
        import pycsou.operator.func.norm as pycn
        from pycsou.opt.solver.pds import ADMM

        # Spline parameters
        degree = 1  # Linear splines

        # Generate random ground-truth linear spline with few knots
        rng = np.random.default_rng(seed=0)  # For reproducibility
        num_knots_gt = 5  # Number of knots of the ground truth
        # Amplitudes of knots of ground truth
        a_gt = rng.uniform(1, 2, size=num_knots_gt) * (-1) ** np.arange(num_knots_gt)
        # Knots of ground truth
        knots_gt = (rng.uniform(size=num_knots_gt) + np.arange(num_knots_gt)) / num_knots_gt
        gt = lambda x: np.sum(a_gt * np.maximum(x - knots_gt.reshape((num_knots_gt, 1)), 0).transpose(), axis=1)

        # Generate data
        M = 10  # Number of data points
        x = ((0.5 + rng.uniform(size=M)) / 2 + np.arange(M)) / M  # Nonuniform sampling locations
        sigma = 1e-2  # Noise variance
        y = gt(x) + sigma * rng.standard_normal(size=M)  # Noisy data points

        # Optimization problem parameters
        knots = bsp.uniform_knots(20, 0, 1, degree)  # Spline knots for the reconstruction (uniform grid)
        knots[degree] = 0
        knots[-degree] = 1
        N = len(knots) - degree - 1  # Number of spline coefficients

        # Data fidelity term
        G = bsp.BSplineSampling(eval_grid=x, knots=knots, degrees=degree)
        f = 1 / 2 * pycop.SquaredL2Norm().argshift(-y) * G
        f._diff_lipschitz = G.gram().lipschitz()
        # Regularization term
        L = bsp.BSplineInnos1D(knots, degree)
        L.lipschitz()
        h = pycn.L1Norm(dim=L.codim)
        lamb = 3*1e-4  # Regularization parameter

        # Solver for ADMM
        tau = 1  # By default, the tau parameter is 1 in ADMM
        A_inv = sp.linalg.inv(G.gram().asarray() + (1 / tau) * L.gram().asarray())
        def solver_ADMM(arr, tau):
            b = (1 / tau) * L.adjoint(arr) + G.adjoint(y)
            return A_inv @ b.squeeze()

        # Solve optimization problem
        ADMM = ADMM(f=f, h=lamb * h, K=L, solver=solver_ADMM)  # Problem formulation
        ADMM.fit(**dict(x0=np.zeros(N)))
        c_opt = ADMM.solution()  # Spline coefficients of reconstructed signal

        # Plots
        grid = np.linspace(0, 1, 1000)
        # Operator to evaluate reconstructed signal on the plotting grid
        plot_op = bsp.BSplineSampling(eval_grid=grid, knots=knots, degrees=degree)

        plt.figure()
        plt.plot(grid, gt(grid), label='Ground truth')
        plt.plot(x, y, 'kx', label='Noisy data points')
        plt.plot(grid, plot_op(c_opt).squeeze(), label='Reconstructed signal')
        plt.legend()

    See Also
    --------
    :py:func:`~pycsou.operator.linop.bspline.BSplineSampling`

    """

    assert _is_valid_knots(knots, degree, bc_type)
    deriv_op = _BSplineDerivative(knots=knots, degrees=degree, deriv_orders=degree, ndim=1, array_module=array_module)
    dim = len(knots) - 2 * degree - 1
    # len(knots) - degree - 1 for the number of spline coefficients before differentiation, and an additional
    # - degree for the differentiation.
    innos_op = _PiecewiseCstInnos1D(dim, bc_type)
    return innos_op * deriv_op


# Below is a multidimensional-dimensional implementation of BSplineInnos. We have chosen to support only on the 1D case
# since the multidimensional case is not well understood, but we leave it here for future reference.
#
# def BSplineInnos(
#     knots: typ.Union[np.ndarray, list[np.ndarray]],
#     degrees: typ.Union[pyct.Integer, list[pyct.Integer]],
#     bc_types: typ.Union[str, list[str]] = None,
#     ndim: pyct.Integer = 1,
#     array_module: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY,
# ) -> pyca.LinOp:
#     r"""
#     Returns linear operator that takes as input B-spline coefficients of a tensor-product spline and returns its
#     innovations within the base interval.
#
#     Parameters
#     ----------
#     knots: ndarray or list[ndarray]
#         Knots of the spline (list if different for each dimension).
#     degrees: int or list[int]
#         Degree of the spline (list if different for each dimension).
#     bc_types: str or list[str], optional
#         Types of boundary conditions (list if different for each dimension). The following conditions are supported:
#
#         * ``"not-a-knot"`` (default): The spline has no knots at the limit points, which amounts
#         to constant extrapolation. This is equivalent to ``None``.
#
#         * ``"periodic"``: The spline has a knot at the starting point (identified with the end
#         point with periodic boundary conditions).
#
#         * ``"zero"``: The spline is zero outside the base interval, and has knots at both limit points.
#     ndim: int
#         Number of dimensions of the B-spline (can be inferred from previous parameters if a list is provided).
#     array_module: :py:class:`~pycsou.util.deps.NDArrayInfo`, optional
#         Supported input ndarray API for the output operator (default is Numpy).
#
#     Returns
#     -------
#     op: :py:class:`~pycsou.abc.operator.LinOp`
#         Linear operator of innovations.
#     """
#     knots, degrees, bc_types, ndim = _convert_to_list(knots, degrees, bc_types, ndim=ndim)
#     for i in range(ndim):
#         assert _is_valid_knots(knots[i], degrees[i], bc_types[i])
#     sDk = _BSplineDerivative(knots=knots, degrees=degrees, deriv_orders=degrees, ndim=ndim, array_module=array_module)
#     shape_in = tuple([len(knots[i]) - 2 * degrees[i] - 1 for i in range(ndim)])
#     # len(knots[i]) - degrees[i] - 1 for the number of spline coefficients before differentiation, and an additional
#     # - degrees[i] for the differentiation.
#     PCD = _PiecewiseCstInnos(shape_in, bc_types)
#     return PCD * sDk
