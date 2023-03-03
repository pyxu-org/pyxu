import functools
import typing as typ
import warnings

import numpy as np
import scipy.optimize as sciop

import pycsou.abc as pyca
import pycsou.abc.solver as pysolver
import pycsou.operator.blocks as pyblock
import pycsou.operator.linop.base as pybase
import pycsou.operator.linop.diff as pydiff
import pycsou.opt.solver as pysol
import pycsou.opt.stop as pystop
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


class _BalloonForce(pyca.Map):
    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None):
        self.arg_shape = arg_shape
        self.size = int(np.prod(arg_shape))
        super().__init__(shape=(1, self.size))
        self.grad = gradient
        if gradient:
            msg = "`gradient.arg_shape`={} inconsistent with `arg_shape`={}.".format(gradient.arg_shape, arg_shape)
            assert gradient.arg_shape == arg_shape, msg

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        raise NotImplementedError


class Dilation(_BalloonForce):
    # disclaimer: very small steps may be required for stability
    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None):
        super().__init__(arg_shape=arg_shape, gradient=gradient)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        grad_arr = self.grad.unravel(self.grad(arr))
        axis = 0 + 1 * (arr.shape[0] > 1)
        return xp.linalg.norm(grad_arr, axis=axis, keepdims=True)


class Erosion(_BalloonForce):
    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None):
        self = -Dilation(arg_shape=arg_shape, gradient=gradient)


class _Diffusivity(pyca.Map):
    """
    Abstract class to define diffusivity functions.
    The class features a method energy_functional(), which can be used to evaluate the energy potential that a
    divergence-based diffusion featuring the considered diffusivity derives from (when it makes sense).
    When implementing a new diffusivity, one needs to define whether it allows a variational interpretation or not.
    """

    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None):
        self.arg_shape = arg_shape
        self.size = int(np.prod(arg_shape))
        super().__init__(shape=(self.size, self.size))
        self.gradient = gradient
        if gradient:
            msg = "`gradient.arg_shape`={} inconsistent with `arg_shape`={}.".format(gradient.arg_shape, arg_shape)
            assert gradient.arg_shape == arg_shape, msg
        self.frozen = False
        self.frozen_diffusivity = None
        self.from_potential = False
        self.bounded = False

    def unravel_grad(self, arr):
        return arr.reshape(*arr.shape[:-1], -1, self.size)

    def freeze(self, arr: pyct.NDArray):
        if self.frozen:
            warnings.warn("Diffusivity has already been frozen. Cannot overwrite previous frozen state.")
        else:
            self.frozen_element = self.apply(arr)
            self.frozen = True

    def set_frozen_diffusivity(self, frozen_diffusivity: pyct.NDArray):
        if self.frozen:
            warnings.warn("Diffusivity has already been frozen. Cannot overwrite previous frozen state.")
        else:
            self.frozen_diffusivity = frozen_diffusivity
            self.frozen = True

    @pycrt.enforce_precision(i="arr")
    def _compute_grad_norm_sq(self, arr: pyct.NDArray, grad: pyct.OpT = None):
        xp = pycu.get_array_module(arr)
        # grad_arr = grad.unravel(grad(arr))
        grad_arr = self.unravel_grad(grad(arr))
        grad_arr **= 2
        return xp.sum(grad_arr, axis=1, keepdims=False)

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        raise NotImplementedError

    def energy_potential(self, arr: pyct.NDArray, grad: pyct.OpT = None) -> pyct.NDArray:
        return NotImplemented


class TikhonovDiffusivity(_Diffusivity):
    def __init__(self, arg_shape: pyct.NDArrayShape):
        super().__init__(arg_shape=arg_shape)
        self.from_potential = True
        self.bounded = True

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.ones(arr.size)

    @pycrt.enforce_precision(i="arr")
    def energy_potential(self, arr: pyct.NDArray, grad: pyct.OpT = None) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = self._compute_grad_norm_sq(arr, grad)
        y *= 0.5
        return xp.sum(y, axis=-1)


class MfiDiffusivity(_Diffusivity):
    # will not necesarily be exhibited
    def __init__(self, arg_shape: pyct.NDArrayShape, tame: bool = True):
        super().__init__(arg_shape=arg_shape)
        self.tame = tame
        self.bounded = tame

    @pycrt.enforce_precision(i="arr")
    def _apply_tame(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.clip(arr, 0, None)
        y += 1
        return 1 / y

    @pycrt.enforce_precision(i="arr")
    def _apply_untamed(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.clip(arr, 0, None)
        return 1 / y

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        f = self._apply_tame if self.tame else self._apply_untamed
        out = f(arr)
        return out


class PeronaMalikDiffusivity(_Diffusivity):
    def __init__(
        self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT, beta: pyct.Real = 1, pm_fct: str = "exponential"
    ):
        assert pm_fct in ["exponential", "rational"], "Unknown `pm_fct`, allowed values are `exponential`, `rational`."
        super().__init__(arg_shape=arg_shape, gradient=gradient)
        self.beta = beta
        self.pm_fct = pm_fct
        self.from_potential = True
        self.bounded = True

    @pycrt.enforce_precision(i="arr")
    def _apply_exponential(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        # Inplace implementation of
        #   xp.exp(-grad_norm_sq/beta**2)
        y = self._compute_grad_norm_sq(arr, self.gradient)
        y /= self.beta**2
        return xp.exp(-y)

    @pycrt.enforce_precision(i="arr")
    def _apply_rational(self, arr: pyct.NDArray) -> pyct.NDArray:
        # Inplace implementation of
        #   1 / (1 + grad_norm_sq/beta**2)
        y = self._compute_grad_norm_sq(arr, self.gradient)
        y /= self.beta**2
        y += 1
        return 1 / y

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        f = dict(
            exponential=self._apply_exponential,
            rational=self._apply_rational,
        ).get(self.pm_fct)
        out = f(arr)
        return out

    @pycrt.enforce_precision(i="arr")
    def _energy_functional_exponential(self, arr: pyct.NDArray, grad: pyct.OpT) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        # Inplace implementation of
        #   0.5*(beta**2)*(1 - xp.exp(-grad_norm_sq/beta**2)
        y = -self._compute_grad_norm_sq(arr, grad)
        y /= self.beta**2
        y = -xp.exp(y)
        y += 1
        y *= self.beta**2
        y *= 0.5
        return xp.sum(y, axis=-1)

    @pycrt.enforce_precision(i="arr")
    def _energy_functional_rational(self, arr: pyct.NDArray, grad: pyct.OpT) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        # Inplace implementation of
        #   0.5*(beta**2)*(xp.log(1+grad_norm_sq/beta**2)
        y = self._compute_grad_norm_sq(arr, grad)
        y /= self.beta**2
        y += 1
        y = xp.log(y)
        y *= self.beta**2
        y *= 0.5
        return xp.sum(y, axis=-1)

    @pycrt.enforce_precision(i="arr")
    def energy_functional(self, arr: pyct.NDArray, grad: pyct.OpT) -> pyct.NDArray:
        f = dict(
            exponential=self._energy_functional_exponential,
            rational=self._energy_functional_rational,
        ).get(self.pm_fct)
        out = f(arr, grad)
        return out


class TotalVariationDiffusivity(_Diffusivity):
    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT, tame: bool = True, beta: pyct.Real = 1):
        super().__init__(arg_shape=arg_shape, gradient=gradient)
        self.tame = tame
        self.beta = beta
        self.from_potential = True
        self.bounded = tame

    @pycrt.enforce_precision(i="arr")
    def _apply_tame(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        # Inplace implementation of
        #   1/(xp.sqrt(1+grad_norm_sq/beta**2)
        y = self._compute_grad_norm_sq(arr, self.gradient)
        y /= self.beta**2
        y += 1
        y = xp.sqrt(y)
        return 1 / y

    @pycrt.enforce_precision(i="arr")
    def _apply_untamed(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = self._compute_grad_norm_sq(arr, self.gradient)
        return 1 / y

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        f = self._apply_tame if self.tame else self._apply_untamed
        out = f(arr)
        return out

    @pycrt.enforce_precision(i="arr")
    def _energy_functional_tame(self, arr: pyct.NDArray, grad: pyct.OpT) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = self._compute_grad_norm_sq(arr, grad)
        y += 1
        y = xp.sqrt(y)
        return xp.sum(y, axis=-1)

    @pycrt.enforce_precision(i="arr")
    def _energy_functional_untamed(self, arr: pyct.NDArray, grad: pyct.OpT) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = self._compute_grad_norm_sq(arr, grad)
        y = xp.sqrt(y)
        return xp.sum(y, axis=-1)

    @pycrt.enforce_precision(i="arr")
    def energy_functional(self, arr: pyct.NDArray, grad: pyct.OpT) -> pyct.NDArray:
        f = self._energy_functional_tame if self.tame else self._energy_functional_untamed
        out = f(arr, grad)
        return out


class _DiffusionCoefficient(pyca.Map):
    # it's not map, return of apply is a, operator! Currently using map with abuse of "notation".
    # should we rather do -> class _DiffusionCoefficient:?
    """
    Abstract class for diffusion tensors.
    The daughter classes DiffusionCoefficientIsotropic, DiffusionTensorAnisotropic allow to handle the isotropic/anisotropic cases.
    """

    def __init__(self, arg_shape: pyct.NDArrayShape, isotropic: bool = True, trace_term: bool = False):
        self.arg_shape = arg_shape
        self.size = int(np.prod(arg_shape))
        self.ndims = len(self.arg_shape)
        super().__init__(shape=(1, self.size))
        self._lipschitz = np.inf
        self.isotropic = isotropic
        self.trace_term = trace_term
        self.from_potential = False
        self.frozen = False
        self.frozen_op = None
        self.bounded = False
        self._coeff_op = np.ones((self.ndims, self.ndims), dtype=int)
        if self.trace_term:
            # set to 2 extra diagonal coefficients
            self._coeff_op *= 2
            self._coeff_op -= np.eye(self.ndims, dtype=int)

    def freeze_tensor(self, arr: pyct.NDArray):
        if self.frozen:
            warnings.warn("DiffusionCoefficient has already been frozen. Cannot overwrite previous frozen state.")
        else:
            self.frozen_op = self.apply(arr)
            self.frozen = True

    def set_frozen_op(self, frozen_op: pyct.OpT):
        if self.frozen:
            warnings.warn("DiffusionCoefficient has already been frozen. Cannot overwrite previous frozen state.")
        else:
            self.frozen_op = frozen_op
            self.frozen = True

    def apply(self, arr: pyct.NDArray) -> pyct.OpT:
        if self.frozen:
            return self.frozen_op
        else:
            raise NotImplementedError


class DiffusionCoeffIsotropic(_DiffusionCoefficient):
    """
    Isotropic diffusion tensor. The diffusion tensor applied to the gradient to compute local flux is thus a diagonal matrix.
    If no diffusivity is specified at initialization, a homogeneous unitary diffusivity is imposed (corresponding to Tikhonov diffusivity).
    The diffusivity can be set at later times via set_diffusivity().
    """

    def __init__(self, arg_shape: pyct.NDArrayShape, diffusivity: pyct.OpT = None, trace_term: bool = False):
        super().__init__(arg_shape=arg_shape, isotropic=True, trace_term=trace_term)
        if diffusivity is None:
            self.diffusivity = TikhonovDiffusivity(arg_shape=arg_shape)
        else:
            msg = "`diffusivity.arg_shape`={} inconsistent with `arg_shape`={}.".format(
                diffusivity.arg_shape, arg_shape
            )
            assert diffusivity.arg_shape == arg_shape, msg
            self.diffusivity = diffusivity
        self.from_potential = self.diffusivity.from_potential * (not trace_term)
        if diffusivity.bounded:
            self.bounded = True

    # @pycrt.enforce_precision(i="arr")
    # @pycu.vectorize("arr")
    # why is vectorize not working? I can understand why enforce_precision doesn't (we are returning an operator, not a real value/array), but vectorize?!
    def apply(self, arr: pyct.NDArray) -> pyct.OpT:
        if not self.frozen:
            xp = pycu.get_array_module(arr)
            y = self.diffusivity(arr)
            if self.trace_term:
                ops = [pybase.DiagonalOp(y.squeeze())]
                ops *= self.ndims
                return pyblock.hstack(ops)
            else:
                return pybase.DiagonalOp(xp.tile(y.squeeze(), len(self.arg_shape)))
        else:
            return self.frozen_op


class _DiffusionCoeffAnisotropic(_DiffusionCoefficient):
    """
    Abstract class to define anisotropic diffusion tensors, where by isotropic we mean that the resulting diffusion tensors are not diagonal.
    This class only considers anisotropic diffusion tensors based obtained as functions of the structur tensor, following Weickert.
    Daughter classes of DiffusionTensorAnisotropic only need to implement the method _compute_diffusivities(), which defines a rule to compute the
    diffusivities associated to the local eigenvectors of the structur tensor. These diffusivities define the smoothing behavior of the tensor.
    """

    def __init__(self, arg_shape: pyct.NDArrayShape, structure_tensor: pyct.OpT, trace_term: bool = False):
        """

        Parameters
        ----------
        arg_shape
        structure_tensor
        """
        super().__init__(arg_shape=arg_shape, isotropic=False, trace_term=trace_term)
        msg = "`structure_tensor.arg_shape`={} inconsistent with `arg_shape`={}.".format(
            structure_tensor.arg_shape, arg_shape
        )
        assert structure_tensor.arg_shape == arg_shape, msg
        self.structure_tensor = structure_tensor
        # compute the indices of the upper triangular structur tensor to be selected to assemble its full version
        full_matrix_indices = np.zeros((self.ndims, self.ndims), dtype=int)
        upper_matrix_index = 0
        for i in range(self.ndims):
            for j in range(self.ndims):
                if j >= i:
                    full_matrix_indices[i, j] = upper_matrix_index
                    upper_matrix_index += 1
                else:
                    full_matrix_indices[i, j] = full_matrix_indices[j, i]
        self.full_matrix_indices = full_matrix_indices.reshape(-1)

    # @pycrt.enforce_precision(i="arr")
    # how to enforce precision on tuple of outputs?
    def _eigendecompose_struct_tensor(self, arr: pyct.NDArray) -> (pyct.NDArray, pyct.NDArray):
        """
        Notes
        ----
        This function decomposes the structur tensor. For each pixel, the eigenvectors and associated eigenvalues are computed.

        If the structur tensor is not smoothed (smooth_sigma=0), then the decomposition could be computed more efficiently from the gradient without assembling
        the structur tensor. We also could, alternatively, assemble the diffusion tensor differently in apply with different _assemble_tensor, directly from structure_tensor.
        Current implementation does not leverage any of this.
        """
        xp = pycu.get_array_module(arr)
        # compute upper/lower triangular component of structure tensor
        structure_tensor = self.structure_tensor.apply(arr)
        # structure_tensor = self._structure_tensor.unravel(structure_tensor)
        # structure_tensor = structure_tensor.reshape(structure_tensor.shape[0],-1)
        # xp.zeros((self.ndims ** 2, structure_tensor.shape[1]))
        #
        structure_tensor = self.structure_tensor.unravel(structure_tensor).squeeze()
        structure_tensor = structure_tensor.reshape(structure_tensor.shape[0], -1).T
        # assemble full structure tensor
        structure_tensor_full = structure_tensor[:, self.full_matrix_indices].reshape(-1, self.ndims, self.ndims)
        # eigendecompose tensor. numpy calls eigh behind the curtains,
        # dask has only svd, cupy both but svd does not call eigh behind the curtains.
        u, e, _ = xp.linalg.svd(structure_tensor_full, hermitian=True)
        return u, e

    @pycrt.enforce_precision(i="eigval_struct")
    def _compute_diffusivities(self, eigval_struct: pyct.NDArray) -> pyct.NDArray:
        raise NotImplementedError

    @pycrt.enforce_precision(i=("u", "lambdas"))
    def _assemble_tensors(self, u: pyct.NDArray, lambdas: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(u)
        diffusion_tensors = xp.zeros((self.size, len(self.arg_shape), len(self.arg_shape)))
        for i in range(len(self.arg_shape)):
            diffusion_tensors += lambdas[:, i].reshape(-1, 1, 1) * (
                u[:, :, i].reshape(self.size, -1, len(self.arg_shape))
                * u[:, :, i].reshape(self.size, len(self.arg_shape), -1)
            )
        return diffusion_tensors

    # @pycrt.enforce_precision(i="arr")
    # @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.OpT:
        """
        Notes
        -------
        """
        if not self.frozen:
            u, e = self._eigendecompose_struct_tensor(arr)
            lambdas = self._compute_diffusivities(e)
            tensors = self._assemble_tensors(u, lambdas)
            # assemble block operator.
            ops = []
            for i in range(self.ndims):
                ops.append([])
                # if trace_term, only upper diagonal entries are considered
                first_idx = i if self.trace_term else 0
                for j in np.arange(first_idx, self.ndims):
                    diag_op = pybase.DiagonalOp(tensors[:, i, j])
                    diag_op *= self._coeff_op[i, j]
                    ops[i].append(pybase.DiagonalOp(tensors[:, i, j]))
                ops[i] = pyblock.hstack(ops[i])
            if self.trace_term:
                return pyblock.hstack(ops)
            else:
                return pyblock.vstack(ops)
        else:
            return self.frozen_op


class DiffusionCoeffAnisoEdgeEnhancing(_DiffusionCoeffAnisotropic):
    """
    Edge-enhancing version of the anisotropic tensor.
    """

    def __init__(self, arg_shape: pyct.NDArrayShape, structure_tensor: pyct.OpT, trace_term: bool = False, beta=1, m=4):
        """

        Parameters
        ----------
        beta
        m
        """
        super().__init__(arg_shape=arg_shape, structure_tensor=structure_tensor, trace_term=trace_term)
        self.beta = beta
        self.m = 4
        self.bounded = True

        # compute normalization constant c
        def f(c: pyct.Real, m: pyct.Integer):
            return 1 - np.exp(-c) * (1 + 2 * m * c)

        self.c = sciop.brentq(functools.partial(f, m), 1e-2, 100)

    @pycrt.enforce_precision(i="eigval_struct")
    def _compute_diffusivities(self, eigval_struct: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(eigval_struct)
        lambdas = xp.ones(eigval_struct.shape, dtype=eigval_struct.dtype)
        nonzero_contrast_locs = ~xp.isclose(eigval_struct[:, 0], 0)
        # Inplace implementation of
        #   lambdas[nonzero_contrast_locs, 0] = 1 - xp.exp(- self.c / ((eigval_struct[nonzero_contrast_locs, 0] / self.beta) ** self.m))
        lambda1_nonzerolocs = eigval_struct[nonzero_contrast_locs, 0]
        lambda1_nonzerolocs /= self.beta
        lambda1_nonzerolocs **= -self.m
        lambda1_nonzerolocs *= -self.c
        lambda1_nonzerolocs = -xp.exp(lambda1_nonzerolocs)
        lambda1_nonzerolocs += 1
        lambdas[nonzero_contrast_locs, 0] = lambda1_nonzerolocs
        lambdas[:, 1] = 1
        return lambdas


class DiffusionCoeffAnisoCoherenceEnhancing(_DiffusionCoeffAnisotropic):
    """
    Coherence-enhancing version of the anisotropic tensor.
    """

    def __init__(
        self, arg_shape: pyct.NDArrayShape, structure_tensor: pyct.OpT, trace_term: bool = False, alpha=0.001, m=1
    ):
        super().__init__(arg_shape=arg_shape, structure_tensor=structure_tensor, trace_term=trace_term)
        self.alpha = alpha
        self.m = m
        self.bounded = True

    @pycrt.enforce_precision(i="eigval_struct")
    def _compute_diffusivities(self, eigval_struct: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(eigval_struct)
        coherence = (eigval_struct[:, 0] - eigval_struct[:, 1]) ** 2
        lambdas = self.alpha * xp.ones(eigval_struct.shape, dtype=eigval_struct.dtype)
        nonzero_coherence_locs = ~np.isclose(coherence, 0)
        # Inplace implementation of
        #   lambdas[nonzero_coherence_locs, 1] = self.alpha + (1-self.alpha)*np.exp(-1./(coherence[nonzero_coherence_locs] ** (2*self.m)))
        lambda2_nonzerolocs = coherence[nonzero_coherence_locs]
        lambda2_nonzerolocs **= -(2 * self.m)
        lambda2_nonzerolocs *= -1
        lambda2_nonzerolocs = xp.exp(lambda2_nonzerolocs)
        lambda2_nonzerolocs *= 1 - self.alpha
        lambda2_nonzerolocs += self.alpha
        lambdas[nonzero_coherence_locs, 1] = lambda2_nonzerolocs
        return lambdas


class _DiffusionOp(pyca.ProxDiffFunc):
    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        gradient: pyct.OpT = None,
        hessian: pyct.OpT = None,
        outer_diffusivity: pyct.OpT = None,
        diffusion_coefficient: pyct.OpT = None,
        balloon_force: pyct.OpT = None,
        outer_trace_diffusivity: pyct.OpT = None,
        trace_diffusion_coefficient: pyct.OpT = None,
        curvature_preservation_field: pyct.NDArray = np.zeros(0),
        prox_sigma: pyct.Real = 2,
    ):
        """

        Parameters
        ----------
        arg_shape: pyct.Shape
            Shape of the pixelised image.
        diffusion_coefficient: pyct.OpT
            DiffusionTensor operator, corresponding to an isotropic/anisotropic diffusion tensor. If None is specified,
            defaults to isotropic homogeneous diffusion with intensity 1 (Tikhonov)
        prox_sigma: pyct.Real
            Size of the structures that should be smoothed out by prox computation (when implemented in PnP-fashion).
            With current implementation, size depends on the size of the image (computational domain) itself,
            it is not in arbitrary units. Defaults to 0.05. Not true! We changed it, fix. Now "normalized" sigma
        """
        self.arg_shape = arg_shape
        self.size = int(np.prod(arg_shape))
        self.ndims = len(self.arg_shape)
        super().__init__(shape=(1, self.size))
        (
            gradient,
            hessian,
            diffusion_coefficient,
            trace_diffusion_coefficient,
            curvature_preservation_field,
            sampling,
        ) = self._sanitize_init_args(
            arg_shape=arg_shape,
            gradient=gradient,
            hessian=hessian,
            outer_diffusivity=outer_diffusivity,
            diffusion_coefficient=diffusion_coefficient,
            balloon_force=balloon_force,
            outer_trace_diffusivity=outer_trace_diffusivity,
            trace_diffusion_coefficient=trace_diffusion_coefficient,
            curvature_preservation_field=curvature_preservation_field,
            prox_sigma=prox_sigma,
        )
        self.outer_diffusivity = outer_diffusivity
        self.diffusion_coefficient = diffusion_coefficient
        self.balloon_force = balloon_force
        self.outer_trace_diffusivity = outer_trace_diffusivity
        self.trace_diffusion_coefficient = trace_diffusion_coefficient
        self.curvature_preservation_field = curvature_preservation_field
        if curvature_preservation_field.size > 0:
            # compute jacobian of the field and apply it to field itself
            self.jacobian = gradient(curvature_preservation_field)
            ops = []
            for i in range(self.ndims):
                vec = 0
                for j in range(self.ndims):
                    vec += self.jacobian[i, self.size * j : self.size * (j + 1)] * curvature_preservation_field[j, :]
                ops.append(pybase.DiagonalOp(vec))
            self._jacobian_onto_field = pyblock.hstack(ops)
        # assess whether diffusion operator descends from a potential formulation or not
        if self.diffusion_coefficient:
            self.from_potential = (
                self.diffusion_coefficient.from_potential
                * (self.outer_diffusivity is None)
                * (self.balloon_force is None)
                * (self.outer_trace_diffusivity is None)
                * (self.trace_diffusion_coefficient is None)
                * (self.curvature_preservation_field is None)
            )
        self.sampling = sampling
        self.gradient = gradient
        self.hessian = hessian
        # estimate number of prox steps necessary to smooth structures of size prox_sigma (linear diffusion analogy)
        self.prox_sigma = prox_sigma
        # Estimated t_final and prox_steps for prox evaluation, referring to the sampling=1 case.
        # In the prox computation, the actual sampling is taken into account to determine the time step size.
        t_final = self.prox_sigma**2 / 2
        self.time_step = 1.0 / (2**self.ndims)
        self.prox_steps = t_final / self.time_step
        # set lipschitz and diff_lipschitz to np.inf
        # lipschitz: think further, when apply exists we may have bounds on it. not crucial.
        self._lipschitz = np.inf
        self._diff_lipschitz = np.inf

    def _sanitize_init_args(
        self,
        arg_shape: pyct.NDArrayShape,
        gradient: pyct.OpT,
        hessian: pyct.OpT,
        outer_diffusivity: pyct.OpT,
        diffusion_coefficient: pyct.OpT,
        balloon_force: pyct.OpT,
        outer_trace_diffusivity: pyct.OpT,
        trace_diffusion_coefficient: pyct.OpT,
        curvature_preservation_field: pyct.NDArray,
        prox_sigma: pyct.Real,
    ):
        if hessian:
            nb_upper_entries = round(self.ndims * (self.ndims + 1) / 2)
            expected_codim = nb_upper_entries * self.size
            assert hessian.codim() == expected_codim, '`hessian` expected to be initialized with `directions`="all"'

        if outer_diffusivity and not diffusion_coefficient:
            raise ValueError("Cannot provide `outer_diffusivity` without providing `diffusion_coefficient`.")

        if outer_trace_diffusivity and not trace_diffusion_coefficient:
            raise ValueError(
                "Cannot provide `outer_trace_diffusivity` without providing `trace_diffusion_coefficient`."
            )

        if (
            (not diffusion_coefficient)
            * (not balloon_force)
            * (not trace_diffusion_coefficient)
            * (curvature_preservation_field.size == 0)
        ):
            msg = "\n".join(
                [
                    "Cannot instantiate the diffusion operator. Pass at least one of the following:",
                    "`diffusion_coefficient`, `balloon_force`, `trace_diffusion_coefficient`, `curvature_preservation_field`.",
                ]
            )
            raise ValueError(msg)

        if diffusion_coefficient and not gradient:
            msg = "\n".join(
                [
                    "No`gradient` was passed, needed for divergence term involving `diffusion_coefficient`.",
                    "Initializing a forward finite difference operator with unitary sampling as default.",
                ]
            )
            warnings.warn(msg)
            gradient = pydiff.Gradient.finite_difference(
                arg_shape=arg_shape, mode="reflect", sampling=1.0, diff_type="central"
            )

        if curvature_preservation_field.size > 0 and not gradient:
            msg = "\n".join(
                [
                    "No `gradient` was passed, needed for term involving `curvature_preservation_field`.",
                    "Initializing a central finite difference operator with unitary sampling as default.",
                ]
            )
            warnings.warn(msg)
            gradient = pydiff.Gradient.finite_difference(
                arg_shape=arg_shape, mode="edge", sampling=1.0, diff_type="central"
            )

        if trace_diffusion_coefficient and not hessian:
            msg = "\n".join(
                [
                    "No `hessian` was passed, needed for trace term involving `trace_diffusion_coefficient`.",
                    "Initializing a central finite difference operator with unitary sampling as default.",
                ]
            )
            warnings.warn(msg)
            hessian = pydiff.Hessian.finite_difference(
                arg_shape=arg_shape, mode="reflect", sampling=1.0, diff_type="central", accuracy=2
            )

        if diffusion_coefficient and not diffusion_coefficient.trace_term:
            if not diffusion_coefficient.frozen:
                warnings.warn("`diffusion_coefficient.trace_term` set to False. Modifying to True.")
                diffusion_coefficient.trace_term = True
            else:
                msg = "\n".join(
                    [
                        "`diffusion_coefficient.trace_term` set to False while `diffusion_coefficient.frozen` set to True.",
                        "Issues are expected. Initialize correctly `diffusion_coefficient.trace_term` to True before freezing.",
                    ]
                )
                raise ValueError(msg)

        if trace_diffusion_coefficient and not trace_diffusion_coefficient.trace_term:
            if not trace_diffusion_coefficient.frozen:
                warnings.warn("`trace_diffusion_coefficient.trace_term` set to False. Modifying to True.")
                trace_diffusion_coefficient.trace_term = True
            else:
                msg = "\n".join(
                    [
                        "`trace_diffusion_coefficient.trace_term` set to False while `trace_diffusion_coefficient.frozen` set to True.",
                        "Issues are expected. Initialize correctly `trace_diffusion_coefficient.trace_term` to True before freezing.",
                    ]
                )
                raise ValueError(msg)

        if curvature_preservation_field.size > 0:
            if curvature_preservation_field.shape != (self.ndims, self.size):
                msg = "\n".join(
                    [
                        "Unexpected shape {} of `curvature_preservation_field`,"
                        "expected ({}, {}).".format(curvature_preservation_field.shape, self.ndims, self.size),
                    ]
                )
                raise ValueError(msg)

        # check arg_shapes consistency
        _to_be_checked = {
            "outer_diffusivity": outer_diffusivity,
            "diffusion_coefficient": diffusion_coefficient,
            "balloon_force": balloon_force,
            "outer_trace_diffusivity": outer_trace_diffusivity,
            "trace_diffusion_coefficient": trace_diffusion_coefficient,
            "gradient": gradient,
            "hessian": hessian,
        }
        for i in _to_be_checked:
            if _to_be_checked[i]:
                msg = "`{}.arg_shape`=({}) inconsistent with `arg_shape`={}.".format(
                    i, _to_be_checked[i].arg_shape, arg_shape
                )
                assert _to_be_checked[i].arg_shape == arg_shape, msg

        # check sampling consistency
        _to_be_checked = {}
        if gradient:
            _to_be_checked["`gradient`"] = gradient.sampling
        if hessian:
            _to_be_checked["`hessian`"] = hessian.sampling
        if balloon_force:
            if balloon_force.gradient:
                _to_be_checked["`balloon_force.gradient`"] = balloon_force.gradient.sampling
        if outer_diffusivity:
            if outer_diffusivity.gradient:
                _to_be_checked["`outer_diffusivity.gradient`"] = outer_diffusivity.gradient.sampling
        if outer_trace_diffusivity:
            if outer_trace_diffusivity.gradient:
                _to_be_checked["`outer_trace_diffusivity.gradient`"] = outer_trace_diffusivity.gradient.sampling
        if diffusion_coefficient:
            if diffusion_coefficient.isotropic:
                if diffusion_coefficient.diffusivity.gradient:
                    _to_be_checked[
                        "`diffusion_coefficient.diffusivity.gradient`"
                    ] = diffusion_coefficient.diffusivity.gradient.sampling
            else:
                if diffusion_coefficient.structure_tensor:
                    _to_be_checked[
                        "`diffusion_coefficient.structure_tensor.gradient`"
                    ] = diffusion_coefficient.structure_tensor.grad.sampling
        if trace_diffusion_coefficient:
            if trace_diffusion_coefficient.isotropic:
                if trace_diffusion_coefficient.diffusivity.gradient:
                    _to_be_checked[
                        "`trace_diffusion_coefficient.diffusivity.gradient`"
                    ] = trace_diffusion_coefficient.diffusivity.gradient.sampling
            else:
                if trace_diffusion_coefficient.structure_tensor:
                    _to_be_checked[
                        "`trace_diffusion_coefficient.structure_tensor.gradient`"
                    ] = trace_diffusion_coefficient.structure_tensor.grad.sampling
        if _to_be_checked:
            s_base = list(_to_be_checked.values())[0]
            op_base = list(_to_be_checked.keys())[0]
            for s in _to_be_checked:
                assert (
                    _to_be_checked[s] == s_base
                ), "Inconsistent `sampling` for differential operators {} and {}.".format(op_base, s)
            sampling = s_base
        else:
            sampling = None

        assert prox_sigma > 0.0, "`prox_sigma` must be strictly positive."

        # if trace_diffusion_coefficient is isotropic,
        # convert hessian to second derivative operator
        if trace_diffusion_coefficient:
            if trace_diffusion_coefficient.isotropic:
                ops = []
                for dim in np.range(self.ndims):
                    # select second order derivative operators
                    ops.append(hessian._block[(dim, 0)])
                hessian = pyblock.hstack(ops)

        # returning only objects that might have been modified.
        return (
            gradient,
            hessian,
            diffusion_coefficient,
            trace_diffusion_coefficient,
            curvature_preservation_field,
            sampling,
        )

    def asloss(self, data: pyct.NDArray = None) -> NotImplemented:
        """
        Notes
        -------
        DivergenceDiffusionOp class is not meant to be used to define a loss functional.
        """
        return NotImplemented

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> NotImplemented:
        """
        Notes
        -------
        Divergence-based diffusion operators may arise from a variational formulation. This is true, for example
        for the isotropic Perona-Malik, TV, Tikhonov and extrinsic diffusivities. For these cases, it is possible
        to define the associated energy functional. When this does not hold, method raises an error.
        """
        if self.from_potential:
            return self.diffusion_coefficient.diffusivity.energy_potential(arr, self.gradient)
        else:
            msg = "\n".join(
                [
                    "DivergenceDiffusionOp not found to be arising from an energy potential formulation.",
                    "If it is, define how to evaluate the associated energy functional.",
                ]
            )
            raise NotImplementedError(msg)

    @pycrt.enforce_precision(i="arr")
    def _compute_divergence_term(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y_div = xp.zeros(arr.shape, dtype=arr.dtype)
        if self.diffusion_coefficient or self.outer_diffusivity:
            y_div = self.gradient(arr)
            if self.diffusion_coefficient:
                # compute flux
                diffusion_coefficient = self.diffusion_coefficient(arr)
                y_div = diffusion_coefficient(y_div)
                # apply divergence
                y_div = self.gradient.T(y_div)
            if self.outer_diffusivity:
                outer_diffusivity = self.outer_diffusivity(arr)
                # rescale divergence
                y_div *= outer_diffusivity
        return y_div

    @pycrt.enforce_precision(i="arr")
    def _compute_balloon_term(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        balloon_force = xp.zeros(arr.shape, dtype=arr.dtype)
        if self.balloon_force:
            balloon_force = self.balloon_force(arr)
        return -balloon_force

    @pycrt.enforce_precision(i="arr")
    def _compute_trace_term(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y_trace = xp.zeros(arr.shape, dtype=arr.dtype)
        if self.trace_diffusion_coefficient:
            hessian = self.hessian.unravel(self.hessian(arr)).squeeze().reshape(1, -1)
            trace_tensor = self.trace_diffusion_coefficient(arr)
            y_trace = trace_tensor(hessian)
            if self.outer_trace_diffusivity:
                outer_trace_diffusivity = self.outer_trace_diffusivity(arr)
                # rescale trace
                y_trace *= outer_trace_diffusivity(arr)
        return -y_trace

    @pycrt.enforce_precision(i="arr")
    def _compute_curvature_preserving_term(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y_curv = xp.zeros(arr.shape, dtype=arr.dtype)
        if self.curvature_preservation_field.size > 0:
            grad_arr = self.gradient(arr)
            y_curv = self._jacobian_onto_field(grad_arr)
        return -y_curv

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        # to avoid @vectorize, all _compute fcts should be changed, suitably stacking
        # all operators and results along the stacking dimensions.
        # For now this has not been done, to be discussed if less naive vectorization
        # is important.
        xp = pycu.get_array_module(arr)
        arr = arr.reshape(1, -1)
        y = xp.zeros(arr.shape, dtype=arr.dtype)
        # compute divergence term
        y += self._compute_divergence_term(arr)
        # compute balloon force term
        y += self._compute_balloon_term(arr)
        # compute trace tensor term
        y += self._compute_trace_term(arr)
        # compute curvature preserving term
        y += self._compute_curvature_preserving_term(arr)
        return y.reshape(self.arg_shape)

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        # Actual prox would be \argmin_{y}1/tau*(arr-y)+self.grad(y).
        # Current implementation, instead, corresponds to a diffusion stopped at given time,
        # achievable applying a number prox_steps of self.grad steps to the initial state arr.
        stop_crit = pystop.MaxIter(self.prox_steps)
        pgd = pysol.PGD(f=self, g=None, show_progress=False, verbosity=100)
        pgd.fit(**dict(mode=pysolver.Mode.BLOCK, x0=arr, stop_crit=stop_crit, acceleration=False))
        return pgd.solution()


class DivergenceDiffusionOp(_DiffusionOp):
    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        gradient: pyct.OpT = None,
        outer_diffusivity: pyct.OpT = None,
        diffusion_coefficient: pyct.OpT = None,
        prox_sigma: pyct.Real = 2,
    ):
        super().__init__(
            arg_shape=arg_shape,
            gradient=gradient,
            outer_diffusivity=outer_diffusivity,
            diffusion_coefficient=diffusion_coefficient,
            prox_sigma=prox_sigma,
        )
        # estimate diff_lipschitz
        _known_diff_lipschitz = False
        if diffusion_coefficient:
            if diffusion_coefficient.bounded:
                _known_diff_lipschitz = True
                if not diffusion_coefficient.isotropic:
                    # extra factor 2 in this case for exact expression?
                    msg = "For anisotropic `diffusion_coefficient`, the estimated `diff_lipschitz` experimentally grants stability but is not guaranteed to hold."
                    warnings.warn(msg)
            if outer_diffusivity:
                _known_diff_lipschitz = _known_diff_lipschitz and outer_diffusivity.bounded
        if _known_diff_lipschitz:
            self._diff_lipschitz = gradient.lipschitz() ** 2

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr = arr.reshape(1, -1)
        y = xp.zeros(arr.shape, dtype=arr.dtype)
        # compute divergence term
        y += self._compute_divergence_term(arr)
        # y=y.reshape(self.arg_shape)
        return y


class SnakeDiffusionOp(_DiffusionOp):
    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        gradient: pyct.OpT = None,
        outer_diffusivity: pyct.OpT = None,
        diffusion_coefficient: pyct.OpT = None,
        balloon_force: pyct.OpT = None,
        prox_sigma: pyct.Real = 2,
    ):
        super().__init__(
            arg_shape=arg_shape,
            gradient=gradient,
            outer_diffusivity=outer_diffusivity,
            diffusion_coefficient=diffusion_coefficient,
            balloon_force=balloon_force,
            prox_sigma=prox_sigma,
        )
        # estimate diff_lipschitz
        _known_diff_lipschitz = False
        if diffusion_coefficient:
            if diffusion_coefficient.bounded:
                _known_diff_lipschitz = True
                if not diffusion_coefficient.isotropic:
                    # extra factor 2 in this case for exact expression?
                    msg = "For anisotropic `diffusion_coefficient`, the estimated `diff_lipschitz` experimentally grants stability but is not guaranteed to hold."
                    warnings.warn(msg)
            if outer_diffusivity:
                _known_diff_lipschitz = _known_diff_lipschitz and outer_diffusivity.bounded
        if _known_diff_lipschitz:
            self._diff_lipschitz = gradient.lipschitz() ** 2
        if balloon_force:
            self._diff_lipschitz += balloon_force._lipschitz

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr = arr.reshape(1, -1)
        y = xp.zeros(arr.shape, dtype=arr.dtype)
        # compute divergence term
        y += self._compute_divergence_term(arr)
        # compute balloon force term
        y += self._compute_balloon_term(arr)
        return y.reshape(self.arg_shape)


class TraceDiffusionOp(_DiffusionOp):
    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        hessian: pyct.OpT = None,
        outer_trace_diffusivity: pyct.OpT = None,
        trace_diffusion_coefficient: pyct.OpT = None,
        prox_sigma: pyct.Real = 2,
    ):
        super().__init__(
            arg_shape=arg_shape,
            outer_trace_diffusivity=outer_trace_diffusivity,
            trace_diffusion_coefficient=trace_diffusion_coefficient,
            hessian=hessian,
            prox_sigma=prox_sigma,
        )
        # estimate diff_lipschitz (further think, extra factors may arise for trace case)
        _known_diff_lipschitz = False
        if trace_diffusion_coefficient:
            if trace_diffusion_coefficient.bounded:
                _known_diff_lipschitz = True
                if not trace_diffusion_coefficient.isotropic:
                    msg = "For anisotropic `trace_diffusion_coefficient`, the estimated `diff_lipschitz` experimentally grants stability but is not guaranteed to hold."
                    warnings.warn(msg)
            if outer_trace_diffusivity:
                _known_diff_lipschitz = _known_diff_lipschitz and outer_trace_diffusivity.bounded
        if _known_diff_lipschitz:
            self._diff_lipschitz = hessian.lipschitz()

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr = arr.reshape(1, -1)
        y = xp.zeros(arr.shape, dtype=arr.dtype)
        # compute trace tensor term
        y += self._compute_trace_term(arr)
        return y.reshape(self.arg_shape)


class CurvaturePreservingDiffusionOp(_DiffusionOp):
    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        gradient: pyct.OpT = None,
        hessian: pyct.OpT = None,
        curvature_preservation_field: pyct.NDArray = np.zeros(0),
        prox_sigma: pyct.Real = 2,
    ):
        if not hessian:
            msg = "\n".join(
                [
                    "No `hessian` was passed, needed for trace term involving `trace_diffusion_coefficient`.",
                    "Initializing a central finite difference operator with unitary sampling as default.",
                ]
            )
            warnings.warn(msg)
            hessian = pydiff.Hessian.finite_difference(
                arg_shape=arg_shape, mode="reflect", sampling=1.0, diff_type="central", accuracy=2
            )
        super().__init__(
            arg_shape=arg_shape,
            curvature_preservation_field=curvature_preservation_field,
            gradient=gradient,
            hessian=hessian,
            prox_sigma=prox_sigma,
        )
        curvature_preservation_field = curvature_preservation_field.T
        tensors = curvature_preservation_field.reshape(self.size, self.ndims, 1) * curvature_preservation_field.reshape(
            self.size, 1, self.ndims
        )
        trace_diffusion_coefficient = tensors.reshape(self.size, -1)
        ops = []
        for i in range(self.ndims):
            for j in range(self.ndims):
                ops.append(pybase.DiagonalOp(trace_diffusion_coefficient[:, i * self.ndims + j]))
        self.trace_diffusion_coefficient = _DiffusionCoefficient(arg_shape=arg_shape, isotropic=False, trace_term=True)
        self.trace_diffusion_coefficient.set_frozen_op(pyblock.hstack(ops))
        self._diff_lipschitz = hessian.lipschitz()
        if self.curvature_preservation_field.size > 0:
            max_norm = np.max(np.linalg.norm(curvature_preservation_field, axis=1))
            self._diff_lipschitz *= max_norm
            # abs(<gradient(u), J_w(w)>) \leq norm(gradient(u)) * norm(J_w(w))
            # \leq L_grad*norm(u)*2*L_grad*(norm(w)**2) = 2*L_grad**2 * norm(u)
            self._diff_lipschitz += 2 * (gradient.lipschitz() ** 2) * max_norm

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr = arr.reshape(1, -1)
        y = xp.zeros(arr.shape, dtype=arr.dtype)
        # compute trace tensor term
        y += self._compute_trace_term(arr)
        # compute curvature preserving term
        y += self._compute_curvature_preserving_term(arr)
        return y.reshape(self.arg_shape)
