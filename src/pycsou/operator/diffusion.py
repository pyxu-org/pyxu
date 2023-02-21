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


class _DiffusionProcessElement(pyca.Func):
    """
    Abstract class to define diffusivity functions.
    The class features a method energy_functional(), which can be used to evaluate the energy potential that a
    divergence-based diffusion featuring the considered diffusivity derives from (when it makes sense).
    When implementing a new diffusivity, one needs to define whether it allows a variational interpretation or not.
    """

    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None):
        self.arg_shape = arg_shape
        self.size = int(np.prod(arg_shape))
        super().__init__(shape=(1, self.size))
        self.grad = gradient
        if gradient:
            msg = "_DiffusionProcessElement: gradient has inconsistent arg_shape."
            assert gradient.arg_shape == arg_shape, msg
        self.frozen = False
        self.frozen_element = None

    def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
        pass

    def freeze_element(self, arr: pyct.NDArray):
        if self.frozen:
            warnings.warn("Element has already been frozen. Cannot overwrite previous frozen state.")
        else:
            self.frozen_element = self.apply(arr)
            self.frozen = True

    def set_frozen_element(self, frozen_element: pyct.NDArray):
        if self.frozen:
            warnings.warn("Element has already been frozen. Cannot overwrite previous frozen state.")
        else:
            self.frozen_element = frozen_element
            self.frozen = True

    @pycrt.enforce_precision(i="arr")
    def _compute_grad_norm_sq(self, arr: pyct.NDArray, grad: pyct.OpT = None):
        xp = pycu.get_array_module(arr)
        grad_arr = grad.unravel(grad(arr))
        grad_arr **= 2
        axis = 0 + 1 * (arr.shape[0] > 1)
        return xp.sum(grad_arr, axis=axis, keepdims=True)

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        raise NotImplementedError


class DilationBalloonForce(_DiffusionProcessElement):
    # disclaimer: very small steps may be required for stability
    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None):
        super().__init__(arg_shape=arg_shape, gradient=gradient)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        grad_norm_sq = self._compute_grad_norm_sq(arr, self.grad)
        return xp.sqrt(grad_norm_sq)


class MCMouterDiffusivity(_DiffusionProcessElement):
    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None):
        self = DilationBalloonForce(arg_shape=arg_shape, gradient=gradient)


class ErosionBalloonForce(_DiffusionProcessElement):
    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None):
        self = -DilationBalloonForce(arg_shape=arg_shape, gradient=gradient)


class _Diffusivity(_DiffusionProcessElement):
    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None):
        super().__init__(arg_shape=arg_shape, gradient=gradient)
        self.from_potential = False

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        raise NotImplementedError

    def energy_potential(self, arr: pyct.NDArray, grad: pyct.OpT = None) -> pyct.NDArray:
        return NotImplemented


class TikhonovDiffusivity(_Diffusivity):
    def __init__(self, arg_shape: pyct.NDArrayShape):
        super().__init__(arg_shape=arg_shape)
        self.from_potential = True

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
    def __init__(self, arg_shape: pyct.NDArrayShape):
        super().__init__(arg_shape=arg_shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.clip(arr, 0, None)
        y += 1
        return 1 / y


class PeronaMalikExponentialDiffusivity(_Diffusivity):
    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None, beta: pyct.Real = 1):
        super().__init__(arg_shape=arg_shape, gradient=gradient)
        self.beta = beta
        self.from_potential = True

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        # Inplace implementation of
        #   xp.exp(-grad_norm_sq/beta**2)
        y = self._compute_grad_norm_sq(arr, self.grad)
        y /= self.beta**2
        return xp.exp(-y)

    @pycrt.enforce_precision(i="arr")
    def energy_functional(self, arr: pyct.NDArray, grad: pyct.OpT = None) -> pyct.NDArray:
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


class PeronaMalikRationalDiffusivity(_Diffusivity):
    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None, beta: pyct.Real = 1):
        super().__init__(arg_shape=arg_shape, gradient=gradient)
        self.beta = beta
        self.from_potential = True

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        # Inplace implementation of
        #   1 / (1 + grad_norm_sq/beta**2)
        y = self._compute_grad_norm_sq(arr, self.grad)
        y /= self.beta**2
        y += 1
        return 1 / y

    @pycrt.enforce_precision(i="arr")
    def energy_functional(self, arr: pyct.NDArray, grad: pyct.OpT = None) -> pyct.NDArray:
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


class TameTotalVariationDiffusivity(_Diffusivity):
    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None, beta: pyct.Real = 1):
        super().__init__(arg_shape=arg_shape, gradient=gradient)
        self.beta = beta
        self.from_potential = True

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        # Inplace implementation of
        #   1/(xp.sqrt(1+grad_norm_sq/beta**2)
        y = self._compute_grad_norm_sq(arr, self.grad)
        y /= self.beta**2
        y += 1
        y = xp.sqrt(y)
        return 1 / y

    @pycrt.enforce_precision(i="arr")
    def energy_functional(self, arr: pyct.NDArray, grad: pyct.OpT = None) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = self._compute_grad_norm_sq(arr, grad)
        y += 1
        y = xp.sqrt(y)
        return xp.sum(y, axis=-1)


class TotalVariationDiffusivity(_Diffusivity):
    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None, beta: pyct.Real = 1):
        super().__init__(arg_shape=arg_shape, gradient=gradient)
        self.beta = beta
        self.from_potential = True

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = self._compute_grad_norm_sq(arr, self.grad)
        return 1 / y

    @pycrt.enforce_precision(i="arr")
    def energy_functional(self, arr: pyct.NDArray, grad: pyct.OpT = None) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = self._compute_grad_norm_sq(arr, grad)
        y = xp.sqrt(y)
        return xp.sum(y, axis=-1)


class _DiffusionProcessTensor(pyca.Map):
    """
    Abstract class for diffusion tensors.
    The daughter classes DiffusionTensorIsotropic, DiffusionTensorAnisotropic allow to handle the isotropic/anisotropic cases.
    """

    def __init__(self, arg_shape: pyct.NDArrayShape, isotropic: bool = True, trace_term: bool = False):
        self.arg_shape = arg_shape
        self.size = int(np.prod(arg_shape))
        self.ndims = len(self.arg_shape)
        super().__init__(shape=(1, self.size))
        self._lipschitz = np.inf
        self.isotropic = isotropic
        self.from_potential = False
        self.frozen = False
        self.frozen_op = None
        self.trace_term = trace_term

    def freeze_tensor(self, arr: pyct.NDArray):
        if self.frozen:
            warnings.warn("Tensor has already been frozen. Cannot overwrite previous frozen state.")
        else:
            self.frozen_op = self.apply(arr)
            self.frozen = True

    def set_frozen_op(self, frozen_op: pyct.OpT):
        if self.frozen:
            warnings.warn("Tensor has already been frozen. Cannot overwrite previous frozen state.")
        else:
            self.frozen_op = frozen_op
            self.frozen = True

    def apply(self, arr: pyct.NDArray) -> pyct.OpT:
        raise NotImplementedError


class DiffusionProcessTensorIsotropic(_DiffusionProcessTensor):
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
            msg = "DiffusionProcessTensorIsotropic: diffusivity has inconsistent arg_shape."
            assert diffusivity.arg_shape == arg_shape, msg
            self.diffusivity = diffusivity
        self.from_potential = self.diffusivity.from_potential * (not trace_term)

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.OpT:
        if not self.frozen:
            xp = pycu.get_array_module(arr)
            y = self.diffusivity(arr)
            if self.trace_term:
                ops = pybase.DiagonalOp(y)
                ops *= self.ndims
                return pyblock.hstack(ops)
            else:
                return pybase.DiagonalOp(xp.tile(y, len(self.arg_shape)))
        else:
            return self.frozen_op


class _DiffusionProcessTensorAnisotropic(_DiffusionProcessTensor):
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
        msg = "_DiffusionProcessTensorAnisotropic: structure_tensor has inconsistent arg_shape."
        assert structure_tensor.arg_shape == arg_shape, msg
        self._structure_tensor = structure_tensor
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

    @pycrt.enforce_precision(i="arr")
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
        structure_tensor = self._structure_tensor.apply(arr)
        # structure_tensor = self._structure_tensor.unravel(structure_tensor)
        # structure_tensor = structure_tensor.reshape(structure_tensor.shape[0],-1)
        # xp.zeros((self.ndims ** 2, structure_tensor.shape[1]))
        #
        structure_tensor = self._structure_tensor.unravel(structure_tensor)
        structure_tensor = structure_tensor.reshape(structure_tensor.shape[0], -1).T
        # assemble full structure tensor
        structure_tensor_full = structure_tensor[:, self.full_matrix_indices].reshape(-1, self.ndims, self.ndims)
        # eigendecompose tensor
        u, e, _ = xp.linalg.svd(structure_tensor_full, hermitian=True)
        return u, e

    @pycrt.enforce_precision(i="eigval_struct")
    def _compute_diffusivities(self, eigval_struct: pyct.NDArray) -> pyct.NDArray:
        raise NotImplementedError

    @pycrt.enforce_precision(i=("u", "lambdas"))
    def _assemble_tensor(self, u: pyct.NDArray, lambdas: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(u)
        diffusion_tensor = xp.zeros((self.size, len(self.arg_shape), len(self.arg_shape)))
        for i in range(len(self.arg_shape)):
            diffusion_tensor += lambdas[:, i] * (
                u[:, :, i].reshape(self.size, -1, len(self.arg_shape))
                * u[:, :, i].reshape(self.size, len(self.arg_shape), -1)
            )
        return diffusion_tensor

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.OpT:
        """
        Notes
        -------
        When assembling the block operator, we could assemble only its upper/lower part (with diagonal entries rescaled by factor 2) and then return
        the operator block + block.T.
        More importantly, we could return a list of length self.ndims of operators, to avoid useless computations in trace based term of diffusion operators.
        """
        if not self.frozen:
            u, e = self._eigendecompose_struct_tensor(arr)
            lambdas = self._compute_diffusivities(e)
            tensor = self._assemble_tensor(u, lambdas)
            # assemble block operator.
            ops = []
            for i in range(self.ndims):
                ops.append([])
                for j in range(self.ndims):
                    ops[i].append(pybase.DiagonalOp(tensor[:, i * self.ndims + j]))
                ops[i] = pyblock.hstack(ops[i])
            if self.trace_term:
                return pyblock.hstack(ops)
            else:
                return pyblock.vstack(ops)
        else:
            return self.frozen_op


class DiffusionProcessTensorAnisotropicEdgeEnhancing(_DiffusionProcessTensorAnisotropic):
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


class DiffusionProcessTensorAnisotropicCoherenceEnhancing(_DiffusionProcessTensorAnisotropic):
    """
    Coherence-enhancing version of the anisotropic tensor.
    """

    def __init__(
        self, arg_shape: pyct.NDArrayShape, structure_tensor: pyct.OpT, trace_term: bool = False, alpha=0.001, m=1
    ):
        super().__init__(arg_shape=arg_shape, structure_tensor=structure_tensor, trace_term=trace_term)
        self.alpha = alpha
        self.m = m

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
        sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]],
        outer_diffusivity: pyct.OpT = None,
        diffusion_tensor: pyct.OpT = None,
        balloon_force: pyct.OpT = None,
        outer_trace_diffusivity: pyct.OpT = None,
        trace_local_smoothing_tensor: pyct.OpT = None,
        curvature_preservation_field: pyct.NDArray = None,
        gradient: pyct.OpT = None,
        hessian: pyct.OpT = None,
        prox_sigma: pyct.Real = 2,
    ):
        """

        Parameters
        ----------
        arg_shape: pyct.Shape
            Shape of the pixelised image.
        diffusion_tensor: pyct.OpT
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
            diffusion_tensor,
            trace_local_smoothing_tensor,
            curvature_preservation_field,
            gradient,
            hessian,
        ) = self._sanitize_init_args(
            arg_shape=arg_shape,
            outer_diffusivity=outer_diffusivity,
            diffusion_tensor=diffusion_tensor,
            balloon_force=balloon_force,
            outer_trace_diffusivity=outer_trace_diffusivity,
            trace_local_smoothing_tensor=trace_local_smoothing_tensor,
            curvature_preservation_field=curvature_preservation_field,
            gradient=gradient,
            hessian=hessian,
            prox_sigma=prox_sigma,
        )
        self.outer_diffusivity = outer_diffusivity
        self.diffusion_tensor = diffusion_tensor
        self.balloon_force = balloon_force
        self.outer_trace_diffusivity = outer_trace_diffusivity
        self.trace_local_smoothing_tensor = trace_local_smoothing_tensor
        self.curvature_preservation_field = curvature_preservation_field
        if trace_local_smoothing_tensor:
            # compute the indices of the upper triangular Hessian to be selected to assemble its full version
            full_hessian_indices = np.zeros((self.ndims, self.ndims), dtype=int)
            upper_hessian_index = 0
            for i in range(self.ndims):
                for j in range(self.ndims):
                    if j >= i:
                        full_hessian_indices[i, j] = upper_hessian_index
                        upper_hessian_index += 1
                    else:
                        full_hessian_indices[i, j] = full_hessian_indices[j, i]
            if trace_local_smoothing_tensor.isotropic:
                # if tensor is isotropic, select the diagonal entries (2nd derivatives)
                self.hessian_indices = np.diag(full_hessian_indices)
            else:
                # if tensor is anisotropic, unroll full hessian column by column (or, analogously, row by row)
                self.hessian_indices = full_hessian_indices.reshape(-1)
        if curvature_preservation_field:
            # compute jacobian of the field and apply it to field itself
            self.jacobian = self.grad(curvature_preservation_field)
            ops = []
            for i in range(self.ndims):
                vec = 0
                for j in range(self.ndims):
                    vec += self.jacobian[i, self.size * j : self.size * (j + 1)] * curvature_preservation_field[j, :]
                ops.append(pybase.DiagonalOp(vec))
            self._jacobian_onto_field = pyblock.hstack(ops)
        # assess whether diffusion operator descends from a potential formulation or not
        self.from_potential = (
            self.diffusion_tensor.from_potential
            * (self.outer_diffusivity is None)
            * (self.balloon_force is None)
            * (self.outer_trace_diffusivity is None)
            * (self.trace_local_smoothing_tensor is None)
            * (self.curvature_preservation_field is None)
        )
        self.sampling = sampling  # this will change, see comments below.
        warnings.warn(
            "Provided `sampling` is assumed to hold for all differential"
            "operators involved. Make sure this is the case."
        )
        condition = (
            diffusion_tensor
            and not outer_diffusivity
            and not trace_local_smoothing_tensor
            and not outer_trace_diffusivity
            and not balloon_force
            and not curvature_preservation_field
        )
        # and diffusion_tensor bounded!!
        if condition:
            # this is true for some cases. need to better investigate.
            # traces should not give issues either (even though, for hessians, different multiplicative factors may arise)
            self._diff_lipschitz = (2 ** (self.ndims - 1)) * 4 / (np.min(sampling) ** 2)
        else:
            self._diff_lipschitz = np.inf
        # self.sampling = (
        #    gradient.sampling if gradient else hessian.sampling)  # will not work for pure balloon force cases, think. Moreover, sampling not stored anywhere. We could pass it as input.
        self.grad = gradient
        self.hessian = hessian
        # estimate number of prox steps necessary to smooth structures of size prox_sigma (linear diffusion analogy)
        self.prox_sigma = prox_sigma
        # time_step = 1 / 2*self._diff_lipschitz
        # self.time_step = (np.min(self.sampling) ** 2 / 4)  # valid only for isotropic case with diffusivity bounded between 0 and 1 (and for anisotropic standard cases?)
        # The estimated final time and time step refer to the sampling=1 case. In the prox computation, the actual sampling parameter is
        # taken into account to determine the time step size.
        t_final = self.prox_sigma**2 / 2
        self.time_step = 1.0 / (2**self.ndims)
        self.prox_steps = t_final / self.time_step

    def _sanitize_init_args(
        self,
        arg_shape: pyct.NDArrayShape,
        outer_diffusivity: pyct.OpT,
        diffusion_tensor: pyct.OpT,
        balloon_force: pyct.OpT,
        outer_trace_diffusivity: pyct.OpT,
        trace_local_smoothing_tensor: pyct.OpT,
        curvature_preservation_field: pyct.NDArray,
        gradient: pyct.OpT,
        hessian: pyct.OpT,
        prox_sigma: pyct.Real,
    ):
        if outer_diffusivity and not diffusion_tensor:
            raise ValueError("Cannot provide an outer_diffusivity without providing a diffusion_tensor.")

        if outer_trace_diffusivity and not trace_local_smoothing_tensor:
            raise ValueError(
                "Cannot provide an outer_trace_diffusivity without providing a trace_local_smoothing_tensor."
            )

        if diffusion_tensor and not gradient:
            msg = "\n".join(
                [
                    "No gradient was passed, needed for divergence term involving diffusion_tensor.",
                    "Initializing a forward finite difference operator with unitary sampling as default.",
                ]
            )
            warnings.warn(msg)
            gradient = pydiff.Gradient.finite_difference(
                arg_shape=arg_shape, mode="reflect", sampling=1.0, diff_type="central"
            )

        if curvature_preservation_field and not gradient:
            msg = "\n".join(
                [
                    "No gradient was passed, needed for term involving curvature_preservation_field.",
                    "Initializing a forward finite difference operator with unitary sampling as default.",
                ]
            )
            warnings.warn(msg)
            gradient = pydiff.Gradient.finite_difference(
                arg_shape=arg_shape, mode="reflect", sampling=1.0, diff_type="central"
            )

        if trace_local_smoothing_tensor and not hessian:
            msg = "\n".join(
                [
                    "No hessian was passed, needed for trace term involving trace_local_smoothing_tensor.",
                    "Initializing a forward finite difference operator with unitary sampling as default.",
                ]
            )
            warnings.warn(msg)
            hessian = pydiff.Hessian.finite_difference(
                arg_shape=arg_shape, mode="reflect", sampling=1.0, diff_type="central"
            )

        if trace_local_smoothing_tensor and not trace_local_smoothing_tensor.trace_term:
            if not trace_local_smoothing_tensor.frozen:
                warnings.warn("Trace_local_smoothing_tensor not initialized as trace_term. Modifying the object.")
            else:
                msg = "\n".join(
                    [
                        "Trace_local_smoothing_tensor not initialized as trace_term and set to a frozen state.",
                        "Issues are expected. Initialize correctly setting trace_term to True before freezing.",
                    ]
                )
                raise ValueError(msg)

        if trace_local_smoothing_tensor and not trace_local_smoothing_tensor.trace_term:
            if not trace_local_smoothing_tensor.frozen:
                warnings.warn("Trace_local_smoothing_tensor not initialized as trace_term. Modifying the object.")
            else:
                msg = "\n".join(
                    [
                        "Trace_local_smoothing_tensor not initialized as trace_term and set to a frozen state.",
                        "Issues are expected. Initialize correctly setting trace_term to True before freezing.",
                    ]
                )
                raise ValueError(msg)

        if curvature_preservation_field:
            if curvature_preservation_field.shape is not (self.ndims, self.size):
                raise ValueError(
                    "Unexpected shape {} of curvature_preservation_field.".format(curvature_preservation_field.shape)
                )
            norm = np.linalg.norm(curvature_preservation_field, axis=0)
            if not np.allclose(norm, 1):
                curvature_preservation_field /= norm
                warnings.warn("Unnormalized vectors detected in curvature_preservation_field. Normalizing to 1.")

        # check arg_shapes
        _to_be_checked = {
            "outer_diffusivity": outer_diffusivity,
            "diffusion_tensor": diffusion_tensor,
            "balloon_force": balloon_force,
            "outer_trace_diffusivity": outer_trace_diffusivity,
            "trace_local_smoothing_tensor": trace_local_smoothing_tensor,
            "gradient": gradient,
            "hessian": hessian,
        }
        for i in _to_be_checked:
            if _to_be_checked[i]:
                assert _to_be_checked[i].arg_shape == arg_shape, "Inconsistent arg_shape for {}".format(i)

        """
        currently sampling is not stored in differential operators. Add, to retrieve and check easily. For now, warning at initialization.
        #check samplings
        _to_be_checked = {"outer_diffusivity": outer_diffusivity, "diffusion_tensor": diffusion_tensor,
                          "balloon_force": balloon_force, "outer_trace_diffusivity": outer_trace_diffusivity,
                          "trace_local_smoothing_tensor": trace_local_smoothing_tensor,
                          "gradient": gradient, "hessian": hessian}
        if diffusion_tensor:
            if diffusion_tensor.isotropic:
                if diffusion_tensor.diffusivity.grad:
                    _to_be_checked["diffusion_tensor.diffusivity.grad":diffusion_tensor.diffusivity.grad]
        for i in _to_be_checked:
            if _to_be_checked[i]:
                assert _to_be_checked[i].sampling == sampling, "Inconsistent sampling for {}".format(i)
        """
        assert prox_sigma > 0.0, "Prox_sigma must be strictly positive."

        # returning only objects that might have been modified.
        return (diffusion_tensor, trace_local_smoothing_tensor, curvature_preservation_field, gradient, hessian)

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
            return self.diffusion_tensor.diffusivity.energy_potential(arr, self.grad)
        else:
            raise NotImplementedError(
                "DivergenceDiffusionOp may not be arising from an energy potential formulation.\
                                        If it is, define how to evaluate the associated energy functional."
            )

    @pycrt.enforce_precision(i="arr")
    def _compute_divergence_term(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y_div = xp.zeros(arr.size, dtype=arr.dtype)
        if self.diffusion_tensor or self.outer_diffusivity:
            y_div = self.grad(arr)
            if self.diffusion_tensor:
                # compute flux
                diffusion_tensor = self.diffusion_tensor(arr)
                y_div = diffusion_tensor(y_div)
                # apply divergence
                y_div = self.grad.T(y_div)
            if self.outer_diffusivity:
                # rescale divergence
                y_div *= self.outer_diffusivity
        return y_div

    @pycrt.enforce_precision(i="arr")
    def _compute_balloon_term(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        balloon_force = xp.zeros(arr.size, dtype=arr.dtype)
        if self.balloon_force:
            balloon_force = self.balloon_force(arr)
        return balloon_force

    @pycrt.enforce_precision(i="arr")
    def _compute_trace_term(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y_trace = xp.zeros(arr.size, dtype=arr.dtype)
        if self.trace_local_smoothing_tensor:
            hessian = self.hessian.unravel(self.hessian(arr))
            hessian = hessian[self.hessian_indices, :, :].reshape(1, -1)
            trace_tensor = self.trace_local_smoothing_tensor(arr)
            y_trace = trace_tensor(hessian)
            if self.outer_trace_diffusivity:
                # rescale trace
                y_trace *= self.outer_trace_diffusivity(arr)
        return y_trace

    @pycrt.enforce_precision(i="arr")
    def _compute_curvature_preserving_term(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y_curv = xp.zeros(arr.size, dtype=arr.dtype)
        if self.curvature_preservation_field:
            grad_arr = self.grad(arr)
            y_curv = self._jacobian_onto_field(grad_arr)
        return y_curv

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.zeros(arr.size, dtype=arr.dtype)
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
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        # Actual prox would correspond to a denoiser. Current implementation, instead, corresponds to a
        # diffusion stopped at given time, achievable applying a given number of grad steps to an initial
        # state. This corresponds to a PnP-like approach (Plug&Play).
        # time_step = 1 / ( 2 * self._diff_lipschitz)  # change time step computation and do it at init
        # for i in range(self.prox_steps):
        #    step = self.grad(arr)
        #    step *= time_step
        #    arr -= step
        # return arr
        stop_crit = pystop.MaxIter(self.prox_steps)
        pgd = pysol.PGD(f=self, g=None, show_progress=False, verbosity=100)
        pgd.fit(**dict(mode=pysolver.Mode.BLOCK, x0=arr, stop_crit=stop_crit, acceleration=False))
        return pgd.solution()


class DivergenceDiffusionOp(_DiffusionOp):
    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]],
        outer_diffusivity: pyct.OpT = None,
        diffusion_tensor: pyct.OpT = None,
        balloon_force: pyct.OpT = None,
        gradient: pyct.OpT = None,
        prox_sigma: pyct.Real = 2,
    ):
        super().__init__(
            arg_shape=arg_shape,
            sampling=sampling,
            outer_diffusivity=outer_diffusivity,
            diffusion_tensor=diffusion_tensor,
            balloon_force=balloon_force,
            gradient=gradient,
            prox_sigma=prox_sigma,
        )

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.zeros(arr.size)
        # compute divergence term
        y += self._compute_divergence_term(arr)
        # compute balloon force term
        y += self._compute_balloon_term(arr)
        return y.reshape(self.arg_shape)


class TraceDiffusionOp(_DiffusionOp):
    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]],
        outer_trace_diffusivity: pyct.OpT = None,
        trace_local_smoothing_tensor: pyct.OpT = None,
        hessian: pyct.OpT = None,
        prox_sigma: pyct.Real = 2,
    ):
        super().__init__(
            arg_shape=arg_shape,
            sampling=sampling,
            outer_trace_diffusivity=outer_trace_diffusivity,
            trace_local_smoothing_tensor=trace_local_smoothing_tensor,
            hessian=hessian,
            prox_sigma=prox_sigma,
        )

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.zeros(arr.size)
        # compute trace tensor term
        y += self._compute_trace_term(arr)
        return y.reshape(self.arg_shape)


class CurvaturePreservingDiffusionOp(_DiffusionOp):
    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        sampling: typ.Union[pyct.Real, tuple[pyct.Real, ...]],
        curvature_preservation_field: pyct.NDArray = None,
        gradient: pyct.OpT = None,
        hessian: pyct.OpT = None,
        prox_sigma: pyct.Real = 2,
    ):
        super().__init__(
            arg_shape=arg_shape,
            sampling=sampling,
            curvature_preservation_field=curvature_preservation_field,
            gradient=gradient,
            hessian=hessian,
            prox_sigma=prox_sigma,
        )
        # trace_local_smoothing_tensor = curvature_preservation_field.T @ curvature_preservation_field # not sure that's correct. let's think about hessian. also, we are allocating a sparse large matrix ...
        tensors = curvature_preservation_field.reshape(self.size, self.ndims, 1) * curvature_preservation_field.reshape(
            self.size, 1, self.ndims
        )
        trace_local_smoothing_tensor = tensors.reshape(self.size, -1)
        ops = []
        for i in range(self.ndims):
            ops.append([])
            for j in range(self.ndims):
                ops[i].append(pybase.DiagonalOp(trace_local_smoothing_tensor[:, i * self.ndims + j]))
            ops[i] = pyblock.block(ops[i], order=1)
        self.trace_local_smoothing_tensor = ops

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.zeros(arr.size)
        # compute trace tensor term
        y += self._compute_trace_term(arr)
        # compute curvature preserving term
        y += self._compute_curvature_preserving_term(arr)
        return y.reshape(self.arg_shape)

    # option to have extrinsic field
    # for extrinsic fields:
    # - define eigenvectors (across and along field) and associated eigenvalues (1, 0);
    # it's just a direction!! if (1,0) can't perform preferential smoothing depending on local behavior of field! Could only generically enhance coherence... discuss with Matthieu.
    # unless we use flux function constant on magnetic surfaces (see discussion we had at some point!). that's good, ask Luke
    # - instantiate desired anisotropic tensor;
    # - call _compute_diffusivities(eigenvalues)
    # - assemble diffusion tensor with _assemble_tensor() method

    # we do like this. instantiate and compute operator.

    # actually, we can reason in terms of flux function! However, is it unique? No! Is there a specific scaling that makes more sense than the other ones? Ask!!

    # lambda1 = np.ones(self.shape.size)
    # nonzero_grad_indices = np.where(grad_norm > 1e-2)
    # lambda1[nonzero_grad_indices] = 1 - np.exp(-3.31488 / ((grad_norm[nonzero_grad_indices] / beta) ** 8))
    # lambda2 = 1

    # @pycrt.enforce_precision(i="arr")
    # def shift_arr(self, shift_dir, arr):
    #    return shifted_arr

    # track total mass

    # give a "bounded between 0 and 1" label to DiffusionTensors, so that we can compute diff_lipschitz as needed

    # eigendecompose structure tensor
    # assemble full structure tensor as a (self.size, len(self.arg_size)**2) vector
    # evaluate structure tensor (its upper/lower triangular component)
    # structure_tensor = self._structure_tensor.apply(arr).unravel()
    # tril_indices = np.tril_indices(len(self.arg_shape), k=-1, m=len(self.arg_shape))
    # tril_matrix_locs = tril_indices[0] * len(self.arg_shape) + tril_indices[1] - np.arange(0, tril_indices[0].size)
    # structure_tensor_full = xp.insert(structure_tensor, tril_matrix_locs, None, axis=1)  # there is no cupy.insert
    # u, e = xp.linalg.eigh(structure_tensor_full.reshape(-1, len(self.arg_shape), len(self.arg_shape)), UPLO='U') # no eigh in dask.
    # REMARK: we could assemble only upper/lower part (with diagonal entries rescaled by factor 2) and then return
    # operator block + block.T. Not very important for low-dimensional (d=2,3) data (targeted use)
    # vec = tensor[:, i * len(self.arg_shape) + j]
    # if i == j:
    #    vec /= 2
    # ops[i].append(pybase.DiagonalOp(vec))
    # ...
    # block_op = pyblock.block(ops, order=1)
    # return block_op + block_op.T

    # for i in range(self.ndims):
    #    ops.append([])
    #   for j in range(self.ndims):
    #      ops[i].append(pybase.DiagonalOp(tensor[:, i * self.ndims + j]))  # avoid nested for loop
    # return pyblock.block(ops, order=1)

    # hessian_chunks = xp.zeros((self.ndims, self.ndims*self.size))
    # for i in range(self.ndims):
    #    hessian_chunks[i, :] = hessian[self.chunk_indices[i, :], :]
    # compute self.ndims matrix-vector products to avoid assemblying full Hessian.
    # trace_tensor = self.trace_local_smoothing_tensor(arr)
    # y_trace = trace_tensor(y_trace) #  needed trace_tensor(y_trace.T).T because pycsou processes rows and not columns, if y_trace assembled as matrix (inefficient).
    # y_trace = xp.trace(y_trace)
    """
    @pycrt.enforce_precision(i="arr")
    def _compute_flux(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        self.grad_arr = self.D.unravel(self.grad(arr))
        # evaluate diffusion tensor
        diffusion_tensor = self.diffusion_tensor(arr)
        # compute flux
        flux = xp.zeros(self.grad_arr.shape)
        # for tensor in diffusion_tensors: multiply by tensor to rescale the flux
        if self.diffusion_tensor.isotropic:
            flux[:, 0] = diffusion_tensor * self.grad_arr[:, 0]
            flux[:, 1] = diffusion_tensor * self.grad_arr[:, 1]
        else:
            flux[:, 0] = diffusion_tensor[:, 0] * self.grad_arr[:, 0]
            flux[:, 0] += diffusion_tensor[:, 1] * self.grad_arr[:, 1]
            flux[:, 1] = diffusion_tensor[:, 2] * self.grad_arr[:, 0]
            flux[:, 1] += diffusion_tensor[:, 3] * self.grad_arr[:, 1]
        return flux
        
                    hessian = self.hessian.unravel(self.hessian(arr))
            hessian = hessian.reshape(hessian.shape[0], -1)
            # REMARK: we pass a list of sub-operators as output of apply of Tensor class: more efficient for Hessian trace case
            trace_tensor = self.trace_local_smoothing_tensor(arr)
            y_trace = [trace_tensor[i](hessian[self.chunk_indices[i, :], :].reshape(1, -1)) for i in np.arange(0, self.ndims)]
            y_trace = sum(y_trace).reshape(1, -1)
            
            
            class DiffusivityGradBased(_Diffusivity):

    Class for gradient-based diffusivities.


    def __init__(self, arg_shape: pyct.Shape, beta: pyct.Real = 1, diff_method="gd", **diff_kwargs):
        super().__init__(arg_shape=arg_shape)
        if diff_method == "fd":
            self.grad = pydiff.Gradient.finite_difference(arg_shape=arg_shape, **diff_kwargs)
        elif diff_method == "gd":
            self.grad = pydiff.Gradient.gaussian_derivative(arg_shape=arg_shape, **diff_kwargs)
        self.beta = beta

    @pycrt.enforce_precision(i="arr")
    def _compute_grad_norm_sq(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        grad = self.grad.unravel(self.grad(arr))
        grad **= 2
        grad_norm_sq = xp.sum(grad, axis=-1, keepdims=True)
        return grad_norm_sq

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        raise NotImplementedError
        
        
        
        
                structure_tensor = self._structure_tensor.unravel(structure_tensor)
        structure_tensor = structure_tensor.reshape(structure_tensor.shape[0], -1).T  # why the transposed?
        # assemble full structure tensor
        structure_tensor_full = xp.zeros((structure_tensor.shape[0], self.ndims ** 2))
        triu_indices = np.triu_indices(self.ndims, k=0, m=self.ndims)
        triu_matrix_locs = triu_indices[0] * self.ndims + triu_indices[1]
        diag_indices = np.diag_indices(self.ndims, self.ndims)
        diag_matrix_locs = self.ndims * diag_indices[0] + diag_indices[1]
        structure_tensor_full[:, triu_matrix_locs] = structure_tensor
        structure_tensor_full[:, diag_matrix_locs] /= 2
        structure_tensor_full = structure_tensor_full.reshape(-1, self.ndims, self.ndims)
        structure_tensor_full += xp.transpose(structure_tensor_full, axes=[0, 2, 1])
        # eigendecompose tensor
        u, e, _ = xp.linalg.svd(structure_tensor_full, hermitian=True)
        """

    # watch out, structur tensor output is (3, N), not (N, 3)!
