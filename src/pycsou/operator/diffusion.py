import functools

import numpy as np
import scipy.optimize as sciop

import pycsou.abc as pyca
import pycsou.operator.linop.diff as pydiff
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


class Diffusivity(pyca.Func):
    """
    Abstract class to define diffusivity functions.
    The class features a method energy_functional(), which can be used to evaluate the energy potential that a
    divergence-based diffusion featuring the considered diffusivity derives from (when it makes sense).
    When implementing a new diffusivity, one needs to define whether it allows a variational interpretation or not.
    """

    def __init__(self, arg_shape: pyct.Shape):
        self.arg_shape = arg_shape
        self.size = int(np.prod(arg_shape))
        super().__init__(shape=(1, self.size))
        self.from_potential = False

    def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
        pass

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        raise NotImplementedError

    def energy_functional(self, arg_fct: pyct.NDArray) -> pyct.NDArray:
        return NotImplemented


class ExtrinsicDiffusivity(Diffusivity):
    """
    Class for extrinsic diffusivities. At initialization, one provides a diffusivity, typically computed according to
    some task-related quantities (original noisy image, prescribed smoothing field,...).
    """

    def __init__(self, arg_shape: pyct.Shape, extrinsic_arr: pyct.NDArray):
        super().__init__(arg_shape=arg_shape)
        self.extrinsic_diffusivity = extrinsic_arr
        self.from_potential = True

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.extrinsic_diffusivity

    @pycrt.enforce_precision(i="arg_fct")
    def energy_functional(self, arg_fct: pyct.NDArray) -> pyct.NDArray:
        return self.extrinsic_diffusivity * arg_fct


class TikhonovDiffusivity(Diffusivity):
    def __init__(self, arg_shape: pyct.Shape):
        super().__init__(arg_shape=arg_shape)
        self.from_potential = True

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.ones(arr.size)

    @pycrt.enforce_precision(i="arg_fct")
    def energy_functional(self, arg_fct: pyct.NDArray) -> pyct.NDArray:
        return 0.5 * arg_fct


class MfiDiffusivity(Diffusivity):
    def __init__(self, arg_shape: pyct.Shape, grad_based: bool = False):
        super().__init__(arg_shape=arg_shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.clip(arr, 0, None)
        y += 1
        return 1 / y


class DiffusivityGradBased(Diffusivity):
    """
    Class for gradient-based diffusivities.
    """

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


class PeronaMalikExponentialDiffusivity(DiffusivityGradBased):
    def __init__(self, arg_shape: pyct.Shape, beta: pyct.Real = 1, grad_based: bool = True, **diff_kwargs):
        super().__init__(arg_shape=arg_shape, beta=beta, **diff_kwargs)
        self.from_potential = True

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        # Inplace implementation of
        #   xp.exp(-grad_norm_sq/beta**2)
        z = self._compute_grad_norm_sq(arr)
        z /= self.beta**2
        return xp.exp(-z)

    @pycrt.enforce_precision(i="arg_fct")
    def energy_functional(self, arg_fct: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arg_fct)
        # Inplace implementation of
        #   0.5*(beta**2)*(1 - xp.exp(-arg_fct/beta**2)
        y = -arg_fct
        y /= self.beta**2
        y = -xp.exp(y)
        y += 1
        y *= self.beta**2
        y *= 0.5
        return y


class PeronaMalikRationalDiffusivity(DiffusivityGradBased):
    def __init__(self, arg_shape: pyct.Shape, beta: pyct.Real = 1, grad_based: bool = True, **diff_kwargs):
        super().__init__(arg_shape=arg_shape, beta=beta, **diff_kwargs)
        self.from_potential = True

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        # Inplace implementation of
        #   1 / (1 + grad_norm_sq/beta**2)
        z = self._compute_grad_norm_sq(arr)
        z /= self.beta**2
        z += 1
        return 1 / z

    @pycrt.enforce_precision(i="arg_fct")
    def energy_functional(self, arg_fct: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arg_fct)
        # Inplace implementation of
        #   0.5*(beta**2)*(xp.log(1+arg_fct/beta**2)
        y = arg_fct / self.beta**2
        y += 1
        y = xp.log(y)
        y *= self.beta**2
        y *= 0.5
        return y


class TameTotalVariationDiffusivity(DiffusivityGradBased):
    def __init__(self, arg_shape: pyct.Shape, beta: pyct.Real = 1, grad_based: bool = True, **diff_kwargs):
        super().__init__(arg_shape=arg_shape, beta=beta, **diff_kwargs)
        self.from_potential = True

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        # Inplace implementation of
        #   1/(xp.sqrt(1+grad_norm_sq/beta**2)
        z = self._compute_grad_norm_sq(arr)
        z /= self.beta**2
        z += 1
        z = xp.sqrt(z)
        return 1 / z

    @pycrt.enforce_precision(i="arg_fct")
    def energy_functional(self, arg_fct: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arg_fct)
        y = arg_fct
        y += 1
        return xp.sqrt(y)


class TotalVariationDiffusivity(DiffusivityGradBased):
    def __init__(self, arg_shape: pyct.Shape, beta: pyct.Real = 1, grad_based: bool = True, **diff_kwargs):
        super().__init__(arg_shape=arg_shape, **diff_kwargs)
        self.from_potential = True

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        z = self._compute_grad_norm_sq(arr)
        return 1 / z

    @pycrt.enforce_precision(i="arg_fct")
    def energy_functional(self, arg_fct: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arg_fct)
        return xp.sqrt(arg_fct)


class DiffusionTensor(pyca.Map):
    """
    Abstract class for diffusion tensors.
    The daughter classes DiffusionTensorIsotropic, DiffusionTensorAnisotropic allow to handle the isotropic/anisotropic cases.
    """

    def __init__(self, arg_shape: pyct.Shape, isotropic: bool = True):
        self.arg_shape = arg_shape
        self.size = int(np.prod(arg_shape))
        super().__init__(shape=(1 + 3 * (not isotropic), self.size))
        self._lipschitz = np.inf
        self.isotropic = isotropic
        self.from_potential = False

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        raise NotImplementedError


class DiffusionTensorIsotropic(DiffusionTensor):
    """
    Isotropic diffusion tensor. The diffusion tensor applied to the gradient to compute local flux is thus a diagonal matrix.
    If no diffusivity is specified at initialization, a homogeneous unitary diffusivity is imposed (corresponding to Tikhonov diffusivity).
    The diffusivity can be set at later times via set_diffusivity().
    """

    def __init__(self, arg_shape: pyct.Shape, diffusivity: pyct.OpT = None):
        super().__init__(arg_shape=arg_shape, isotropic=True)
        if diffusivity is None:
            self.diffusivity = TikhonovDiffusivity(arg_shape=arg_shape)
        else:
            self.diffusivity = diffusivity
        self.from_potential = self.diffusivity.from_potential

    def set_diffusivity(self, diffusivity: pyct.OpT):
        self.diffusivity = diffusivity

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = self.diffusivity(arr)
        return y


class DiffusionTensorAnisotropic(DiffusionTensor):
    """
    Abstract class to define anisotropic diffusion tensors, where by isotropic we mean that the resulting diffusion tensors are not diagonal.
    This class only considers anisotropic diffusion tensors based obtained as functions of the structur tensor, following Weickert.
    Daughter classes of DiffusionTensorAnisotropic only need to implement the method _compute_diffusivities(), which defines a rule to compute the
    diffusivities associated to the local eigenvectors of the structur tensor. These diffusivities define the smoothing behavior of the tensor.
    """

    def __init__(self, arg_shape: pyct.Shape, smooth_sigma=0, smooth_truncate=3, **diff_kwargs):
        """

        Parameters
        ----------
        arg_shape
        smooth_sigma
        smooth_truncate
        diff_kwargs
        """
        super().__init__(arg_shape=arg_shape, isotropic=False)
        self._structure_tensor = pydiff.StructureTensor(
            arg_shape=arg_shape,
            diff_method="gd",
            smooth_sigma=smooth_sigma,
            smooth_truncate=smooth_truncate,
            **diff_kwargs
        )
        self.smooth_sigma = smooth_sigma

    @pycrt.enforce_precision(i="arr")
    def _eigendec_struct_tensor(self, arr: pyct.NDArray):
        """
        Notes
        ----
        This function decomposes the structur tensor. For each pixel, the eigenvectors and associated eigenvalues are computed.
        If the structur tensor is not smoothed (smooth_sigma=0), then the decomposition is computed from the gradient without assembling
        the structur tensor.
        """
        xp = pycu.get_array_module(arr)
        if not self.smooth_sigma:
            # we could, alternatively, assemble the diffusion tensor differently in apply without _assemble_tensor, directly from _structure_tensor
            u = xp.zeros((self.size, 2, 2))
            grad = self._structure_tensor.grad.unravel(self._structure_tensor.grad(arr))
            grad **= 2
            grad_norm_sq = xp.sum(grad, axis=-1, keepdims=True)
            zero_grad_locs = xp.isclose(grad_norm_sq, 0)
            # set zero entries to 1 to avoid division by 0 when normalizing
            grad_norm_sq[zero_grad_locs] = 1
            u[:, 0, 0] = grad[:, 0]
            u[:, 1, 0] = grad[:, 1]
            u[:, 0, 1] = -grad[:, 1]
            u[:, 1, 1] = grad[:, 0]
            u /= grad_norm_sq.reshape(-1, 1, 1)
            s = xp.zeros((self.size, 2))
            s[:, 0] = grad_norm_sq
        else:
            _structure_tensor = self._structure_tensor.apply(arr)
            u, s, _ = xp.linalg.svd(_structure_tensor.reshape(2, 2, -1), hermitian=True)
        return u, s

    @pycrt.enforce_precision(i="eigval_struct")
    def _compute_diffusivities(self, eigval_struct: pyct.NDArray) -> pyct.NDArray:
        raise NotImplementedError

    @pycrt.enforce_precision(i=("u", "lambdas"))
    def _assemble_tensor(self, u: pyct.NDArray, lambdas: pyct.NDArray) -> pyct.NDArray:
        diffusion_tensor = lambdas[:, 0] * (
            u[:, :, 0].reshape(self.size, -1, 2) * u[:, :, 0].reshape(self.size, 2, -1)
        ) + lambdas[:, 1] * (u[:, :, 1].reshape(self.size, -1, 2) * u[:, :, 1].reshape(self.size, 2, -1))
        return diffusion_tensor

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        u, s = self._eigendec_struct_tensor(arr)
        lambdas = self._compute_diffusivities(s)
        return self._assemble_tensor(u, lambdas)


class DiffusionTensorAnisotropicEdgeEnhancing(DiffusionTensorAnisotropic):
    """
    Edge-enhancing version of the anisotropic tensor.
    """

    def __init__(self, shape: pyct.Shape, beta=1, m=4, smooth_sigma=1, smooth_truncate=3, **diff_kwargs):
        """

        Parameters
        ----------
        shape
        beta
        m
        smooth_sigma
        smooth_truncate
        diff_kwargs
        """
        super().__init__(shape=shape, smooth_sigma=smooth_sigma, smooth_truncate=smooth_truncate, **diff_kwargs)
        self.beta = beta
        self.m = 4

        # compute normalization constant c
        def f(c: pyct.Real, m: pyct.Integer):
            return 1 - np.exp(-c) * (1 + 2 * m * c)

        self.c = sciop.brentq(functools.partial(f, m), 1e-2, 100)

    @pycrt.enforce_precision(i="eigval_struct")
    def _compute_diffusivities(self, eigval_struct: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(eigval_struct)
        lambdas = xp.ones(eigval_struct.shape)
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


class DiffusionTensorAnisotropicCoherenceEnhancing(DiffusionTensorAnisotropic):
    """
    Coherence-enhancing version of the anisotropic tensor.
    """

    def __init__(self, shape: pyct.Shape, alpha=0.001, m=1, smooth_sigma=1, smooth_truncate=3, **diff_kwargs):
        super().__init__(shape=shape, smooth_sigma=1, smooth_truncate=3, **diff_kwargs)
        self.alpha = alpha
        self.m = m

    @pycrt.enforce_precision(i="eigval_struct")
    def _compute_diffusivities(self, eigval_struct: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(eigval_struct)
        coherence = (eigval_struct[:, 0] - eigval_struct[:, 1]) ** 2
        lambdas = self.alpha * xp.ones(eigval_struct.shape)
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


class DivergenceDiffusionOp(pyca.ProxDiffFunc):
    r"""
    Abstract class to deal with divergence based diffusion operators.
    """

    def __init__(
        self,
        arg_shape: pyct.Shape = None,
        diffusion_tensor: pyct.OpT = None,
        diff_type: str = "forward",
        h: pyct.Real = 1.0,
        boundary: str = "reflect",
        prox_sigma: pyct.Real = 0.05,
    ):
        """

        Parameters
        ----------
        arg_shape: pyct.Shape
            Shape of the pixelised image.
        diffusion_tensor: pyct.OpT
            DiffusionTensor operator, corresponding to an isotropic/anisotropic diffusion tensor. If None is specified,
            defaults to isotropic homogeneous diffusion with intensity 1 (Tikhonov)
        diff_type: str
            Finite Difference scheme used for the gradient. Allowed: "forward" and "central"
        h: pyct.Real
            Discretization parameter in space (sampling), corresponding to pixel size (for square pixels)
        boundary: str
            Boundary conditions handling. Recommended: "reflect" (Neumann homogeneous half-sample symmetric)
            or "mirror" (Neumann homogeneous full-sample symmetric) or "none" (Dirichlet)
        prox_sigma: pyct.Real
            Size of the structures that should be smoothed out by prox computation (when implemented in PnP-fashion).
            With current implementation, size depends on the size of the image (computational domain) itself,
            it is not in arbitrary units. Defaults to 0.05.
        """
        self.arg_shape = arg_shape
        self.size = int(np.prod(arg_shape))
        super().__init__(shape=(1, self.size))
        if diffusion_tensor is None:
            # defaults to Tikhonov isotropic homogeneous diffusion (Gaussian smoothing)
            self.diffusion_tensor = DiffusionTensorIsotropic(arg_shape=arg_shape)
        else:
            self.diffusion_tensor = diffusion_tensor
        self.diff_type = diff_type
        self.h = h
        self.boundary = boundary
        self.D = pydiff.Gradient.finite_difference(
            arg_shape=arg_shape, diff_type=diff_type, boundary=boundary, sampling=h
        )
        # estimate number of prox steps necessary to smooth structures of size prox_sigma (linear diffusion analogy)
        self.prox_sigma = prox_sigma
        # time_step = 1 / self._diff_lipschitz
        time_step = (
            h**2 / 4
        )  # valid only for isotropic case with diffusivity bounded between 0 and 1 (and for anisotropic standard cases?)
        t_final = self.prox_sigma**2 / 2
        self.prox_steps = t_final / time_step

    def asloss(self, data: pyct.NDArray = None) -> NotImplemented:
        """
        Notes
        -------
        DivergenceDiffusionOp class is not meant to be used to define a loss functional.
        """
        return NotImplemented

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> NotImplemented:
        """
        Notes
        -------
        Divergence-based diffusion operators may arise from a variational formulation. This is true, for example
        for the isotropic Perona-Malik, TV, Tikhonov and extrinsic diffusivities. For these cases, it is possible
        to define the associated energy functional. When this does not hold, method raises an error.
        """
        # if self.diffusion_tensors.size==1 and self.divergence_scalings.size==0 and self.diffusion_tensors[0].from_potential:
        if self.diffusion_tensor.from_potential:
            xp = pycu.get_array_module(arr)
            grad_arr = self.D.unravel(self.grad(arr))
            norm_grad_sq = xp.linalg.norm(grad_arr, axis=1)
            y = self.diffusion_tensor.diffusivity.energy_functional(norm_grad_sq)
            return xp.sum(y, axis=1)
        else:
            raise NotImplementedError(
                "DivergenceDiffusionOp may not be arising from an energy potential formulation.\
                                        If it is, define how to evaluate the associated energy functional."
            )

    def ravel(self, arr):
        return arr.swapaxes(0, 1).reshape(*arr.shape[: -1 - len(self.arg_shape)], -1)

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

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        arr = arr.flatten()
        flux = self._compute_flux(arr)
        # compute divergence of flux
        y = self.D.T(self.ravel(flux))
        # for scaling in divergence_scalings: multiply by tensor to transform divergence
        return y.reshape(self.arg_shape)

    @pycrt.enforce_precision(i="arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        # Actual prox would correspond to a denoiser. Current implementation, instead, corresponds to a
        # diffusion stopped at given time, achievable applying a given number of grad steps to an initial
        # state. This corresponds to a PnP-like approach (Plug&Play).
        time_step = 1 / self._diff_lipschitz
        for i in range(self.prox_steps):
            step = self.grad(arr)
            step *= time_step
            arr -= step
        pass


class MeanCurvatureMotion(DivergenceDiffusionOp):
    """
    Notes
    -------
    MCM leads to instability in explicit Euler time marching schemes due to presence of TotalVariation diffusivity.
    Currently, we use TameTotalVariation diffusivity to avoid issue. Still, factor grad(arr) in front of divergence also causes
     issues. Address.
    """

    def __init__(
        self,
        arg_shape: pyct.Shape = None,
        diffusion_tensor=None,
        diff_type: str = "forward",
        h: float = 1.0,
        boundary: str = "reflect",
        prox_sigma: pyct.Real = 0.05,
        affine_invariant: bool = False,
    ):
        """

        Parameters
        ----------
        affine_invariant: bool
            Whether the affine invariant version should be implemented or not. The affine invariant version transform the curvature c in MCM as c^(1/3).
        """
        super().__init__(
            arg_shape=arg_shape,
            diffusion_tensor=None,
            diff_type=diff_type,
            h=h,
            boundary=boundary,
            prox_sigma=prox_sigma,
        )
        self.affine_invariant = affine_invariant

    @pycrt.enforce_precision(i="arr")
    def _compute_flux(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        flux = super()._compute_flux(arr)
        self.grad_norm = xp.linalg.norm(self.grad_arr, axis=1)
        z = self.grad_norm**2
        z += 1
        z = xp.sqrt(z)
        flux[:, 0] /= z
        flux[:, 1] /= z
        return flux

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        div_term = super().grad(arr)
        if self.affine_invariant:
            div_term **= 1 / 3
        return div_term * self.grad_norm.reshape(self.arg_shape)


class Snakes(DivergenceDiffusionOp):
    """
    Notes
    -------
    This class allows to deal with implicit snakes, geodesic snakes and self-snakes. These different behviors can be
    obtained defining suitable diffusion_tensor and divergence_scaling terms.

    Snakes suffer from instability in explicit Euler time marching schemes due to presence of TotalVariation diffusivity.
    Currently, we use TameTotalVariation diffusivity to avoid issue. Still, factor grad(arr) in front of divergence may also
     cause issues if the divergence scaling does not decay sufficiently fast to zero for large values of gradient norm. Address.
    """

    def __init__(
        self,
        arg_shape: pyct.Shape = None,
        diffusion_tensor: pyct.OpT = None,
        diff_type: str = "forward",
        h: float = 1.0,
        boundary: str = "reflect",
        prox_sigma: pyct.Real = 0.05,
        divergence_scaling: pyct.OpT = None,
        balloon_force: pyct.Real = 0.0,
    ):
        """

        Parameters
        ----------
        diffusion_tensor: pyct.OpT
            It can be used to further rescale the flux (argument of the divergence operator) by an edge-stopping function/other.
        divergence_scaling: pyct.OpT
            It can be used to rescale the divergence of the flux by an edge-stopping function/other.
        balloon_force: pyct.Real
            Balloon force coefficient providing dilation/erosion effect depending on its positive/negative sign.
        """
        super().__init__(
            arg_shape=arg_shape,
            diffusion_tensor=diffusion_tensor,
            diff_type=diff_type,
            h=h,
            boundary=boundary,
            prox_sigma=prox_sigma,
        )
        self.divergence_scaling = divergence_scaling
        self.balloon_force = balloon_force

    @pycrt.enforce_precision(i="arr")
    def _compute_flux(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        flux = super()._compute_flux(arr)
        self.grad_norm = xp.linalg.norm(self.grad_arr, axis=1)
        z = self.grad_norm**2
        z += 1
        z = xp.sqrt(z)
        flux[:, 0] /= z
        flux[:, 1] /= z
        return flux

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        z = super().grad(arr)
        # add \nu  (non-zero values give implicit snakes)
        z += self.balloon_force
        z *= self.grad_norm.reshape(self.arg_shape)
        if self.divergence_scaling is not None:
            # rescale by edge-stopping function (self-snakes)
            scaling = self.divergence_scaling(arr)
            z /= scaling
        return z


# option to have extrinsic field
# for extrinsic fields:
# - define eigenvectors (across and along field) and associated eigenvalues (1, 0);
# it's just a direction!! if (1,0) can't perform preferential smoothing depending on local behavior of field! Could only generically enhance coherence... discuss with Matthieu.
# unless we use flux function constant on magnetic surfaces (see discussion we had at some point!). that's good, ask Luke
# - instantiate desired anisotropic tensor;
# - call _compute_diffusivities(eigenvalues)
# - assemble diffusion tensor with _assemble_tensor() method

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
