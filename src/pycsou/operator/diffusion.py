import functools
import warnings

import numpy as np
import scipy.optimize as sciop

import pycsou.abc as pyca
import pycsou.abc.operator as pyco
import pycsou.abc.solver as pysolver
import pycsou.operator.blocks as pyblock
import pycsou.operator.linop.base as pybase
import pycsou.operator.linop.diff as pydiff
import pycsou.opt.solver as pysol
import pycsou.opt.stop as pystop
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct


class _BalloonForce(pyca.Map):
    r"""
    Abstract balloon force operator. Daughter classes implement specific balloon force terms.
    """

    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None):
        """
        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array.
        gradient: :py:class:`~pycsou.operator.linop.diff.Gradient`
            Gradient operator. Defaults to `None`.
        """
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
    r"""
    Dilation operator.

    Notes
    -----
    Given a :math:`D`-dimensional signal:

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0}, \dots, N_{D-1}},

    its ``.apply()`` method computes

    .. math::

        \vert \nabla \mathbf{f} \vert \in \mathbb{R}^{N_{0}, \dots, N_{D-1}},

    where :math:`\vert\cdot\vert` represents the :math:`L^2`-norm of the gradient (on each pixel).

    Can be used to obtain the PDE version of the morphological dilation operator, which reads

    .. math::

        \frac{\partial \mathbf{f}}{\partial t}=\vert \nabla \mathbf{f}\vert.
    """

    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None):
        """

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array.
        gradient: :py:class:`~pycsou.operator.linop.diff.Gradient`
            Gradient operator. Defaults to `None`.
        """
        super().__init__(arg_shape=arg_shape, gradient=gradient)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        grad_arr = self.grad.unravel(self.grad(arr))
        return xp.linalg.norm(grad_arr, axis=1, keepdims=True)


class Erosion(_BalloonForce):
    r"""
    Erosion operator.

    Notes
    -----
    Given a :math:`D`-dimensional signal:

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0}, \dots, N_{D-1}},

    its ``.apply()`` method computes

    .. math::

        - \vert \nabla \mathbf{f} \vert \in \mathbb{R}^{N_{0}, \dots, N_{D-1}},

    where :math:`\vert\cdot\vert` represents the :math:`L^2`-norm of the gradient (on each pixel).

    Can be used to obtain the PDE version of the morphological dilation operator, which reads

    .. math::

        \frac{\partial \mathbf{f}}{\partial t}=-\vert \nabla \mathbf{f}\vert.
    """

    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None):
        """

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array.
        gradient: :py:class:`~pycsou.operator.linop.diff.Gradient`
            Gradient operator. Defaults to `None`.
        """
        self = -Dilation(arg_shape=arg_shape, gradient=gradient)


class _Diffusivity(pyca.Map):
    r"""
    Abstract diffusivity operator. Daughter classes implement specific diffusivity functions.

    Notes
    -----
    Given a :math:`D`-dimensional signal,

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0}, \dots, N_{D-1}},

    its ``.apply()`` method returns the :math:`D`-dimensional signal

    .. math::

        g(\mathbf{f}) \in \mathbb{R}^{N_{0}, \dots, N_{D-1}},

    where :math:`g(\cdot)` is defined through the so-called diffusivity function.

    **Remark 1**

    The class features a ``.freeze()`` method. When applied to an array `arr`, it freezes the diffusivity
    at the value obtained applying ``.apply()``  to `arr`.

    The class also features a ``.set_frozen_diffusivity()`` method. When applied to an array `arr`, it freezes the diffusivity
    at the value ``arr``.

    **Remark 2**

    The class features a ``.energy_functional()`` method, which can be used to evaluate the energy potential
    that a divergence-based diffusion term featuring the considered diffusivity derives from (when it makes sense).
    When implementing a new diffusivity, one should check whether this variational interpretation holds: if this is the case,
    attribute ``from_potential`` should be set to `True` and method ``.energy_functional()`` should be implemented.

    **Remark 3**

    The class features the attribute ``bounded``. If ``True``, this signals that the map returns values
    in the range :math:`(0, 1]`. When implementing a new diffusivity, one should check whether this holds: if this is the case,
    ``from_potential`` should be set to `True`.
    """

    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT = None):
        """
        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array.
        gradient: :py:class:`~pycsou.operator.linop.diff.Gradient`
            Gradient operator. Defaults to `None`.
        """
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
            self.frozen_diffusivity = self.apply(arr)
            self.frozen = True

    def set_frozen_diffusivity(self, arr: pyct.NDArray):
        if self.frozen:
            warnings.warn("Diffusivity has already been frozen. Cannot overwrite previous frozen state.")
        else:
            self.frozen_diffusivity = arr
            self.frozen = True

    @pycrt.enforce_precision(i="arr")
    def _compute_grad_norm_sq(self, arr: pyct.NDArray, grad: pyct.OpT = None):
        # compute squared norm of gradient (on each pixel), needed for several diffusivities.
        xp = pycu.get_array_module(arr)
        grad_arr = self.unravel_grad(grad(arr))
        grad_arr **= 2
        return xp.sum(grad_arr, axis=1, keepdims=False)

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        raise NotImplementedError

    def energy_potential(self, arr: pyct.NDArray, grad: pyct.OpT = None) -> pyct.NDArray:
        """
        Notes
        -----
        A gradient operator is passed instead of using the `gradient` attribute of the class itself
        because the two operators might be different. The one used in the diffusivity computation
        typically features Gaussian derivatives for stability reasons.

        """
        return NotImplemented


class TikhonovDiffusivity(_Diffusivity):
    r"""
    Diffusivity associated to Tikhonov regularization.

    Let :math:`f_i` be an entry (pixel) of the vectorisation of the :math:`D`-dimensional signal,

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0}, \dots, N_{D-1}}.

    Then the Tikhonov diffusivity function reads

    .. math ::

        (g(\mathbf{f}))_i = 1, \quad \forall i.

    """

    def __init__(self, arg_shape: pyct.NDArrayShape):
        """

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array.
        """
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
    r"""
    Minimum Fisher Information (MFI) inspired diffusivity [see `Anton <https://iopscience.iop.org/article/10.1088/0741-3335/38/11/001/pdf>`_].

    Let :math:`f_i` be an entry (pixel) of the vectorisation of the :math:`D`-dimensional signal,

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0}, \dots, N_{D-1}}.

    Then the MFI diffusivity function reads

    * If ``tame`` is ``False``,
    .. math ::

        (g(\mathbf{f}))_i = \frac{1} { \max \{ 0, f_i \} }, \quad \forall i;

    * If ``tame`` is ``True``,
    .. math ::

        (g(\mathbf{f}))_i = \frac{1} {1 + \max \{ 0, f_i \} }, \quad \forall i.


    **Remark**

    In both cases, the corresponding divergence-based diffusion term does not allow a variational interpretation.
    Indeed, the Euler-Lagrange equations arising from the original variational formulation yield an extra term
    that cannot be written in divergence form.
    """

    def __init__(self, arg_shape: pyct.NDArrayShape, tame: bool = True):
        """

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array.
        tame: bool
            Whether to consider the tame version bounded in :math:`(0, 1]` or not. Defaults to `True`.
        """
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
    r"""
    Perona-Malik diffusivity [see `Perona-Malik <http://image.diku.dk/imagecanon/material/PeronaMalik1990.pdf>`_].

    Let :math:`f_i` be the :math:`i`-th entry (pixel) of the vectorisation of the :math:`D`-dimensional signal,

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0}, \dots, N_{D-1}}.

    Furthermore, let :math:`(\nabla \mathbf{f})_i = \big((\frac{\partial \mathbf{f}}{\partial x_1})_i,\dots,(\frac{\partial \mathbf{f}}{\partial x_{D-1}})_i\big)^T`.

    Then the Perona-Malik diffusivity function reads

    * In the exponential case,
        .. math ::

            (g(\mathbf{f}))_i = \exp(-\vert (\nabla \mathbf{f})_i \vert ^2 / \beta^2), \quad \forall i;

    * in the rational case,
        .. math ::

            (g(\mathbf{f}))_i = \frac{1} { 1+\vert (\nabla \mathbf{f})_i \vert ^2 / \beta^2}, \quad \forall i,

    where :math:`\beta` is the contrast parameter.

    In both cases, the corresponding divergence-based diffusion term allows a variational interpretation
    [see `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_ for exponential case].

    **Remark**

    It is recommended to provide a Gaussian derivative-based gradient (:math:`\nabla=\nabla_\sigma`). This acts as regularization
    when the diffusivity is used  for the ill-posed Perona-Malik diffusion process [see `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_].
    """

    def __init__(
        self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT, beta: pyct.Real = 1, pm_fct: str = "exponential"
    ):
        """

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array.
        gradient: :py:class:`~pycsou.operator.linop.diff.Gradient`
            Gradient operator. Defaults to `None`.
        beta: pyct.Real
            Contrast parameter. Defaults to `1`.
        pm_fct: str
            Perona-Malik function type. Defaults to `exponential`. Allowed values are `exponential`, `rational`.
        """
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
    r"""
    Total Variation (TV) diffusivity [see `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_].

    Let :math:`f_i` be an entry (pixel) of the vectorisation of the :math:`D`-dimensional signal,

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0}, \dots, N_{D-1}}.

    Furthermore, let :math:`(\nabla \mathbf{f})_i = \big((\frac{\partial \mathbf{f}}{\partial x_1})_i,\dots,(\frac{\partial \mathbf{f}}{\partial x_{D-1}})_i\big)^T`.

    Then the Total Variation diffusivity function reads

    * If ``tame`` is ``False``,
        .. math ::

            (g(\mathbf{f}))_i = \frac{1} { \vert (\nabla \mathbf{f})_i \vert}, \quad \forall i;

    * If ``tame`` is ``True``,
        .. math ::

            (g(\mathbf{f}))_i = \frac{1} { \sqrt{1+ \vert (\nabla \mathbf{f})_i \vert ^2 / \beta^2}}, \quad \forall i,

        where :math:`\beta` is the contrast parameter.

    In both cases, the corresponding divergence-based diffusion term allows a variational interpretation
    [see `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_ for untamed case].

    **Remark**

    It is recommended to provide a Gaussian derivative-based gradient (:math:`\nabla=\nabla_\sigma`) to reduce sensitivity to noise.
    """

    def __init__(self, arg_shape: pyct.NDArrayShape, gradient: pyct.OpT, beta: pyct.Real = 1, tame: bool = True):
        """

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array.
        gradient: :py:class:`~pycsou.operator.linop.diff.Gradient`
            Gradient operator. Defaults to `None`.
        beta: pyct.Real
            Contrast parameter. Defaults to `1`.
        tame: bool
            Whether to consider tame version or not. Defaults to `True`.
        """
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


class _DiffusionCoefficient:
    r"""
    Abstract class for (tensorial) diffusion coefficients. Daughter classes :py:class:`~pycsou.operator.diffusion.DiffusionCoeffIsotropic`
    and :py:class:`~pycsou.operator.diffusion._DiffusionCoeffAnisotropic` handle the isotropic/anisotropic cases.

    Notes
    -----
    Given a :math:`D`-dimensional signal:

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0}, \dots, N_{D-1}},

    ``_DiffusionCoefficient`` operators :math:`\mathbf{D}` can be used to define

    * divergence-based diffusion operators
        .. math::
            \text{div} (\mathbf{D} \nabla \mathbf{f}),

    * trace-based diffusion operators
        .. math::
            \text{trace}(\mathbf{D} \mathbf{H}(\mathbf{f})),
        where :math:`\mathbf{H}(\cdot)` is the Hessian.


    **Remark 1**

    In principle ``_DiffusionCoefficient`` depends on the input signal itself (or on some other quantity), so
    that :math:`\mathbf{D}=\mathbf{D}(\mathbf{f})`. The ``.apply()`` method, when applied to an array `arr`, returns
    the operator associated to the diffusion coefficient evaluated in `arr`.

    **Remark 2**

    The meaning of the ``_DiffusionCoefficient`` :math:`\mathbf{D}` can be better understood focusing on the :math:`i`-th entry (pixel) :math:`f_i`
    of the vectorisation of :math:`\mathbf{f}`. Furthermore, let
    :math:`(\nabla \mathbf{f})_i = \big((\frac{\partial \mathbf{f}}{\partial x_1})_i,\dots,(\frac{\partial \mathbf{f}}{\partial x_{D-1}})_i\big)^T`.
    We consider the divergence-based case. Then, the :math:`i`-th component of :math:`\mathbf{D}`
    is a tensor :math:`D_i\in\mathbb{R}^{D\times D}`. When applied to the :math:`i`-th component of :math:`\nabla\mathbf{f}`, this gives
    the flux

    .. math::
        \Phi_i = D_i (\nabla\mathbf{f})_i \in \mathbb{R}^D,

    which, applying the divergence, yields

    .. math::
        \Delta f_i = \text{div}(\Phi_i) \in \mathbb{R}.

    In the context of PDE-based image processing, :math:`\Delta f_i` represents the update of :math:`f_i`
    in a denoising/reconstruction process. ``_DiffusionCoefficient`` operators are obtained
    suitably stacking the tensors :math:`D_i, i=1,\dots, N_0\cdot\dots\cdot N_{D-1}`.

    **Remark 3**

    The class features a ``.freeze()`` method. When applied to an array `arr`, it freezes the diffusion coefficient
    at the operator obtained applying ``.apply()`` to `arr`.

    The class also features a ``.set_frozen_diffusivity()`` method. When fed an operator `frozen_op`, it freezes the
    diffusion coefficient at the operator `frozen_op`.

    **Remark 4**

    The class features the boolean attribute ``trace_term``, indicating whether the diffusion coefficient is meant to
    be used in a divergence-based (``trace_term`` should be set to `False`) or in a trace-based operator (``trace_term``
    should be set to `True`). The stacking used to generate the operator in the ``.apply()`` method is different in the
    two cases. When ``trace_term`` is ``True``, the output of ``.apply()`` is an operator which, when applied to a suitable
    object, already computes the trace of the diffusion tensor applied to that object.

    **Remark 5**

    The class features the attributes ``from_potential`` and ``bounded``. See discussion for ``_Diffusivity``.

    Developer notes
    --------------
    Currently, _DiffusionCoefficient is not a pycsou operator. This is because the method``.apply()`` returns
    a LinOp/DiagonalOp and not a scalar NDArray. We define some basic arithmetic that allows to consider sums
    between different diffusion coefficient objects and multiplying/dividing by scalars. _DiffusionCoefficient
    do not allow multidimensional inputs though. Maybe acceptable since it is not a pycsou operator and these
    operators will likely only ever be used in the context of diffusion processes?
    """

    def __init__(self, arg_shape: pyct.NDArrayShape, isotropic: bool = True, trace_term: bool = False):
        r"""
        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array.
        isotropic: bool
            Whether ``_DiffusionCoefficient`` is isotropic or not. Defaults to `True`.
        trace_term: bool
            Whether ``_DiffusionCoefficient`` is meant to be used in a trace formulation or not.
            Defaults to `False`. Method ``.apply()`` acts differently depending on value of `trace_term`.

        """
        self.arg_shape = arg_shape
        self.size = int(np.prod(arg_shape))
        self.ndims = len(self.arg_shape)
        self.isotropic = isotropic
        self.trace_term = trace_term
        self.from_potential = False
        self.frozen = False
        self.frozen_op = None
        self.bounded = False
        # compute scaling coefficients for more efficient computation in trace-based case
        self._coeff_op = np.ones((self.ndims, self.ndims), dtype=int)
        if self.trace_term:
            # set extra diagonal coefficients to 2: terms need to be considered twice because of symmetry of both Hessian and diffusion coefficient
            self._coeff_op *= 2
            self._coeff_op -= np.eye(self.ndims, dtype=int)

    def freeze(self, arr: pyct.NDArray):
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

    def __call__(self, arr: pyct.NDArray) -> pyct.NDArray:
        """
        Alias for :py:meth:`~pycsou.abc.operator.diffusion._DiffusionCoefficient`.
        """
        return self.apply(arr)


class DiffusionCoeffIsotropic(_DiffusionCoefficient):
    r"""
    Class for isotropic diffusion coefficients, where we follow the definition of isotropy from
    [`Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_].

    **Remark 1**

    By isotropic, we thus mean that the diffusion tensor is fully described by a diffusivity function :math:`g(\cdot)`.
    Indeed, let :math:`\mathbf{D}` be a ``DiffusionCoeffIsotropic`` and let :math:`f_i` be the :math:`i`-th entry
    (pixel) of the vectorisation of the :math:`D`-dimensional signal

    .. math::
        \mathbf{f} \in \mathbb{R}^{N_{0}, \dots, N_{D-1}}.

    Furthermore, let :math:`(\nabla \mathbf{f})_i = \big((\frac{\partial \mathbf{f}}{\partial x_1})_i,\dots,(\frac{\partial \mathbf{f}}{\partial x_{D-1}})_i\big)^T`.
    We consider the divergence-based case. Then, the :math:`i`-th component of :math:`\mathbf{D}`
    is the tensor :math:`D_i=(g(\mathbf{f}))_i\,I_D`, where :math:`(g(\mathbf{f}))_i\in\mathbb{R}` and :math:`I_D` is the :math:`D`-dimensional identity
    matrix.

    Applying :math:`D_i` to the :math:`i`-th component of :math:`\nabla\mathbf{f}` gives
    the flux

    .. math::
        \Phi_i = (g(\mathbf{f}))_i\,I_D (\nabla\mathbf{f})_i = (g(\mathbf{f}))_i (\nabla\mathbf{f})_i \in \mathbb{R}^D.

    **Remark 2**
    Instances of :py:class:`~pycsou.operator.diffusion.DiffusionCoeffIsotropic` inherit attributes
    ``from_potential`` and ``bounded`` from the diffusivity.
    """

    def __init__(self, arg_shape: pyct.NDArrayShape, diffusivity: pyct.OpT = None, trace_term: bool = False):
        r"""

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array.
        diffusivity: :py:class:`~pycsou.operator.diffusion._Diffusivity`
            Map defining the diffusivity associated to the isotropic coefficient. Defaults to `None`, in which case
            Tikhonov diffusivity is used.
        trace_term: bool
            Whether diffusion coefficient is meant to be used in a trace formulation or not.
            Defaults to `False`. Method ``.apply()`` acts differently depending on value of `trace_term`.

        """
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

    def apply(self, arr: pyct.NDArray) -> pyct.OpT:
        r"""

        Notes
        -----
        Let :math:`N_{tot}=N_0\cdot\ldots\cdot N_{D-1}`, where :math:`D` is the dimension of the signal. The method
        returns an operator which is:

        * if ``trace_term`` is ``True`` a :py:class:`~pycsou.abc.arithmetic.LinOp` of shape :math:`(N_{tot}, N_{tot}D)`;
        * if ``trace_term`` is ``False`` a :py:class:`~pycsou.operator.linop.base.DiagonalOp` of shape :math:`(N_{tot}D, N_{tot}D)`.

        """
        if not self.frozen:
            xp = pycu.get_array_module(arr)
            y = self.diffusivity(arr)
            if self.trace_term:
                # assemble and return a LinOp(self.size, self.ndims*self.size)
                ops = [pybase.DiagonalOp(y.squeeze())]
                ops *= self.ndims
                return pyblock.hstack(ops)
            else:
                # assemble and return a DiagonalOp(self.ndim*self.size, self.ndims*self.size)
                return pybase.DiagonalOp(xp.tile(y.squeeze(), len(self.arg_shape)))
        else:
            return self.frozen_op


class _DiffusionCoeffAnisotropic(_DiffusionCoefficient):
    r"""
    Abstract class for anisotropic diffusion coefficients, where we follow the definition of anisotropy from
    [`Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_].

    **Remark 1**

    The class is designed for diffusion tensors obtained as function of the structure tensor (it could be named
    ``_DiffusionCoeffStructTensorBased``). Other types of anisotropic tensors are not meant to be implemented as
    daughter classes of :py:class:`~pycsou.operator.diffusion._DiffusionCoeffAnisotropic`.

    **Remark 2**

    By `anisotropic` we mean that the diffusion tensors, locally, are not multiples of the identity matrix.
    Indeed, let :math:`\mathbf{D}` be a ``DiffusionCoeffAnisotropic`` and let :math:`f_i` be the :math:`i`-th entry
    (pixel) of the vectorisation of the :math:`D`-dimensional signal

    .. math::
        \mathbf{f} \in \mathbb{R}^{N_{0}, \dots, N_{D-1}}.

    Furthermore, let :math:`(\nabla \mathbf{f})_i = \big((\frac{\partial \mathbf{f}}{\partial x_1})_i,\dots,(\frac{\partial \mathbf{f}}{\partial x_{D-1}})_i\big)^T`.
    We consider the divergence-based case. Then, the :math:`i`-th component of :math:`\mathbf{D}`
    is the symmetric matrix

    .. math::
        D_i=
        \left(\begin{array}{ccc}
        (D_i)_{11} & \cdots & (D_i)_{1D} \\
        \vdots & \ddots & \vdots \\
        (D_i)_{1D} & \cdots & (D_i)_{DD}
        \end{array}\right)\in\mathbb{R}^{D\times D}.

    Applying :math:`D_i` to the :math:`i`-th component of :math:`\nabla\mathbf{f}` gives
    the flux

    .. math::
        \Phi_i = D_i (\nabla\mathbf{f})_i \in \mathbb{R}^D,

    which, since :math:`D_i` is not a multiple of the identity matrix :math:`I_D`, is not a simple rescaling of the
    gradient. As a consequence, the flux can point towards directions different from the gradient,
    allowing smoothing processes along interesting directions. These directions can be chosen to enhance, for example,
    the `edges` or the `coherence` of the signal.

    **Remark 3**

    As mentioned above, this class considers diffusion coefficients which depend on the structure tensor
    (see :py:class:`~pycsou.operator.linop.diff.StructureTensor`), as we now describe. Let us consider, for each pixel,
    the structure tensor

    .. math::
        S_i = (\nabla\mathbf{f})_i(\nabla \mathbf{f})_i^T\,\in\mathbb{R}^{D\times D}.

    The matrix :math:`S_i` is a symmetric positive semidefinite matrix. From its eigenvalue decomposition,
    we obtain the eigenvectors :math:`\mathbf{v}_0,\dots,\mathbf{v}_{D-1}` and the associated eigenvalues
    :math:`e_0, \dots,e_{D-1}`, with

    .. math::
        S_i = \sum_{j=0}^{D-1} e_j\mathbf{v}_j(\mathbf{v}_j)^T.

    The :math:`i`-th component :math:`D_i` of the ``_DiffusionCoeffAnisotropic`` :math:`\mathbf{D}` is given
    by

    .. math::
        D_i = \sum_{j=0}^{D-1} \lambda_j\mathbf{v}_j(\mathbf{v}_j)^T,

    where :math:`\lambda_j=\lambda_j(e_0,\dots,e_{D-1}), j=0,\dots,D-1`. This corresponds to assigning intensities
    :math:`\lambda_j` different from the eigenvalues :math:`e_j` to the eigenvectors of the structure tensor. The result
    is a diffusion coefficient that, when used in the context of diffusion operators, will enhance or dampen features
    by smoothing with different intensities along the different eigenvector directions.

    **Remark 4**

    Daughter classes of :py:class:`~pycsou.operator.diffusion._DiffusionCoeffAnisotropic` only need to implement the
    method ``_compute_intensities()``, which defines a rule to compute the smoothing intensities associated to each
    eigenvector of the structure tensor. These intensities define the smoothing behavior of the tensor
    (edge-enhancing, coherence-enhancing).

    """

    def __init__(self, arg_shape: pyct.NDArrayShape, structure_tensor: pyct.OpT, trace_term: bool = False):
        """

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array.
        structure_tensor: :py:class:`~pycsou.operator.linop.diff.StructureTensor`
            Structure tensor operator.
        trace_term: bool
            Whether diffusion coefficient is meant to be used in a trace formulation or not.
            Defaults to `False`. Method ``.apply()`` acts differently depending on value of `trace_term`.
        """
        super().__init__(arg_shape=arg_shape, isotropic=False, trace_term=trace_term)
        msg = "`structure_tensor.arg_shape`={} inconsistent with `arg_shape`={}.".format(
            structure_tensor.arg_shape, arg_shape
        )
        assert structure_tensor.arg_shape == arg_shape, msg
        self.structure_tensor = structure_tensor
        # compute the indices of the upper triangular structure tensor to be selected to assemble its full version
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

    # how to enforce precision on tuple of outputs? should I simply use coerce?
    # Actually, both structure_tensor and svd preserve precision, so should not be an issue
    def _eigendecompose_struct_tensor(self, arr: pyct.NDArray) -> (pyct.NDArray, pyct.NDArray):
        """
        Notes
        ----
        This function decomposes the structure tensor. For each pixel, the eigenvectors and associated eigenvalues are computed.

        Developer notes
        --------------
        **Remark 1**
        Currently, ``xp.linalg.svd`` is used to decompose the matrices.
        * In NUMPY case, the argument Hermitian=True prompts a call to the efficient ``numpy.linalg.eigh()``.
        * In CUPY case, the argument Hermitian does not exist. There is a method ``cupy.linalg.eigh()`` though, we could leverage it.
        * In DASK case, the argument Hermitian does not exist. Moreover, there is no dask version of ``.eigh()``. We should therefore use ``.svd()``.

        **Remark 2**

        In the two-dimensional case :math:`D=2`, where the input signal is an image :math:`\mathbf{f}\in\mathbb{R}^{N_0,N_1}`, closed formulas
        could be used for the eigendecomposition of the structur tensor. To keep things general and be able to work in :math:`D` dimensions,
        we do not exploit them and apply ``.svd()`` instead.
        """
        xp = pycu.get_array_module(arr)
        # compute upper/lower triangular component of structure tensor
        structure_tensor = self.structure_tensor.apply(arr)
        structure_tensor = self.structure_tensor.unravel(structure_tensor).squeeze()
        structure_tensor = structure_tensor.reshape(structure_tensor.shape[0], -1).T
        # assemble full structure tensor
        structure_tensor_full = structure_tensor[:, self.full_matrix_indices].reshape(-1, self.ndims, self.ndims)
        # eigendecompose tensor
        N = pycd.NDArrayInfo
        is_numpy = N.from_obj(arr) == N.NUMPY
        if is_numpy:
            u, e, _ = xp.linalg.svd(structure_tensor_full, hermitian=True)
        else:
            u, e, _ = xp.linalg.svd(structure_tensor_full)
        return u, e

    @pycrt.enforce_precision(i="eigval_struct")
    def _compute_intensities(self, eigval_struct: pyct.NDArray) -> pyct.NDArray:
        raise NotImplementedError

    @pycrt.enforce_precision(i=("u", "lambdas"))
    def _assemble_tensors(self, u: pyct.NDArray, lambdas: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(u)
        diffusion_tensors = xp.zeros((self.size, len(self.arg_shape), len(self.arg_shape)))
        for i in range(len(self.arg_shape)):
            # compute rank 1 matrices from eigenvectors, multiply them by intensities and sum up
            diffusion_tensors += lambdas[:, i].reshape(-1, 1, 1) * (
                u[:, :, i].reshape(self.size, -1, len(self.arg_shape))
                * u[:, :, i].reshape(self.size, len(self.arg_shape), -1)
            )
        return diffusion_tensors

    def apply(self, arr: pyct.NDArray) -> pyct.OpT:
        r"""

        Notes
        -----
        Let :math:`N_{tot}=N_0\cdot\ldots\cdot N_{D-1}`, where :math:`D` is the dimension of the signal. The number of
        extra-diagonal elements in a :math:`\mathbb{R}^{D\times D}` matrix is :math:`D_{extra}=D(D+1)/2`. The method
        returns an operator which is:

        * if ``trace_term`` is ``True`` a :py:class:`~pycsou.abc.arithmetic.LinOp` of shape :math:`(N_{tot}, N_{tot}D_{extra})`;
        * if ``trace_term`` is ``False`` a :py:class:`~pycsou.operator.linop.base.DiagonalOp` of shape :math:`(N_{tot}D, N_{tot}D)`.

        **Remark**
        The current implementation in the trace-based case (``trace_term`` is ``True``) relies on the fact that, for each pixel,
        both the Hessian and the diffusion tensor are symmetric.
        """
        if not self.frozen:
            u, e = self._eigendecompose_struct_tensor(arr)
            lambdas = self._compute_intensities(e)
            tensors = self._assemble_tensors(u, lambdas)
            # assemble block operator
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
                # assemble and return a LinOp(self.size, self.ndims*(self.ndims+1)*self.size/2)
                return pyblock.hstack(ops)
            else:
                # assemble and return a DiagonalOp(self.ndims*self.size,self.ndims*self.size)
                return pyblock.vstack(ops)
        else:
            return self.frozen_op


class DiffusionCoeffAnisoEdgeEnhancing(_DiffusionCoeffAnisotropic):
    r"""
    Edge-enhancing anisotropic diffusion coefficient, based on structure tensor [see `Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_].

    Notes
    -----

    Let us consider the two-dimensional case :math:`D=2`, where the input signal is an image :math:`\mathbf{f}\in\mathbb{R}^{N_0,N_1}`.
    We follow the notation from the documentation of :py:class:`~pycsou.operator.diffusion._DiffusionCoeffAnisotropic`. In the context of
    diffusion operators, operators of :py:class:`~pycsou.operator.diffusion.DiffusionCoeffAnisoEdgeEnhancing` can be
    used to enhance the edges in the image.
    Let us consider the :math:`i`-th pixel of the image. The edge enhancing effect is achieved by the following choice
    of smoothing intensities associated to the eigenvalues of the structure tensor :math:`S_i`:

    .. math::
        \lambda_0 = g(e_0),\\
        \lambda_1 = 1,

    with

    .. math::
        g(e_0) :=
       \begin{cases}
           1 & \text{if } e_0 \leq 0 \\
           1 - \exp\big(\frac{-C}{(e_0/\beta)^m}\big) & \text{if } e_0 >0,
       \end{cases}

    where :math:`\beta` is a contrast parameter, :math:`m` controls the decay rate of :math:`\lambda_0` as a function
    of :math:`e_0`, and :math:`C\in\mathbb{R}` is a constant.

    The edge enhancement is achieved by reducing the smoothing intensity in the first eigendirection (connected to the direction
    of largest variation of :math:`\mathbf{f}`, thus perpendicular ot the edges) for large values of the first
    eigenvalue of :math:`S_i` (the contrast in the first eigendirection, connected to the magnitude of the gradient).

    **Remark 1**

    Currently, only two-dimensional case :math:`D=2` is handled. Need to implement rules to compute intensity for case :math:`D>2`.

    **Remark 2**

    Performance of the method can be quite sensitive to the hyperparameters :math:`\beta, m`, particularly :math:`\beta`.
    """

    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        structure_tensor: pyct.OpT,
        trace_term: bool = False,
        beta: pyct.Real = 1.0,
        m: pyct.Real = 4.0,
    ):
        """

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array.
        structure_tensor: :py:class:`~pycsou.operator.linop.diff.StructureTensor`
            Structure tensor operator.
        trace_term: bool
            Whether diffusion coefficient is meant to be used in a trace formulation or not.
            Defaults to `False`. Method ``.apply()`` acts differently depending on value of `trace_term`.
        beta: pyct.Real
            Contrast parameter. Defaults to `1`.
        m: pyct.Real
            Decay rate in intensity expression. Defaults to `4`.
        """
        assert len(arg_shape) == 2, "`arg_shape` has more than two dimensions, not handled yet"
        super().__init__(arg_shape=arg_shape, structure_tensor=structure_tensor, trace_term=trace_term)
        assert beta > 0, "contrast parameter `beta` must be strictly positive"
        self.beta = beta
        assert m > 0, "decay rate `m` must be strictly positive"
        self.m = 4
        self.bounded = True

        # compute normalization constant c [see `Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_]
        def f(c: pyct.Real, m: pyct.Integer):
            return 1 - np.exp(-c) * (1 + 2 * m * c)

        self.c = sciop.brentq(functools.partial(f, m), 1e-2, 100)

    @pycrt.enforce_precision(i="eigval_struct")
    def _compute_intensities(self, eigval_struct: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(eigval_struct)
        lambdas = xp.ones(eigval_struct.shape, dtype=eigval_struct.dtype)
        nonzero_contrast_locs = ~xp.isclose(eigval_struct[:, 0], 0)
        # Inplace implementation of
        #   lambdas[nonzero_contrast_locs, 0] = 1 - xp.exp(- self.c / ((eigval_struct[nonzero_contrast_locs, 0] / self.beta) ** self.m))
        lambda0_nonzerolocs = eigval_struct[nonzero_contrast_locs, 0]
        lambda0_nonzerolocs /= self.beta
        lambda0_nonzerolocs **= -self.m
        lambda0_nonzerolocs *= -self.c
        lambda0_nonzerolocs = -xp.exp(lambda0_nonzerolocs)
        lambda0_nonzerolocs += 1
        lambdas[nonzero_contrast_locs, 0] = lambda0_nonzerolocs
        return lambdas


class DiffusionCoeffAnisoCoherenceEnhancing(_DiffusionCoeffAnisotropic):
    r"""
    Coherence-enhancing anisotropic diffusion coefficient, based on structure tensor [see `Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_].

    Notes
    -----

    Let us consider the two-dimensional case :math:`D=2`, where the input signal is an image :math:`\mathbf{f}\in\mathbb{R}^{N_0,N_1}`.
    We follow the notation from the documentation of :py:class:`~pycsou.operator.diffusion._DiffusionCoeffAnisotropic`. In the context of
    diffusion operators, operators of :py:class:`~pycsou.operator.diffusion.DiffusionCoeffAnisoCoherenceEnhancing` can be
    used to enhance the coherence in the image.
    Let us consider the :math:`i`-th pixel of the image. The coherence enhancing effect is achieved by the following choice
    of smoothing intensities associated to the eigenvalues of the structure tensor :math:`S_i`:

    .. math::
        \lambda_0 = \alpha,\\
        \lambda_1 = h(e_0, e_1),

    with

    .. math::
        h(e_0, e_1) :=
       \begin{cases}
           \alpha & \text{if } e_0=e_1 \\
           \alpha + (1-\alpha) \exp \big(\frac{-C}{(e_0-e_1)^{2m}}\big) & \text{otherwise},
       \end{cases}

    where :math:`\alpha` controls the smoothing intensity in first eigendirection, :math:`m` controls the decay
    rate of :math:`\lambda_0` as a function of :math:`(e_0-e_1)`, and :math:`C\in\mathbb{R}` is a constant.

    The coherence enhancement is achieved by increasing the smoothing intensity in the second eigendirection
    (connected to the direction of smallest variation of :math:`\mathbf{f}`, thus parallel ot the edges) for large values
    of the coherence, measured as :math:`(e_0-e_1)^2`.

    **Remark 1**

    Currently, only two-dimensional case :math:`D=2` is handled. Need to implement rules to compute intensity for case :math:`D>2`.

    **Remark 2**

    Performance of the method can be quite sensitive to the hyperparameters :math:`\alpha, m`, particularly :math:`\alpha`.
    """

    def __init__(
        self, arg_shape: pyct.NDArrayShape, structure_tensor: pyct.OpT, trace_term: bool = False, alpha=0.1, m=1
    ):
        """

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array.
        structure_tensor: :py:class:`~pycsou.operator.linop.diff.StructureTensor`
            Structure tensor operator.
        trace_term: bool
            Whether diffusion coefficient is meant to be used in a trace formulation or not.
            Defaults to `False`. Method ``.apply()`` acts differently depending on value of `trace_term`.
        alpha: pyct.Real
            Smoothing intensity in first eigendirection. Defaults to `0.1`.
        m: pyct.Real
            Decay rate in intensity expression. Defaults to `1`.
        """
        assert len(arg_shape) == 2, "`arg_shape` has more than two dimensions, not handled yet"
        super().__init__(arg_shape=arg_shape, structure_tensor=structure_tensor, trace_term=trace_term)
        assert alpha > 0, "intensity parameter `alpha` must be strictly positive"
        self.alpha = alpha
        assert m > 0, "decay rate `m` must be strictly positive"
        self.m = m
        # constant C set to 1 for now
        self.c = 1.0
        self.bounded = True

    @pycrt.enforce_precision(i="eigval_struct")
    def _compute_intensities(self, eigval_struct: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(eigval_struct)
        coherence = (eigval_struct[:, 0] - eigval_struct[:, 1]) ** 2
        lambdas = self.alpha * xp.ones(eigval_struct.shape, dtype=eigval_struct.dtype)
        nonzero_coherence_locs = ~np.isclose(coherence, 0)
        # Inplace implementation of
        #   lambdas[nonzero_coherence_locs, 1] = self.alpha + (1-self.alpha)*np.exp(-1./(coherence[nonzero_coherence_locs] ** (2*self.m)))
        lambda1_nonzerolocs = coherence[nonzero_coherence_locs]
        lambda1_nonzerolocs **= -(2 * self.m)
        lambda1_nonzerolocs *= -self.c
        lambda1_nonzerolocs = xp.exp(lambda1_nonzerolocs)
        lambda1_nonzerolocs *= 1 - self.alpha
        lambda1_nonzerolocs += self.alpha
        lambdas[nonzero_coherence_locs, 1] = lambda1_nonzerolocs
        return lambdas


class _DiffusionOp(pyca.ProxDiffFunc):
    r"""
    Abstract class for diffusion operators
    [see `Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_ and `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_].

    Notes
    -----

    This class provides an interface to deal with PDE-based regularisation. For simplicity, throughout the
    documentation we consider a :math:`2`-dimensional signal :math:`\mathbf{f} \in \mathbb{R}^{N_{0},N_1}`,
    but higher dimensional signals could be considered. We denote by :math:`f_i` the :math:`i`-th entry (pixel)
    of the vectorisation of :math:`\mathbf{f}`, :math:`i=0,\dots,(N_0N_1-1)`. Furthermore, let
    :math:`(\nabla \mathbf{f})_i = \big((\frac{\partial \mathbf{f}}{\partial x_1})_i,\dots,(\frac{\partial \mathbf{f}}{\partial x_{D-1}})_i\big)^T`.

    To give some intuition, let us first consider a simple case: Tikhonov regularisation, corresponding to linear, isotropic,
    homogeneous smoothing. This yields the regularisation functional

    .. math::
        \phi(\mathbf{f}) = \frac{1}{2}\Vert \nabla \mathbf{f} \Vert_2^2
                         = \frac{1}{2}\sum_{i=0}^{N_0 N_1-1} \Vert (\nabla \mathbf{f})_i \Vert_2^2.

    Then, we have

    .. math::
        \nabla \phi(\mathbf{f}) = \nabla^T\nabla\mathbf{f}
                                = -\text{div}(\nabla\mathbf{f})

    where :math:`\nabla^T` is the adjoint of the gradient operator and where we exploited the fact that
    :math:`\nabla^T = -\text{div}`, the divergence. If we now wanted to solve the
    optimization problem

    .. math::
        \text{argmin}_\mathbf{f}\phi(\mathbf{f}),

    we could apply gradient descent starting from an initial state :math:`\mathbf{f}_0`, so that

    .. math::
        \mathbf{f}_1 = \mathbf{f}_0 + \eta \text{div}(\nabla\mathbf{f}_0),

    where :math:`\eta` represents the step size of the algorithm. The above update equation can be interpreted as
    one step in time of the explicit Euler integration method applied to the PDE

    .. math::
        \frac{\partial\mathbf{f}}{\partial t} = \text{div}(\nabla\mathbf{f})
                                              = \Delta \mathbf{f},

    with initial condition :math:`\mathbf{f}=\mathbf{f}_0\,\text{for }t=0` and time step size :math:`\Delta t=\eta`.
    This time-dependent PDE represents the gradient flow formulation of the original optimization problem, where by
    `time` we refer to an artificial time characterising the optimization process. We recognise, moreover, that
    we actually obtained the well-known heat equation.

    We can thus let the PDE evolve in time until it reaches a steady-state :math:`\frac{\partial\mathbf{f}}{\partial t}=0`.
    The solution will therefore satisfy the first order optimality condition :math:`\nabla \phi(\mathbf{f})=0`.

    If formulated as above, a trivial steady-state corresponding to an infinitely flat solution will be obtained.
    However, if the functional :math:`\phi(\cdot)` is combined with a data-fidelity functional :math:`\ell(\cdot)`
    in the context of an inverse problem, an extra term :math:`\nabla \ell(\cdot)` will arise in the gradient flow
    formulation. This will lead to a non-trivial steady-state representing the equilibrium between the data-fidelity
    and regularisation term.

    In the context of PDE-based regularisation, it is not necessary to limit ourselves to consider cases where it is possible
    to explicitly define a variational functional :math:`\phi(\cdot)`. In the spirit of Plug&Play (PnP) approaches,
    we can consider diffusion operators that are only characterised by their smoothing action in gradient flow form: no
    underlying functional :math:`\phi(\cdot)` may exist. This allows to study complex diffusion processes designed to
    enhance specific features of the image.

    In particular, we consider diffusion processes that, in their most general form, can be written as the composite term

    .. math::
        \frac{\partial\mathbf{f}}{\partial t} = \mathbf{D}_{out}\text{div}(\mathbf{D}_{in}\nabla\mathbf{f})
        + \mathbf{B} + \mathbf{T}_{out}\text{trace}\big(\mathbf{T}_{in}\mathbf{H}(\mathbf{f})\big) + (\nabla\mathbf{f})^T \mathbf{J}_{\mathbf{w}}\mathbf{w},

    where
        * :math:`\mathbf{D}_{out} = \mathbf{D}_{out}(\mathbf{f})` is the outer diffusivity for the divergence term;
        * :math:`\mathbf{D}_{in} = \mathbf{D}_{in}(\mathbf{f})` is the diffusion coefficient for the divergence term;
        * :math:`\mathbf{B} = \mathbf{B}(\mathbf{f})` is the balloon force;
        * :math:`\mathbf{T}_{out} = \mathbf{T}_{out}(\mathbf{f})` is the outer diffusivity for the trace term;
        * :math:`\mathbf{T}_{in} = \mathbf{T}_{in}(\mathbf{f})` is the diffusion coefficient for the trace term;
        * :math:`\mathbf{w}` is a vector field assigning a :math:`2`-dimensional vector to each pixel;
        * :math:`\mathbf{J}_\mathbf{w}` is the Jacobian of the vector field :math:`\mathbf{w}`.

    The right-hand side of the above PDE represents the output of the ``.grad()`` method applied to the image :math:`\mathbf{f}`.

    To conclude, we remark that the action of the diffusion operator on an image :math:`\mathbf{f}` can be better understood
    focusing on a single pixel :math:`f_i` of the vectorisation of :math:`\mathbf{f}` (see, e.g., discussion in
    :py:class:`~pycsou.operator.diffusion._DiffusionCoefficient`).

    **Remark 1**

    ``_DiffusionOp`` represents an atypical :py:class:`~pycsou.abc.operator.ProxDiffFunc`. Indeed,
    the ``.apply()`` method is not necessarily defined, in the case of implicitly defined functionals.
    The key method is ``.grad()``, necessary to perform gradient flow optimization (and also used by
    ``.prox()``).

    **Remark 2**

    The ``.apply()`` method raises a ``NotImplementedError`` unless the diffusion term is known to derive from
    a variational formulation. Currently, only the case where :math:`\mathbf{D}_{in}` is a member of
    :py:class:`~pycsou.operator.diffusion.DiffusionCoeffIsotropic` and all other diffusion coefficients/diffusivities
    are `None` may detect an underlying variational formulation. Other cases exist but are not treated for now.

    **Remark 3**

    This class is not meant to be directly used, hence the underscore ``_DiffusionOp`` signalling it is private.
    In principle, users should rely on the daughter classes :py:class:`~pycsou.operator.diffusion.DivergenceDiffusionOp`,
    :py:class:`~pycsou.operator.diffusion.SnakeDiffusionOp`, :py:class:`~pycsou.operator.diffusion.TraceDiffusionOp`,
    :py:class:`~pycsou.operator.diffusion.CurvaturePreservingDiffusionOp`.


    Developer Notes
    ---------------
    * In method ``.grad()``, to avoid using the @vectorize decorator, all ``_compute()`` functions should be changed, suitably stacking
      all operators and results along the stacking dimensions. For now this has not been done, to be discussed if less naif
      vectorisation is important. It would be cumbersome especially for the terms involving diffusion coefficients, whose
      ``.apply()`` method returns a pycsou operator.

    * For now, user is meant to initialize independently all the building blocks and provide them at initialization of a
      diffusion operator. We could provide, of course, simpler interfaces for some of the most standard diffusion operators.
      Still, even in current form, module should hopefully be relatively simple to use when provided with some examples.
    """

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
        r"""

        Parameters
        ----------
        arg_shape: pyct.Shape
            Shape of the input array.
        gradient: :py:class:`~pycsou.operator.linop.diff.Gradient`
            Gradient operator. Defaults to `None`.
        hessian:  :py:class:`~pycsou.operator.linop.diff.Hessian`
            Hessian operator. Defaults to `None`.
        outer_diffusivity: :py:class:`~pycsou.operator.diffusion._Diffusivity`
            Outer diffusivity operator, to be applied to the divergence term.
        diffusion_coefficient: :py:class:`~pycsou.operator.diffusion._DiffusionCoefficient`
            Diffusion coefficient operator, featured inside divergence term.
        balloon_force: :py:class:`~pycsou.operator.diffusion._BalloonForce`
            Balloon force operator.
        outer_trace_diffusivity: :py:class:`~pycsou.operator.diffusion._Diffusivity`
            Outer diffusivity operator, to be applied to the trace term.
        trace_diffusion_coefficient: :py:class:`~pycsou.operator.diffusion._DiffusionCoefficient`
            Diffusion coefficient operator, featured inside trace term.
        curvature_preservation_field: pyct.NDArray
            Vector field along which curvature should be preserved. Defaults to `np.zeros(0)`.
        prox_sigma: pyct.Real
            Size of the structures that should be smoothed out by prox computation (when implemented in PnP-fashion).

        Notes
        ----
        The parameter ``prox_sigma`` is expressed in pixel units. Independently of the ``sampling`` of the differential
        operators involved, ``prox_sigma`` equal to `1` is meant to yield a prox which smoothes structures of the order
        of one pixel.
        """
        self.arg_shape = arg_shape
        self.size = int(np.prod(arg_shape))
        self.ndims = len(self.arg_shape)
        super().__init__(shape=(1, self.size))
        # sanitize inputs
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
            assert hessian.codim == expected_codim, '`hessian` expected to be initialized with `directions`="all"'

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

        if diffusion_coefficient and diffusion_coefficient.trace_term:
            if not diffusion_coefficient.frozen:
                warnings.warn("`diffusion_coefficient.trace_term` set to True. Modifying to False.")
                diffusion_coefficient.trace_term = True
            else:
                msg = "\n".join(
                    [
                        "`diffusion_coefficient.trace_term` set to True and `diffusion_coefficient.frozen` set to True.",
                        "Issues are expected. Initialize correctly `diffusion_coefficient.trace_term` to False before freezing.",
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
                    ops.append(hessian._block[(dim, dim)])
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
        r"""
        Notes
        -------
        Divergence-based diffusion operators may arise from a variational formulation. This is true, e.g.,
        for the isotropic Perona-Malik, TV, Tikhonov. For these cases, it is possible
        to define the associated energy functional. When no variational formulation is detected, the method raises an error.
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
        r"""

        Notes
        -----
        Let :math:`\phi(\cdot)` be the functional (it could be defined only implicitly) underlying
        the diffusion operator. The actual prox operator would read

        .. math::
            \text{prox}_{\tau \phi}(arr) = \text{argmin}_{y}\frac{1}{2\tau}\Vert arr - y\Vert _2^2+\phi(y).

        If we were to compute the prox solution by means of gradient descent, the :math:`n`-th iteration of the
        algorithm would take steps in the direction

        .. math::
            \frac{1}{\tau}(y_n - arr)-\nabla\phi(y_n).

        The prox applied to `arr` can be interpreted as a denoised version of `arr` [see `Romano <https://arxiv.org/pdf/1611.02862.pdf>`_],
        where the data-fidelity term :math:`\frac{1}{2*\tau}\Vert arr - y\Vert _2^2` ensures that the result does not
        get too far from `arr` and the regularisation term :math:`\phi(y)` tries to make the image smoother.

        In our setting :math:`\nabla\phi(\cdot)` would correspond to ``.grad()``.

        In a Plug&Play (PnP) spirit [see `Romano <https://arxiv.org/pdf/1611.02862.pdf>`_], we
        replace the solution of the prox problem by the activation of a denoising engine consisting
        in performing a fixed number of ``.grad()`` steps. This approach allows us to:

        * bypass the problem of the explicit definition of the regularisation prior :math:`\phi(\cdot)`;
        * have a prox operator that can be evaluated at a fixed cost (the number of ``.grad()`` calls
          chosen, i.e., ``prox_steps``), which does not depend on the number of iterations needed to
          achieve convergence as in the actual prox computation.

        This denoising approach relies on the scale-space interpretation of diffusion operators
        [see `Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_], according to which
        denoising at a characteristic noise scale :math:`\sigma` can be achieved by stopping the diffusion
        at a given time :math:`T`. In the linear isotropic diffusion case where ``.grad()`` is the
        Laplacian, smoothing structures of order :math:`\sigma` is achieved stopping the diffusion
        process at :math:`T=\frac{\sigma^2}{2}`. Following the linear diffusion analogy,
        the stopping time for prox computation is computed as :math:`T=\frac{\text{prox\_sigma}^2}{2}`,
        where ``prox_sigma`` is provided at initialization. Better estimates of stopping time
        could/should be studied.
        """
        stop_crit = pystop.MaxIter(self.prox_steps)
        pgd = pysol.PGD(f=self, g=None, show_progress=False, verbosity=100)
        pgd.fit(**dict(mode=pysolver.Mode.BLOCK, x0=arr, stop_crit=stop_crit, acceleration=False))
        return pgd.solution()


class DivergenceDiffusionOp(_DiffusionOp):
    r"""
    Class for divergence-based diffusion operators [see `Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_].

    This class provides an interface to deal with divergence-based diffusion operators in the context of PDE-based regularisation.
    In particular, we consider diffusion processes that can be written as

    .. math::
        \frac{\partial\mathbf{f}}{\partial t} = \mathbf{D}_{out}\text{div}(\mathbf{D}_{in}\nabla\mathbf{f}),

    where
        * :math:`\mathbf{D}_{out} = \mathbf{D}_{out}(\mathbf{f})` is the outer diffusivity for the divergence term;
        * :math:`\mathbf{D}_{in} = \mathbf{D}_{in}(\mathbf{f})` is the diffusion coefficient for the divergence term.

    The right-hand side of the above PDE represents the output of the ``.grad()`` method applied to the image :math:`\mathbf{f}`.

    The action of the :py:class:`~pycsou.operator.diffusion.DivergenceDiffusionOp` on an image :math:`\mathbf{f}` can be better understood
    focusing on a single pixel :math:`f_i` of the vectorisation of :math:`\mathbf{f}` (see, e.g., discussion in
    :py:class:`~pycsou.operator.diffusion._DiffusionCoefficient`).
    """

    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        gradient: pyct.OpT = None,
        outer_diffusivity: pyct.OpT = None,
        diffusion_coefficient: pyct.OpT = None,
        prox_sigma: pyct.Real = 2,
    ):
        r"""

        Parameters
        ----------
        arg_shape: pyct.Shape
            Shape of the input array.
        gradient: :py:class:`~pycsou.operator.linop.diff.Gradient`
            Gradient operator. Defaults to `None`.
        outer_diffusivity: :py:class:`~pycsou.operator.diffusion._Diffusivity`
            Outer diffusivity operator, to be applied to the divergence term.
        diffusion_coefficient: :py:class:`~pycsou.operator.diffusion._DiffusionCoefficient`
            Diffusion coefficient operator, featured inside divergence term.
        prox_sigma: pyct.Real
            Size of the structures that should be smoothed out by prox computation (when implemented in PnP-fashion).

        Notes
        ----
        The parameter ``prox_sigma`` is expressed in pixel units. Independently of the ``sampling`` of the differential
        operators involved, ``prox_sigma`` equal to `1` is meant to yield a prox which smoothes structures of the order
        of one pixel.
        """
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
        return y


class SnakeDiffusionOp(_DiffusionOp):
    r"""
    Class for snake diffusion operators (active contour models) [see `Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_].

    This class provides an interface to deal with snake diffusion operators in the context of PDE-based regularisation.
    In particular, we consider diffusion processes that can be written as

    .. math::
        \frac{\partial\mathbf{f}}{\partial t} = \mathbf{D}_{out}\text{div}(\mathbf{D}_{in}\nabla\mathbf{f})+ \mathbf{B},
    where
        * :math:`\mathbf{D}_{out} = \mathbf{D}_{out}(\mathbf{f})` is the outer diffusivity for the divergence term;
        * :math:`\mathbf{D}_{in} = \mathbf{D}_{in}(\mathbf{f})` is the diffusion coefficient for the divergence term;
        * :math:`\mathbf{B} = \mathbf{B}(\mathbf{f})` is the balloon force.

    The right-hand side of the above PDE represents the output of the ``.grad()`` method applied to the image :math:`\mathbf{f}`.

    The action of the :py:class:`~pycsou.operator.diffusion.SnakeDiffusionOp` on an image :math:`\mathbf{f}` can be better understood
    focusing on a single pixel :math:`f_i` of the vectorisation of :math:`\mathbf{f}` (see, e.g., discussion in
    :py:class:`~pycsou.operator.diffusion._DiffusionCoefficient`).
    """

    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        gradient: pyct.OpT = None,
        outer_diffusivity: pyct.OpT = None,
        diffusion_coefficient: pyct.OpT = None,
        balloon_force: pyct.OpT = None,
        prox_sigma: pyct.Real = 2,
    ):
        r"""

        Parameters
        ----------
        arg_shape: pyct.Shape
            Shape of the input array.
        gradient: :py:class:`~pycsou.operator.linop.diff.Gradient`
            Gradient operator. Defaults to `None`.
        outer_diffusivity: :py:class:`~pycsou.operator.diffusion._Diffusivity`
            Outer diffusivity operator, to be applied to the divergence term.
        diffusion_coefficient: :py:class:`~pycsou.operator.diffusion._DiffusionCoefficient`
            Diffusion coefficient operator, featured inside divergence term.
        balloon_force: :py:class:`~pycsou.operator.diffusion._BalloonForce`
            Balloon force operator.
        prox_sigma: pyct.Real
            Size of the structures that should be smoothed out by prox computation (when implemented in PnP-fashion).

        Notes
        ----
        The parameter ``prox_sigma`` is expressed in pixel units. Independently of the ``sampling`` of the differential
        operators involved, ``prox_sigma`` equal to `1` is meant to yield a prox which smoothes structures of the order
        of one pixel.
        """
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
        return y


class TraceDiffusionOp(_DiffusionOp):
    r"""
    Class for trace-based diffusion operators [see `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_].

    This class provides an interface to deal with trace-based diffusion operators in the context of PDE-based regularisation.
    In particular, we consider diffusion processes that can be written as

    .. math::
        \frac{\partial\mathbf{f}}{\partial t} = \mathbf{T}_{out}\text{trace}\big(\mathbf{T}_{in}\mathbf{H}(\mathbf{f})\big),

    where
        * :math:`\mathbf{T}_{out} = \mathbf{T}_{out}(\mathbf{f})` is the outer diffusivity for the trace term;
        * :math:`\mathbf{T}_{in} = \mathbf{T}_{in}(\mathbf{f})` is the diffusion coefficient for the trace term.

    The right-hand side of the above PDE represents the output of the ``.grad()`` method applied to the image :math:`\mathbf{f}`.

    The action of the :py:class:`~pycsou.operator.diffusion.TraceDiffusionOp` on an image :math:`\mathbf{f}` can be better understood
    focusing on a single pixel :math:`f_i` of the vectorisation of :math:`\mathbf{f}` (see, e.g., discussion in
    :py:class:`~pycsou.operator.diffusion._DiffusionCoefficient`).
    """

    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        hessian: pyct.OpT = None,
        outer_trace_diffusivity: pyct.OpT = None,
        trace_diffusion_coefficient: pyct.OpT = None,
        prox_sigma: pyct.Real = 2,
    ):
        r"""

        Parameters
        ----------
        arg_shape: pyct.Shape
            Shape of the pixelised image.
        hessian:  :py:class:`~pycsou.operator.linop.diff.Hessian`
            Hessian operator. Defaults to `None`.
        outer_trace_diffusivity: :py:class:`~pycsou.operator.diffusion._Diffusivity`
            Outer diffusivity operator, to be applied to the trace term.
        trace_diffusion_coefficient: :py:class:`~pycsou.operator.diffusion._DiffusionCoefficient`
            Diffusion coefficient operator, featured inside trace term.
        prox_sigma: pyct.Real
            Size of the structures that should be smoothed out by prox computation (when implemented in PnP-fashion).

        Notes
        ----
        The parameter ``prox_sigma`` is expressed in pixel units. Independently of the ``sampling`` of the differential
        operators involved, ``prox_sigma`` equal to `1` is meant to yield a prox which smoothes structures of the order
        of one pixel.
        """
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
        return y


class CurvaturePreservingDiffusionOp(_DiffusionOp):
    r"""
    Class for curvature preserving diffusion operators [see `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_].

    This class provides an interface to deal with curvature preserving diffusion operators in the context of PDE-based regularisation.
    In particular, we consider diffusion processes that can be written as

    .. math::
        \frac{\partial\mathbf{f}}{\partial t} = \mathbf{T}_{out}\text{trace}\big(\mathbf{T}_{in}\mathbf{H}(\mathbf{f})\big) + (\nabla\mathbf{f})^T \mathbf{J}_{\mathbf{w}}\mathbf{w},

    where
        * :math:`\mathbf{T}_{out} = \mathbf{T}_{out}(\mathbf{f})` is the outer diffusivity for the trace term;
        * :math:`\mathbf{T}_{in} = \mathbf{T}_{in}(\mathbf{f})` is the diffusion coefficient for the trace term;
        * :math:`\mathbf{w}` is a vector field assigning a :math:`2`-dimensional vector to each pixel;
        * :math:`\mathbf{J}_\mathbf{w}` is the Jacobian of the vector field :math:`\mathbf{w}`.

    The right-hand side of the above PDE represents the output of the ``.grad()`` method applied to the image :math:`\mathbf{f}`.

    The resulting smoothing process tries to preserve the curvature of the vector field :math:`\mathbf{w}`.

    The action of the :py:class:`~pycsou.operator.diffusion.CurvaturePreservingDiffusionOp` on an image :math:`\mathbf{f}` can be better understood
    focusing on a single pixel :math:`f_i` of the vectorisation of :math:`\mathbf{f}` (see, e.g., discussion in
    :py:class:`~pycsou.operator.diffusion._DiffusionCoefficient`).
    """

    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        gradient: pyct.OpT = None,
        hessian: pyct.OpT = None,
        curvature_preservation_field: pyct.NDArray = np.zeros(0),
        prox_sigma: pyct.Real = 2,
    ):
        r"""

        Parameters
        ----------
        arg_shape: pyct.Shape
            Shape of the pixelised image.
        gradient: :py:class:`~pycsou.operator.linop.diff.Gradient`
            Gradient operator. Defaults to `None`.
        hessian:  :py:class:`~pycsou.operator.linop.diff.Hessian`
            Hessian operator. Defaults to `None`.
        curvature_preservation_field: pyct.NDArray
            Vector field along which curvature should be preserved. Defaults to `np.zeros(0)`.
        prox_sigma: pyct.Real
            Size of the structures that should be smoothed out by prox computation (when implemented in PnP-fashion).

        Notes
        ----
        The parameter ``prox_sigma`` is expressed in pixel units. Independently of the ``sampling`` of the differential
        operators involved, ``prox_sigma`` equal to `1` is meant to yield a prox which smoothes structures of the order
        of one pixel.
        """
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
        # assemble trace diffusion coefficient corresponding to the curvature preserving field
        curvature_preservation_field = curvature_preservation_field.T
        tensors = curvature_preservation_field.reshape(self.size, self.ndims, 1) * curvature_preservation_field.reshape(
            self.size, 1, self.ndims
        )
        trace_diffusion_coefficient = tensors.reshape(self.size, -1)
        ops = []
        for i in range(self.ndims):
            # only upper diagonal entries are considered (symmetric tensors)
            first_idx = i
            for j in np.arange(first_idx, self.ndims):
                op = pybase.DiagonalOp(trace_diffusion_coefficient[:, i * self.ndims + j])
                if j > i:
                    # multiply by 2 extra diagonal elements
                    op *= 2.0
                ops.append(op)
        self.trace_diffusion_coefficient = _DiffusionCoefficient(arg_shape=arg_shape, isotropic=False, trace_term=True)
        self.trace_diffusion_coefficient.set_frozen_op(pyblock.hstack(ops))
        # estimate diff_lipschitz
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
        return y
