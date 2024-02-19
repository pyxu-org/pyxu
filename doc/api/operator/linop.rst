pyxu.operator.linop
===================

.. contents:: Table of Contents
   :local:
   :depth: 2

Basic Operators
---------------

.. autoclass:: pyxu.operator.SubSample
   :no-members:
   :members: apply, adjoint, TrimSpec, IndexSpec
   :special-members: __init__

.. autofunction:: pyxu.operator.Trim

.. autoclass:: pyxu.operator.Sum

.. autoclass:: pyxu.operator.IdentityOp
   :no-members:

.. autoclass:: pyxu.operator.NullOp
   :no-members:

.. autofunction:: pyxu.operator.NullFunc

.. autofunction:: pyxu.operator.HomothetyOp

.. autofunction:: pyxu.operator.DiagonalOp

.. autoclass:: pyxu.operator.Pad
   :no-members:
   :members: WidthSpec, ModeSpec
   :special-members: __init__

Transforms
----------

.. autoclass:: pyxu.operator.FFT
   :no-members:
   :members: apply, adjoint
   :special-members: __init__

.. autoclass:: pyxu.operator.CZT
   :no-members:
   :members: apply, adjoint
   :special-members: __init__

.. autoclass:: pyxu.operator.NUFFT1
   :no-members:
   :members: apply, adjoint, config

.. autofunction:: pyxu.operator.NUFFT2

Stencils & Convolutions
-----------------------

.. autoclass:: pyxu.operator._Stencil
   :no-members:
   :members: init, apply, IndexSpec

.. autoclass:: pyxu.operator.Stencil
   :no-members:
   :members: KernelSpec, configure_dispatcher, kernel, center, relative_indices, visualize
   :special-members: __init__

.. autoclass:: pyxu.operator.Correlate

.. autoclass:: pyxu.operator.Convolve

.. autoclass:: pyxu.operator.FFTCorrelate

.. autoclass:: pyxu.operator.FFTConvolve

.. autoclass:: pyxu.operator.UniformSpread

Filters
-------

.. autofunction:: pyxu.operator.MovingAverage

.. autofunction:: pyxu.operator.Gaussian

.. autofunction:: pyxu.operator.DifferenceOfGaussians

.. autofunction:: pyxu.operator.DoG

.. autofunction:: pyxu.operator.Laplace

.. autofunction:: pyxu.operator.Sobel

.. autofunction:: pyxu.operator.Prewitt

.. autofunction:: pyxu.operator.Scharr

.. autoclass:: pyxu.operator.StructureTensor

Derivatives
-----------

.. autoclass:: pyxu.operator.PartialDerivative
   :members: finite_difference, gaussian_derivative

.. autofunction:: pyxu.operator.Gradient

.. autofunction:: pyxu.operator.Jacobian

.. autofunction:: pyxu.operator.Divergence

.. autofunction:: pyxu.operator.Hessian

.. autofunction:: pyxu.operator.Laplacian

.. autofunction:: pyxu.operator.DirectionalDerivative

.. autofunction:: pyxu.operator.DirectionalGradient

.. autofunction:: pyxu.operator.DirectionalLaplacian

.. autofunction:: pyxu.operator.DirectionalHessian

Tensor Products
---------------

.. autofunction:: pyxu.operator.kron

.. autofunction:: pyxu.operator.khatri_rao
