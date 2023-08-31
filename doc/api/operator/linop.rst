pyxu.operator.linop
===================

.. contents:: Table of Contents
   :local:
   :depth: 1

Basic operators
---------------

.. autoclass:: pyxu.operator.linop.select.SubSample
   :no-members:
   :members: apply, adjoint, TrimSpec, IndexSpec
   :special-members: __init__

.. autofunction:: pyxu.operator.linop.select.Trim

.. autofunction:: pyxu.operator.linop.reduce.Sum

.. autofunction:: pyxu.operator.linop.kron.kron

.. autofunction:: pyxu.operator.linop.kron.khatri_rao

.. autoclass:: pyxu.operator.linop.base.IdentityOp
   :no-members:

.. autoclass:: pyxu.operator.linop.base.NullOp
   :no-members:

.. autofunction:: pyxu.operator.linop.base.NullFunc

.. autofunction:: pyxu.operator.linop.base.HomothetyOp

.. autofunction:: pyxu.operator.linop.base.DiagonalOp

.. autofunction:: pyxu.operator.linop.base._ExplicitLinOp

.. autoclass:: pyxu.operator.linop.pad.Pad
   :no-members:
   :members: WidthSpec, ModeSpec
   :special-members: __init__


Stencils and Convolutions
-------------------------

.. autoclass:: pyxu.operator.linop.stencil._stencil._Stencil
   :no-members:
   :members: init, apply, IndexSpec

.. autoclass:: pyxu.operator.linop.stencil.stencil.Stencil
   :no-members:
   :members: KernelSpec, configure_dispatcher, kernel, center, relative_indices, visualize
   :special-members: __init__

.. autoclass:: pyxu.operator.linop.stencil.stencil.Correlate

.. autofunction:: pyxu.operator.linop.stencil.stencil.Convolve

Transforms
----------

.. autoclass:: pyxu.operator.linop.fft.fft.FFT
   :no-members:
   :members: apply, adjoint
   :special-members: __init__

.. autoclass:: pyxu.operator.linop.fft.nufft.NUFFT
   :no-members:
   :members: type1, type2, type3, apply, adjoint, ascomplexarray, mesh, plot_kernel, params, auto_chunk, allocate, diagnostic_plot, stats

Derivatives
-----------

.. autoclass:: pyxu.operator.linop.diff.PartialDerivative
   :members: finite_difference, gaussian_derivative

.. autofunction:: pyxu.operator.linop.diff.Gradient

.. autofunction:: pyxu.operator.linop.diff.Jacobian

.. autofunction:: pyxu.operator.linop.diff.Divergence

.. autofunction:: pyxu.operator.linop.diff.Hessian

.. autofunction:: pyxu.operator.linop.diff.Laplacian

.. autofunction:: pyxu.operator.linop.diff.DirectionalDerivative

.. autofunction:: pyxu.operator.linop.diff.DirectionalGradient

.. autofunction:: pyxu.operator.linop.diff.DirectionalLaplacian

.. autofunction:: pyxu.operator.linop.diff.DirectionalHessian

Filters
-------

.. autofunction:: pyxu.operator.linop.filter.MovingAverage

.. autofunction:: pyxu.operator.linop.filter.Gaussian

.. autofunction:: pyxu.operator.linop.filter.DifferenceOfGaussians

.. autofunction:: pyxu.operator.linop.filter.DoG

.. autofunction:: pyxu.operator.linop.filter.Laplace

.. autofunction:: pyxu.operator.linop.filter.Sobel

.. autofunction:: pyxu.operator.linop.filter.Prewitt

.. autofunction:: pyxu.operator.linop.filter.Scharr
